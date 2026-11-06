import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import Categorical, Normal
import gymnasium as gym

torch.set_num_threads(1)


class VirtualBatchNorm1d(nn.Module):
    """
    True Virtual Batch Normalization (Salimans et al., 2017 Sec. 2.2):
    Normalizes layer activations using statistics computed from a fixed reference
    minibatch forwarded through the current perturbed network parameters.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("ref_mean", torch.zeros(1, num_features))
        self.register_buffer("ref_var", torch.ones(1, num_features))

    def update_reference_stats(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        self.ref_mean.copy_(mean)
        self.ref_var.copy_(var)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return ((x - self.ref_mean) / torch.sqrt(self.ref_var + self.eps)).squeeze(0)
        return (x - self.ref_mean) / torch.sqrt(self.ref_var + self.eps)


class OpenAIESPolicy(nn.Module):
    """
    PyTorch MLP Policy (two 64-unit tanh layers) with Observation Normalization
    and dynamic Virtual Batch Normalization.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        is_discrete: bool,
        act_low=None,
        act_high=None,
        hidden_sizes=(64, 64),
        use_vbn: bool = False,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.is_discrete = is_discrete
        self.act_low = act_low
        self.act_high = act_high
        self.use_vbn = use_vbn

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("ref_batch", None)

        self.fc1 = nn.Linear(obs_dim, hidden_sizes[0])
        self.vbn1 = VirtualBatchNorm1d(hidden_sizes[0]) if use_vbn else nn.Identity()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.vbn2 = VirtualBatchNorm1d(hidden_sizes[1]) if use_vbn else nn.Identity()
        self.out = nn.Linear(hidden_sizes[1], act_dim)

        if not is_discrete and requires_grad:
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        else:
            self.register_parameter("log_std", None)

        self._init_weights()
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad_(False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    @torch.no_grad()
    def initialize_vbn(self, ref_obs: np.ndarray) -> None:
        """Sets observation normalization statistics and stores the fixed VBN reference batch."""
        ref_t = torch.as_tensor(ref_obs, dtype=torch.float32)
        self.obs_mean.copy_(ref_t.mean(dim=0))
        self.obs_std.copy_(ref_t.std(dim=0) + 1e-6)
        self.ref_batch = (ref_t - self.obs_mean) / self.obs_std
        self._refresh_vbn_stats()

    @torch.no_grad()
    def _refresh_vbn_stats(self) -> None:
        """Recomputes Virtual BatchNorm statistics whenever flat parameters theta change."""
        if not self.use_vbn or self.ref_batch is None:
            return
        h1 = self.vbn1.update_reference_stats(self.fc1(self.ref_batch))
        h1_act = torch.tanh(h1)
        self.vbn2.update_reference_stats(self.fc2(h1_act))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.obs_mean) / self.obs_std
        h1 = torch.tanh(self.vbn1(self.fc1(x_norm)))
        h2 = torch.tanh(self.vbn2(self.fc2(h1)))
        return self.out(h2)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray | int:
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.forward_logits(x)
        if self.is_discrete:
            return int(torch.argmax(logits).item())
        action = torch.tanh(logits).numpy()
        if self.act_low is not None and self.act_high is not None:
            action = self.act_low + 0.5 * (action + 1.0) * (self.act_high - self.act_low)
        return action.astype(np.float32)

    def forward_distribution(self, obs: torch.Tensor):
        logits = self.forward_logits(obs)
        if self.is_discrete:
            return Categorical(logits=logits)
        mean = torch.tanh(logits)
        if self.act_low is not None and self.act_high is not None:
            low = torch.as_tensor(self.act_low, dtype=torch.float32, device=obs.device)
            high = torch.as_tensor(self.act_high, dtype=torch.float32, device=obs.device)
            mean = low + 0.5 * (mean + 1.0) * (high - low)
            scale = 0.5 * (high - low)
        else:
            scale = 1.0
        std = torch.exp(torch.clamp(self.log_std, -3.0, 0.5)) * scale
        return Normal(mean, std)

    def get_flat_params(self) -> np.ndarray:
        return parameters_to_vector(self.parameters()).detach().cpu().numpy().copy()

    def set_flat_params(self, flat_theta: np.ndarray) -> None:
        vec = torch.as_tensor(flat_theta, dtype=torch.float32)
        vector_to_parameters(vec, self.parameters())
        if self.use_vbn:
            self._refresh_vbn_stats()


def get_env_specs(env_id: str):
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        act_dim = int(env.action_space.n)
        act_low, act_high = None, None
    elif isinstance(env.action_space, gym.spaces.Box):
        is_discrete = False
        act_dim = int(np.prod(env.action_space.shape))
        act_low, act_high = env.action_space.low, env.action_space.high
    else:
        env.close()
        raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")
    env.close()
    return obs_dim, act_dim, is_discrete, act_low, act_high


def collect_reference_batch(env_id: str, batch_size: int = 256, seed: int = 0) -> np.ndarray:
    """Collects a fixed minibatch of random observations at the start of training."""
    env = gym.make(env_id)
    obs_list = []
    obs, _ = env.reset(seed=seed)
    obs_list.append(obs)
    while len(obs_list) < batch_size:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        obs_list.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.asarray(obs_list[:batch_size], dtype=np.float32)