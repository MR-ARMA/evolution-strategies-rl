import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import torch
import gymnasium as gym

from models import OpenAIESPolicy, get_env_specs, collect_reference_batch

torch.set_num_threads(1)


@dataclass
class TrainingHistory:
    algorithm: str
    seed: int
    steps: List[int] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    best_theta: Optional[np.ndarray] = None
    best_eval_reward: float = -np.inf
    ref_obs: Optional[np.ndarray] = None


def centered_rank_transformation(returns: np.ndarray) -> np.ndarray:
    """
    Fitness shaping via rank transformation linearly scaled to [-0.5, 0.5]
    (Salimans et al., 2017 Sec. 2.1; Wierstra et al., 2014).
    """
    n = len(returns)
    ranks = np.empty(n, dtype=np.float32)
    ranks[np.argsort(returns)] = np.arange(n, dtype=np.float32)
    if n > 1:
        ranks /= (n - 1)
    ranks -= 0.5
    return ranks


class AdamOptimizer:
    """First-order Adam optimizer for updating parameter vector theta in parameter space."""

    def __init__(self, dim: int, lr: float = 0.03, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def update(self, theta: np.ndarray, grad_ascent: np.ndarray) -> np.ndarray:
        self.t += 1
        g = -grad_ascent
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Persistent worker state for zero-overhead parallel rollouts
_W_ENV = None
_W_POLICY = None


def _init_worker(env_id: str, use_vbn: bool, ref_obs: np.ndarray):
    global _W_ENV, _W_POLICY
    _W_ENV = gym.make(env_id)
    obs_dim, act_dim, is_discrete, act_low, act_high = get_env_specs(env_id)
    _W_POLICY = OpenAIESPolicy(
        obs_dim, act_dim, is_discrete, act_low, act_high, use_vbn=use_vbn, requires_grad=False
    )
    if ref_obs is not None:
        _W_POLICY.initialize_vbn(ref_obs)


def _run_rollout(
    env: gym.Env,
    policy: OpenAIESPolicy,
    theta: np.ndarray,
    env_seed: int,
    max_ep_steps: int,
    action_repeat: int = 1,
) -> Tuple[float, int]:
    """Executes one episode with optional action repeat (frame-skip) and dynamic step cap."""
    policy.set_flat_params(theta)
    obs, _ = env.reset(seed=env_seed)
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_ep_steps:
        action = policy.act(obs)
        for _ in range(action_repeat):
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if done or steps >= max_ep_steps:
                break

    return total_reward, steps


def _es_worker_task(args: Tuple[np.ndarray, int, float, int, int, int, bool]) -> Tuple[int, float, float, int]:
    """
    Algorithm 2 Parallel Worker Task:
    Reconstructs perturbation eps from noise_seed locally and returns only scalars:
    (noise_seed, return_pos, return_neg, steps_taken).
    """
    global _W_ENV, _W_POLICY
    theta, noise_seed, sigma, env_seed, max_ep_steps, action_repeat, mirrored = args

    rng = np.random.default_rng(noise_seed)
    eps = rng.standard_normal(theta.size, dtype=np.float32)

    ret_pos, steps_pos = _run_rollout(
        _W_ENV, _W_POLICY, theta + sigma * eps, env_seed, max_ep_steps, action_repeat
    )
    if mirrored:
        ret_neg, steps_neg = _run_rollout(
            _W_ENV, _W_POLICY, theta - sigma * eps, env_seed, max_ep_steps, action_repeat
        )
        return noise_seed, ret_pos, ret_neg, steps_pos + steps_neg
    else:
        return noise_seed, ret_pos, 0.0, steps_pos


def evaluate_policy(
    env: gym.Env,
    policy: OpenAIESPolicy,
    theta: np.ndarray,
    n_episodes: int = 5,
    base_seed: int = 8888,
    action_repeat: int = 1,
) -> float:
    """Evaluates the deterministic mean policy over n_episodes without parameter noise."""
    scores = []
    for i in range(n_episodes):
        ret, _ = _run_rollout(env, policy, theta, base_seed + i, max_ep_steps=10_000, action_repeat=action_repeat)
        scores.append(ret)
    return float(np.mean(scores))


def train_openai_es(
    env_id: str,
    max_steps: int,
    seed: int = 0,
    pop_size: int = 40,
    sigma: float = 0.05,
    lr: float = 0.03,
    weight_decay: float = 0.005,
    num_workers: int = 4,
    mirrored: bool = True,
    rank_fitness: bool = True,
    use_vbn: bool = True,
    dynamic_step_cap: bool = True,
    action_repeat: int = 1,
    label: str = "OpenAI-ES",
) -> TrainingHistory:
    """
    Full implementation of Algorithm 2 from Salimans et al. (2017):
    Parallelized Evolution Strategies with Common Random Numbers, Antithetic Sampling,
    Rank Fitness Shaping, Virtual Batch Normalization, Weight Decay, and Dynamic Step Capping.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    obs_dim, act_dim, is_discrete, act_low, act_high = get_env_specs(env_id)
    ref_obs = collect_reference_batch(env_id, batch_size=128, seed=seed)
    master_policy = OpenAIESPolicy(
        obs_dim, act_dim, is_discrete, act_low, act_high, use_vbn=use_vbn, requires_grad=False
    )
    if use_vbn:
        master_policy.initialize_vbn(ref_obs)

    theta = master_policy.get_flat_params()
    dim = theta.size
    optimizer = AdamOptimizer(dim, lr=lr)
    eval_env = gym.make(env_id)

    history = TrainingHistory(algorithm=label, seed=seed, ref_obs=ref_obs)
    start_time = time.perf_counter()
    total_steps = 0
    max_ep_steps = 10_000

    # Initial evaluation at step 0
    init_score = evaluate_policy(eval_env, master_policy, theta, n_episodes=5, base_seed=seed * 1000, action_repeat=action_repeat)
    history.steps.append(0)
    history.wall_times.append(0.0)
    history.eval_rewards.append(init_score)
    history.best_eval_reward = init_score
    history.best_theta = theta.copy()

    num_tasks = (pop_size // 2) if mirrored else pop_size
    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(env_id, use_vbn, ref_obs),
    ) as pool:
        while total_steps < max_steps:
            tasks = []
            for _ in range(num_tasks):
                noise_seed = int(rng.integers(0, 2**31 - 1))
                env_seed = int(rng.integers(0, 2**31 - 1))
                tasks.append((theta, noise_seed, sigma, env_seed, max_ep_steps, action_repeat, mirrored))

            results = pool.map(_es_worker_task, tasks)

            noise_seeds = []
            raw_returns = []
            gen_steps = 0

            for noise_seed, ret_pos, ret_neg, steps_taken in results:
                noise_seeds.append(noise_seed)
                if mirrored:
                    raw_returns.extend([ret_pos, ret_neg])
                else:
                    raw_returns.append(ret_pos)
                gen_steps += steps_taken

            total_steps += gen_steps
            returns_arr = np.asarray(raw_returns, dtype=np.float32)

            # Dynamic episode length cap: m = 2 * mean_steps (Salimans et al., Sec. 2.1)
            if dynamic_step_cap:
                mean_ep_steps = gen_steps / float(len(raw_returns))
                max_ep_steps = max(100, int(2.0 * mean_ep_steps))

            # Fitness shaping: rank transformation vs. standardized raw returns
            if rank_fitness:
                utilities = centered_rank_transformation(returns_arr)
            else:
                std = returns_arr.std()
                utilities = (returns_arr - returns_arr.mean()) / (std + 1e-8)

            # Reconstruct perturbations from known scalar seeds (Algorithm 2, lines 9-11)
            grad = np.zeros(dim, dtype=np.float32)
            for i, noise_seed in enumerate(noise_seeds):
                eps = np.random.default_rng(noise_seed).standard_normal(dim, dtype=np.float32)
                if mirrored:
                    grad += (utilities[2 * i] - utilities[2 * i + 1]) * eps
                else:
                    grad += utilities[i] * eps

            grad /= (len(raw_returns) * sigma)
            # L2 weight decay regularization (Salimans et al., Sec. 2.1)
            grad -= weight_decay * theta

            theta = optimizer.update(theta, grad)

            # Evaluate updated mean policy
            eval_score = evaluate_policy(
                eval_env, master_policy, theta, n_episodes=5, base_seed=seed * 1000, action_repeat=action_repeat
            )
            elapsed = time.perf_counter() - start_time
            if eval_score >= history.best_eval_reward:
                history.best_eval_reward = eval_score
                history.best_theta = theta.copy()

            # Log current evaluation score
            history.steps.append(total_steps)
            history.wall_times.append(elapsed)
            history.eval_rewards.append(eval_score)

    eval_env.close()
    return history


def train_random_hill_climbing(
    env_id: str,
    max_steps: int,
    seed: int = 0,
    sigma: float = 0.08,
    episodes_per_eval: int = 3,
    eval_interval_steps: int = 4000,
    use_vbn: bool = True,
) -> TrainingHistory:
    """
    Baseline 1: Random Hill Climbing / (1+1)-ES.
    Evaluates a single mutated parameter vector per iteration and replaces the parent
    only if the child's mean return improves upon the parent.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    obs_dim, act_dim, is_discrete, act_low, act_high = get_env_specs(env_id)
    ref_obs = collect_reference_batch(env_id, batch_size=128, seed=seed)
    policy = OpenAIESPolicy(
        obs_dim, act_dim, is_discrete, act_low, act_high, use_vbn=use_vbn, requires_grad=False
    )
    if use_vbn:
        policy.initialize_vbn(ref_obs)

    theta = policy.get_flat_params()
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    history = TrainingHistory(algorithm="RHC / (1+1)-ES", seed=seed, ref_obs=ref_obs)
    start_time = time.perf_counter()
    total_steps = 0
    next_eval_step = 0

    best_train_score = evaluate_policy(env, policy, theta, n_episodes=episodes_per_eval, base_seed=seed)

    while total_steps <= max_steps:
        if total_steps >= next_eval_step:
            eval_score = evaluate_policy(eval_env, policy, theta, n_episodes=5, base_seed=seed * 1000)
            elapsed = time.perf_counter() - start_time
            history.steps.append(total_steps)
            history.wall_times.append(elapsed)
            history.eval_rewards.append(eval_score)
            if eval_score >= history.best_eval_reward:
                history.best_eval_reward = eval_score
                history.best_theta = theta.copy()
            next_eval_step += eval_interval_steps

        if total_steps >= max_steps:
            break

        cand_theta = theta + sigma * rng.standard_normal(theta.size, dtype=np.float32)
        cand_scores = []
        for _ in range(episodes_per_eval):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            ret, steps = _run_rollout(env, policy, cand_theta, ep_seed, max_ep_steps=10_000)
            cand_scores.append(ret)
            total_steps += steps

        cand_mean = float(np.mean(cand_scores))
        if cand_mean >= best_train_score:
            theta = cand_theta
            best_train_score = cand_mean

    env.close()
    eval_env.close()
    return history


def train_reinforce(
    env_id: str,
    max_steps: int,
    seed: int = 0,
    lr: float = 3e-3,
    gamma: float = 0.99,
    batch_episodes: int = 5,
    eval_interval_steps: int = 4000,
    action_repeat: int = 1,
    use_vbn: bool = True,
) -> TrainingHistory:
    """
    Baseline 2: Monte Carlo Policy Gradient (REINFORCE) with return standardization baseline.
    Adds exploration noise in action space and computes gradients via backpropagation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    obs_dim, act_dim, is_discrete, act_low, act_high = get_env_specs(env_id)
    ref_obs = collect_reference_batch(env_id, batch_size=128, seed=seed)
    policy = OpenAIESPolicy(
        obs_dim, act_dim, is_discrete, act_low, act_high, use_vbn=use_vbn, requires_grad=True
    )
    if use_vbn:
        policy.initialize_vbn(ref_obs)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    history = TrainingHistory(algorithm="REINFORCE", seed=seed, ref_obs=ref_obs)
    start_time = time.perf_counter()
    total_steps = 0
    next_eval_step = 0

    while total_steps <= max_steps:
        if total_steps >= next_eval_step:
            theta_curr = policy.get_flat_params()
            eval_score = evaluate_policy(
                eval_env, policy, theta_curr, n_episodes=5, base_seed=seed * 1000, action_repeat=action_repeat
            )
            elapsed = time.perf_counter() - start_time
            history.steps.append(total_steps)
            history.wall_times.append(elapsed)
            history.eval_rewards.append(eval_score)
            if eval_score >= history.best_eval_reward:
                history.best_eval_reward = eval_score
                history.best_theta = theta_curr.copy()
            next_eval_step += eval_interval_steps

        if total_steps >= max_steps:
            break

        batch_log_probs = []
        batch_returns = []

        for _ in range(batch_episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            ep_log_probs = []
            ep_rewards = []

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = policy.forward_distribution(obs_t)
                action_t = dist.sample()

                if is_discrete:
                    log_prob = dist.log_prob(action_t).squeeze(0)
                    env_action = int(action_t.item())
                else:
                    log_prob = dist.log_prob(action_t).sum(dim=-1).squeeze(0)
                    env_action = action_t.squeeze(0).detach().cpu().numpy()
                    if act_low is not None and act_high is not None:
                        env_action = np.clip(env_action, act_low, act_high)

                step_reward = 0.0
                for _ in range(action_repeat):
                    obs, r, terminated, truncated, _ = env.step(env_action)
                    step_reward += float(r)
                    total_steps += 1
                    done = terminated or truncated
                    if done:
                        break

                ep_log_probs.append(log_prob)
                ep_rewards.append(step_reward)

            G = 0.0
            disc_returns = []
            for r in reversed(ep_rewards):
                G = r + gamma * G
                disc_returns.insert(0, G)

            batch_log_probs.extend(ep_log_probs)
            batch_returns.extend(disc_returns)

        returns_t = torch.as_tensor(batch_returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(batch_log_probs)
        loss = -(log_probs_t * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    env.close()
    eval_env.close()
    return history