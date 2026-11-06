import os
from typing import List, Dict
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from models import OpenAIESPolicy, get_env_specs
from algorithms import TrainingHistory


def _interpolate_histories(histories: List[TrainingHistory], x_attr: str, num_points: int = 100):
    max_x = min(getattr(h, x_attr)[-1] for h in histories)
    grid = np.linspace(0.0, max_x, num_points)
    interp_matrix = np.zeros((len(histories), num_points), dtype=np.float64)

    for i, h in enumerate(histories):
        x_vals = np.asarray(getattr(h, x_attr), dtype=np.float64)
        y_vals = np.asarray(h.eval_rewards, dtype=np.float64)
        interp_matrix[i] = np.interp(grid, x_vals, y_vals)

    mean = np.mean(interp_matrix, axis=0)
    std_err = stats.sem(interp_matrix, axis=0) if len(histories) > 1 else np.zeros_like(mean)
    ci_95 = 1.96 * std_err
    return grid, mean, ci_95


def plot_comparative_curves(
    all_results: Dict[str, List[TrainingHistory]],
    env_id: str,
    output_path: str,
    title_suffix: str = "Algorithm Comparison",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    palette = sns.color_palette("deep", n_colors=len(all_results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    ax_steps = axes[0]
    for (algo_name, histories), color in zip(all_results.items(), palette):
        grid, mean, ci = _interpolate_histories(histories, x_attr="steps")
        ax_steps.plot(grid, mean, label=algo_name, color=color, linewidth=2.2)
        ax_steps.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.22)

    ax_steps.set_title(f"{env_id}: Sample Efficiency ({title_suffix})", fontweight="bold")
    ax_steps.set_xlabel("Environment Steps")
    ax_steps.set_ylabel("Mean Evaluation Return (95% CI)")
    ax_steps.legend(loc="lower right", frameon=True)

    ax_time = axes[1]
    for (algo_name, histories), color in zip(all_results.items(), palette):
        grid, mean, ci = _interpolate_histories(histories, x_attr="wall_times")
        ax_time.plot(grid, mean, label=algo_name, color=color, linewidth=2.2)
        ax_time.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.22)

    ax_time.set_title(f"{env_id}: Wall-Clock Efficiency ({title_suffix})", fontweight="bold")
    ax_time.set_xlabel("Wall-Clock Time (Seconds)")
    ax_time.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved learning curves to {output_path}")


def plot_scaling_curve(worker_counts: List[int], wall_times: List[float], env_id: str, output_path: str) -> None:
    """Reproduces Figure 1 from Salimans et al. (2017): Wall-clock time vs. CPU workers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(worker_counts, wall_times, marker="s", linewidth=2.2, color="#1f77b4", label="Measured ES Runtime")
    ideal = [wall_times[0] * (worker_counts[0] / w) for w in worker_counts]
    ax.plot(worker_counts, ideal, linestyle="--", linewidth=1.8, color="#d62728", label="Ideal Linear Speedup")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(worker_counts)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title(f"{env_id}: Parallel CPU Scaling (Salimans et al. Fig. 1)", fontweight="bold")
    ax.set_xlabel("Number of Parallel CPU Workers")
    ax.set_ylabel("Wall-Clock Training Time (Seconds, Log Scale)")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved CPU scaling plot to {output_path}")


def run_statistical_analysis(all_results: Dict[str, List[TrainingHistory]], reference_key: str = "OpenAI-ES") -> None:
    print("\n" + "=" * 82)
    print("FINAL PERFORMANCE SUMMARY & MANN-WHITNEY U STATISTICAL TESTS")
    print("=" * 82)

    final_scores = {}
    peak_scores = {}
    for algo_name, histories in all_results.items():
        finals = np.array([h.eval_rewards[-1] for h in histories], dtype=np.float64)
        peaks = np.array([h.best_eval_reward for h in histories], dtype=np.float64)
        times = np.array([h.wall_times[-1] for h in histories], dtype=np.float64)
        final_scores[algo_name] = finals
        peak_scores[algo_name] = peaks
        print(
            f"{algo_name:<22} | Final: {finals.mean():7.2f} ± {finals.std():6.2f} "
            f"| Best: {peaks.mean():7.2f} ± {peaks.std():6.2f} | Time: {times.mean():5.2f}s"
        )

    if reference_key in final_scores:
        ref_vals = final_scores[reference_key]
        print("-" * 82)
        for algo_name, scores in final_scores.items():
            if algo_name == reference_key:
                continue
            u_stat, p_val = stats.mannwhitneyu(ref_vals, scores, alternative="two-sided")
            sig = "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant (p >= 0.05)"
            print(f"Mann-Whitney U ({reference_key} vs. {algo_name:<18}): U = {u_stat:5.1f}, p = {p_val:.4f} -> {sig}")
    print("=" * 82 + "\n")


def record_trained_agent_video(
    env_id: str,
    best_history: TrainingHistory,
    video_folder: str = "outputs/videos",
    seed: int = 42,
) -> None:
    if best_history.best_theta is None:
        return
    os.makedirs(video_folder, exist_ok=True)
    obs_dim, act_dim, is_discrete, act_low, act_high = get_env_specs(env_id)
    policy = OpenAIESPolicy(
        obs_dim, act_dim, is_discrete, act_low, act_high, use_vbn=(best_history.ref_obs is not None), requires_grad=False
    )
    if best_history.ref_obs is not None:
        policy.initialize_vbn(best_history.ref_obs)
    policy.set_flat_params(best_history.best_theta)

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"es_best_{env_id}",
        episode_trigger=lambda ep_id: True,
        disable_logger=True,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

    env.close()
    print(f"[Video] Recorded trained OpenAI-ES agent (Return: {total_reward:.2f}) in {video_folder}/")