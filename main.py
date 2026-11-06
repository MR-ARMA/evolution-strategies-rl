import argparse
import multiprocessing as mp
import numpy as np

from algorithms import train_openai_es, train_random_hill_climbing, train_reinforce
from visualization import (
    plot_comparative_curves,
    plot_scaling_curve,
    run_statistical_analysis,
    record_trained_agent_video,
)

ENV_PRESETS = {
    "CartPole-v1": {
        "max_steps": 100_000,
        "eval_interval": 4_000,
        "es_pop": 24,
        "es_sigma": 0.08,
        "es_lr": 0.05,
        "rhc_sigma": 0.08,
        "pg_lr": 3e-3,
    },
    "Pendulum-v1": {
        "max_steps": 400_000,
        "eval_interval": 10_000,
        "es_pop": 24,
        "es_sigma": 0.10,
        "es_lr": 0.05,
        "rhc_sigma": 0.10,
        "pg_lr": 2e-3,
    },
    "Acrobot-v1": {
        "max_steps": 160_000,
        "eval_interval": 6_000,
        "es_pop": 28,
        "es_sigma": 0.06,
        "es_lr": 0.04,
        "rhc_sigma": 0.06,
        "pg_lr": 2e-3,
    },
}


def run_benchmark(env_id: str, seeds: int, workers: int, max_steps: int, preset: dict, no_video: bool):
    print(f"\n[Mode: Benchmark] {env_id} | {seeds} seeds | {max_steps} steps/seed | {workers} ES workers")
    results = {"OpenAI-ES": [], "RHC / (1+1)-ES": [], "REINFORCE": []}
    best_es_hist = None

    for s in range(seeds):
        seed_val = 100 + s
        print(f"\n--- Seed {s + 1}/{seeds} (seed={seed_val}) ---")

        es_h = train_openai_es(
            env_id=env_id,
            max_steps=max_steps,
            seed=seed_val,
            pop_size=preset["es_pop"],
            sigma=preset["es_sigma"],
            lr=preset["es_lr"],
            num_workers=workers,
        )
        results["OpenAI-ES"].append(es_h)
        print(f"  [OpenAI-ES]      Final: {es_h.eval_rewards[-1]:7.2f} | Best: {es_h.best_eval_reward:7.2f} | Time: {es_h.wall_times[-1]:5.2f}s")

        if best_es_hist is None or es_h.best_eval_reward > best_es_hist.best_eval_reward:
            best_es_hist = es_h

        rhc_h = train_random_hill_climbing(
            env_id=env_id,
            max_steps=max_steps,
            seed=seed_val,
            sigma=preset["rhc_sigma"],
            eval_interval_steps=preset["eval_interval"],
        )
        results["RHC / (1+1)-ES"].append(rhc_h)
        print(f"  [RHC / (1+1)-ES] Final: {rhc_h.eval_rewards[-1]:7.2f} | Best: {rhc_h.best_eval_reward:7.2f} | Time: {rhc_h.wall_times[-1]:5.2f}s")

        pg_h = train_reinforce(
            env_id=env_id,
            max_steps=max_steps,
            seed=seed_val,
            lr=preset["pg_lr"],
            eval_interval_steps=preset["eval_interval"],
        )
        results["REINFORCE"].append(pg_h)
        print(f"  [REINFORCE]      Final: {pg_h.eval_rewards[-1]:7.2f} | Best: {pg_h.best_eval_reward:7.2f} | Time: {pg_h.wall_times[-1]:5.2f}s")

    run_statistical_analysis(results, reference_key="OpenAI-ES")
    plot_comparative_curves(results, env_id, f"outputs/{env_id}_benchmark.png", "ES vs. Baselines")

    if not no_video and best_es_hist is not None:
        record_trained_agent_video(env_id, best_es_hist, video_folder=f"outputs/videos_{env_id}")


def run_ablation(env_id: str, seeds: int, workers: int, max_steps: int, preset: dict):
    """Evaluates the key algorithmic mechanisms from Salimans et al. (2017) Sec. 2.1 & 2.2."""
    print(f"\n[Mode: Ablation Study] {env_id} | Testing VBN, Mirrored Sampling, and Rank Shaping")
    variants = {
        "OpenAI-ES (Full)": dict(mirrored=True, rank_fitness=True, use_vbn=True),
        "No Virtual BatchNorm": dict(mirrored=True, rank_fitness=True, use_vbn=False),
        "No Rank Shaping": dict(mirrored=True, rank_fitness=False, use_vbn=True),
        "No Mirrored Sampling": dict(mirrored=False, rank_fitness=True, use_vbn=True),
    }
    results = {k: [] for k in variants}

    for s in range(seeds):
        seed_val = 200 + s
        print(f"\n--- Ablation Seed {s + 1}/{seeds} (seed={seed_val}) ---")
        for name, cfg in variants.items():
            h = train_openai_es(
                env_id=env_id,
                max_steps=max_steps,
                seed=seed_val,
                pop_size=preset["es_pop"],
                sigma=preset["es_sigma"],
                lr=preset["es_lr"],
                num_workers=workers,
                label=name,
                **cfg,
            )
            results[name].append(h)
            print(f"  [{name:<20}] Final: {h.eval_rewards[-1]:7.2f} | Best: {h.best_eval_reward:7.2f} | Time: {h.wall_times[-1]:5.2f}s")

    run_statistical_analysis(results, reference_key="OpenAI-ES (Full)")
    plot_comparative_curves(results, env_id, f"outputs/{env_id}_ablation.png", "Component Ablation")


def run_frameskip_study(env_id: str, seeds: int, workers: int, max_steps: int, preset: dict):
    """Reproduces Figure 2 of Salimans et al. (2017): Invariance to Frame-Skip in {1, 2, 3, 4}."""
    print(f"\n[Mode: Frame-Skip Invariance (Fig. 2)] {env_id} | FrameSkip in {{1, 2, 3, 4}}")
    skips = [1, 2, 3, 4]
    results = {f"ES FrameSkip={k}": [] for k in skips}

    for s in range(seeds):
        seed_val = 300 + s
        for k in skips:
            h = train_openai_es(
                env_id=env_id,
                max_steps=max_steps,
                seed=seed_val,
                pop_size=preset["es_pop"],
                sigma=preset["es_sigma"],
                lr=preset["es_lr"],
                num_workers=workers,
                action_repeat=k,
                label=f"ES FrameSkip={k}",
            )
            results[f"ES FrameSkip={k}"].append(h)
            print(f"  [Seed {s+1} | FrameSkip={k}] Final: {h.eval_rewards[-1]:7.2f} | Time: {h.wall_times[-1]:5.2f}s")

    plot_comparative_curves(results, env_id, f"outputs/{env_id}_frameskip.png", "Temporal Resolution Invariance")


def run_scaling_study(env_id: str, max_steps: int, preset: dict):
    """Reproduces Figure 1 of Salimans et al. (2017): Parallel speedup across CPU cores."""
    max_cpus = mp.cpu_count()
    worker_list = sorted(list({1, 2, min(4, max_cpus), max_cpus}))
    print(f"\n[Mode: CPU Scaling (Fig. 1)] {env_id} | Testing worker counts: {worker_list}")

    median_times = []
    for w in worker_list:
        times = []
        for s in range(3):
            h = train_openai_es(
                env_id=env_id,
                max_steps=max_steps,
                seed=400 + s,
                pop_size=max(preset["es_pop"], w * 4),
                sigma=preset["es_sigma"],
                lr=preset["es_lr"],
                num_workers=w,
            )
            times.append(h.wall_times[-1])
        med_t = float(np.median(times))
        median_times.append(med_t)
        print(f"  Workers: {w:2d} | Median Wall-Clock Time: {med_t:6.2f}s")

    plot_scaling_curve(worker_list, median_times, env_id, f"outputs/{env_id}_cpu_scaling.png")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Evolution Strategies Suite (Salimans et al., 2017)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment (CartPole-v1, Pendulum-v1, Acrobot-v1)")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "ablation", "frameskip", "scaling", "all"])
    parser.add_argument("--seeds", type=int, default=5, help="Number of independent random seeds")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Parallel CPU workers")
    parser.add_argument("--steps", type=int, default=None, help="Override total environment steps per seed")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 video recording")
    args = parser.parse_args()

    preset = ENV_PRESETS.get(args.env, ENV_PRESETS["CartPole-v1"])
    max_steps = args.steps if args.steps is not None else preset["max_steps"]

    if args.mode in ("benchmark", "all"):
        run_benchmark(args.env, args.seeds, args.workers, max_steps, preset, args.no_video)
    if args.mode in ("ablation", "all"):
        run_ablation(args.env, args.seeds, args.workers, max_steps, preset)
    if args.mode in ("frameskip", "all"):
        run_frameskip_study(args.env, args.seeds, args.workers, max_steps, preset)
    if args.mode in ("scaling", "all"):
        run_scaling_study(args.env, max_steps, preset)


if __name__ == "__main__":
    main()