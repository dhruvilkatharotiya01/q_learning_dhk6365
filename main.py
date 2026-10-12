"""Train and evaluate the Q-learning agent.

python main.py                      # train with default settings, save results/
python main.py --episodes 500 --alpha 0.2
python main.py --render             # watch the greedy policy (needs pygame)
python main.py --sweep              # compare learning rates
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gridworld.env import GridWorldEnv
from gridworld.q_learning import Config, greedy_path, shortest_path_length, train


def summarise(env: GridWorldEnv, q: np.ndarray, hist, cfg: Config) -> dict:
    path, ret, reached = greedy_path(env, q)
    last = hist.outcomes[-100:]
    first_hit = next((i for i, o in enumerate(hist.outcomes) if o == "goal"), None)
    return {
        "config": asdict(cfg),
        "greedy_policy_reaches_goal": reached,
        "greedy_path_steps": len(path) - 1,
        "optimal_steps": shortest_path_length(env),
        "greedy_return": ret,
        "goal_rate_last_100": round(last.count("goal") / len(last), 3),
        "black_hole_rate_last_100": round(last.count("black_hole") / len(last), 3),
        "first_goal_episode": first_hit,
        "converged_at_episode": hist.converged_at,
        "mean_return_last_100": round(float(np.mean(hist.returns[-100:])), 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--alpha", type=float, default=d.alpha, help="learning rate")
    p.add_argument("--gamma", type=float, default=d.gamma, help="discount factor")
    p.add_argument("--epsilon-decay", type=float, default=d.epsilon_decay)
    p.add_argument("--epsilon-min", type=float, default=d.epsilon_min)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--render", action="store_true", help="show the greedy policy with pygame after training")
    p.add_argument("--sweep", action="store_true", help="compare learning rates over 5 seeds each")
    args = p.parse_args(argv)

    if args.sweep:
        return sweep(args)

    cfg = Config(args.episodes, args.alpha, args.gamma, 1.0, args.epsilon_min, args.epsilon_decay, args.seed)
    env = GridWorldEnv()
    q, hist = train(env, cfg)
    report = summarise(env, q, hist, cfg)
    print(json.dumps(report, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "q_table.npy", q)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    from gridworld.plots import plot_policy, plot_q_values, plot_training

    plot_training(hist, args.out / "training.png")
    plot_q_values(q, env, args.out / "q_values.png")
    plot_policy(q, env, greedy_path(env, q)[0], args.out / "policy.png")
    print(f"Saved Q-table, summary and plots to {args.out}/")

    if args.render:
        env = GridWorldEnv(render_mode="human")
        greedy_path(env, q)
        env.close()
    return 0


def _median(values: list[int | None]) -> str:
    done = [v for v in values if v is not None]
    return str(int(np.median(done))) if len(done) == len(values) else f"{len(done)}/{len(values)} converged"


def sweep(args) -> int:
    rows = []
    for alpha in (0.01, 0.05, 0.1, 0.3, 0.5):
        stats = []
        for seed in range(5):
            cfg = Config(args.episodes, alpha, args.gamma, 1.0, args.epsilon_min, args.epsilon_decay, seed)
            env = GridWorldEnv()
            q, hist = train(env, cfg)
            stats.append(summarise(env, q, hist, cfg))
        rows.append(
            {
                "alpha": alpha,
                "optimal_policy": f"{sum(s['greedy_path_steps'] == s['optimal_steps'] for s in stats)}/5",
                "converged_at_episode (median)": _median([s["converged_at_episode"] for s in stats]),
                "goal_rate_last_100": round(float(np.mean([s["goal_rate_last_100"] for s in stats])), 3),
                "mean_return_last_100": round(float(np.mean([s["mean_return_last_100"] for s in stats])), 1),
            }
        )
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
