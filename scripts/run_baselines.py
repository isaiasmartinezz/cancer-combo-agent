"""Run the non-LLM methods and report strong combinations found.

Writes runs/baselines/<method>_seed<N>.csv. Usage:
  python -m scripts.run_baselines --methods random --seeds 0 1 2 3 4
"""
import argparse
from pathlib import Path

from ccagent.baselines import random_chooser
from ccagent.evaluation import add_strong_flags, visible_truth
from ccagent.lab import SimLab
from ccagent.loop import run_selection
from ccagent.strong import PRIMARY

METHODS = {"random": lambda seed: random_chooser}
OUT = Path("runs/baselines")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    truth = visible_truth()
    OUT.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        counts = []
        for seed in args.seeds:
            log = run_selection(SimLab.from_files(), METHODS[method](seed), seed=seed)
            log = add_strong_flags(log, truth)
            log.to_csv(OUT / f"{method}_seed{seed}.csv", index=False)
            counts.append(int(log[f"strong_{PRIMARY}"].sum()))
        mean = sum(counts) / len(counts)
        print(f"{method:12} strong found per seed {counts}  mean {mean:.1f}")


if __name__ == "__main__":
    main()
