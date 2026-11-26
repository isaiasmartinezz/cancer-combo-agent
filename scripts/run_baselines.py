"""Run the non-language-model methods and report how many strong combinations each one finds.

Writes runs/baselines/<method>_<target>_seed<N>.csv. Offline; no model calls. Usage:
  python -m scripts.run_baselines --methods random uncertainty bo shortlist_random --seeds 0 1 2 3 4
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ccagent.baselines import METHODS, make_baseline
from ccagent.evaluation import add_strong_flags, visible_truth
from ccagent.lab import SimLab
from ccagent.loop import run_selection
from ccagent.shortlist import KAPPA
from ccagent.strong import PRIMARY
from ccagent.targets import TARGETS

OUT = Path("runs/baselines")


def cell_line_tissue() -> dict:
    lines = pd.read_csv("data/processed/cell_lines.csv")
    return dict(zip(lines["cell_line"], lines["tissue"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--target", choices=TARGETS, default="joint",
                        help="what the methods maximize")
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--no-genetics", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for the output name, e.g. _k2")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    truth = visible_truth()
    tissue = cell_line_tissue()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        found, started = [], time.time()
        for seed in args.seeds:
            chooser = make_baseline(method, target=args.target, tissue=tissue, seed=seed,
                                    kappa=args.kappa, with_genetics=not args.no_genetics) \
                if method != "random" else make_baseline(method)
            log = run_selection(SimLab.from_files(), chooser, seed=seed)
            log = add_strong_flags(log, truth)
            log.to_csv(out_dir / f"{method}{args.tag}_{args.target}_seed{seed}.csv", index=False)
            found.append(int(log[f"strong_{PRIMARY}"].sum()))
        spread = float(np.std(found, ddof=1)) if len(found) > 1 else 0.0
        print(f"{method + args.tag:18} {np.mean(found):5.1f} +/- {spread:4.1f}   "
              f"per seed {found}   ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
