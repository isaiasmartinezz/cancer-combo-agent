"""Run SHELDON on the O'Neil lab (ccagent/sheldon_arm.py) and count the strong combinations found.

Offline by default: SHELDON's scripted fake model, which plays BO through the tools (get the
shortlist, run its first 10). That is a wiring check, not a result. Real calls need
--client real, --allow-llm and LLM_API_KEY, as for the agent.

Same model and effort as the agent (settings.py). Tools: only the benchmark's own (probes = []
removes SHELDON's web search, web fetch and shell). Money: each seed's SHELDON cap (max_usd) is
the smaller of --max-usd and what the key's budget has left, and the spend is written to that
key's spend log (settings.KEYS; --key 2 uses LLM_API_KEY_2) at our list prices, so its $30 cap
counts it. A seed whose run log already exists is skipped unless --overwrite is given.

Writes runs/sheldon/sheldon_joint_seed<N>.csv (run log); SHELDON's own run folders, traces and
model-call cache go under runs/sheldon/.
Usage: python -m scripts.run_sheldon --seeds 0 [--client real --allow-llm] [--max-usd 10]
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from ccagent.llm import settings
from ccagent.llm.guard import add_llm_args, spend_log
from ccagent.llm.real import SpendLedger
from ccagent.llm.sheldon_client import EFFORT, MODEL, make_llm
from ccagent.sheldon_arm import NAME, scripted_bo_responder

OUT = Path("runs/sheldon")
MARGIN = 0.50       # budget left untouched by a SHELDON run
MIN_CAP = 1.00      # below this a seed is skipped rather than started


def build_config(seed: int, max_usd: float, rounds: int, literature: bool, out: Path,
                 target: str = "joint"):
    from sheldon.core.config import Config

    cfg = Config(
        benchmark=NAME, method="sheldon", split="eval", seed=seed, engine="toolloop",
        provider="anthropic", model=MODEL, effort=EFFORT, thinking=True,
        max_tokens=settings.MAX_TOKENS, max_rounds=rounds, max_calls=None, max_usd=max_usd,
        concurrency=1, cache=True, resume=False, tag=f"seed{seed}",
        params={"probes": [], "mode": "staged", "rounds": rounds,
                "benchmark_opts": {"seeds": [seed], "literature": literature, "target": target,
                                   "log_dir": str(out)}})
    if cfg.params["probes"] != []:
        raise ValueError("SHELDON must run with the benchmark's tools only (probes = [])")
    return cfg


def usd(usage: dict) -> float:
    """What the run cost at our list prices (SHELDON's own estimate rounds cache reads up)."""
    inp, out, read, write = settings.PRICES[MODEL]
    return (usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out
            + usage.get("cache_read_tokens", 0) * read
            + usage.get("cache_write_tokens", 0) * write)


def main() -> None:
    from sheldon.runner import run_experiment

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--client", choices=["fake", "real"], default="fake")
    parser.add_argument("--max-usd", type=float, default=10.0, help="cap for one seed")
    parser.add_argument("--rounds", type=int, default=40,
                        help="round cap; the run normally ends when the experiments run out")
    parser.add_argument("--no-literature", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dry-run", action="store_true", help="print the config and stop")
    parser.add_argument("--overwrite", action="store_true",
                        help="rerun seeds whose run log already exists (skipped by default)")
    add_llm_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    os.environ["SHELDON_RUNS"] = str(out / "sheldon_runs")
    ledger = SpendLedger(spend_log(args.key))
    if args.client == "real":
        from dotenv import load_dotenv

        load_dotenv()
        print(f"model {MODEL}, effort {EFFORT}; key {args.key} spent so far "
              f"${ledger.total():.2f} of ${ledger.budget_usd:.2f}", flush=True)

    for seed in args.seeds:
        done = out / f"sheldon_joint_seed{seed}.csv"
        if done.exists() and not args.overwrite and not args.dry_run:
            print(f"seed {seed} skipped: already finished ({done}); --overwrite reruns it")
            continue
        cap = args.max_usd
        if args.client == "real":
            cap = min(cap, ledger.budget_usd - ledger.total() - MARGIN)
            if cap < MIN_CAP:
                print(f"seed {seed} skipped: only ${cap:.2f} of budget left for it")
                continue
        cfg = build_config(seed, round(cap, 2), args.rounds, not args.no_literature, out)
        if args.dry_run:
            print(json.dumps(cfg.to_dict(), indent=1, default=str))
            continue
        llm = make_llm(args.client, allow_llm=args.allow_llm,
                       cache_path=out / "cache" / "llm.db",
                       tool_responder=scripted_bo_responder) if args.client == "fake" else \
            make_llm("real", allow_llm=args.allow_llm, cache_path=out / "cache" / "llm.db",
                     key=args.key)
        started = time.time()
        try:
            summary = asyncio.run(run_experiment(cfg, llm=llm))
        finally:
            if args.client == "real":  # record what was billed, even if the run failed
                billed = llm.billed
                tokens = {k: getattr(billed, k) for k in
                          ("input_tokens", "output_tokens", "cache_read_tokens",
                           "cache_write_tokens")}
                ledger.record({"run_id": f"sheldon_joint_seed{seed}", "step": "sheldon_total",
                               "model": MODEL, "calls": billed.calls, **tokens,
                               "cost_usd": round(usd(tokens), 5),
                               "sheldon_estimate_usd": round(billed.cost_usd, 5)})
        m = summary.get("metrics", {})
        print(f"seed {seed}: {summary.get('score', 0):.0f} strong in "
              f"{m.get('experiments_run', 0):.0f} experiments, "
              f"{summary.get('usage', {}).get('calls', 0)} model calls, "
              f"${usd(summary.get('usage', {})):.2f}, {time.time() - started:.0f}s; "
              f"errors {summary.get('errors', 0)}", flush=True)


if __name__ == "__main__":
    main()
