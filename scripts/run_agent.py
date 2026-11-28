"""Run the literature-informed agent and count the strong combinations it finds.

Offline by default: the model is the scripted top-k fake, which must reproduce Bayesian
optimization exactly (a wiring check, not a result). Real model calls need --client real,
the --allow-llm flag and an API key. Literature comes from the cache only
(scripts/prefetch_literature.py fills it).

Writes runs/agent/<name>_seed<N>.csv (run log), _hypotheses.json, _calls.jsonl and
_heldout_predictions.json (null if the model's answer failed twice). The run log is written
last, so a seed counts as finished only once all four exist; finished seeds are skipped (never
paid for twice) unless --overwrite is given. --key 2 uses LLM_API_KEY_2 and its own spend log.
Usage: python -m scripts.run_agent --seeds 0 1 2 3 4 [--ablation no_literature] [--key 2]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ccagent.agent import ABLATIONS, AgentChooser, AgentConfig
from ccagent.data.synonyms import synonyms_by_drug
from ccagent.evaluation import add_strong_flags, visible_truth
from ccagent.lab import SimLab, read_heldout
from ccagent.literature import Exclusions, Literature, PubMed, class_terms, drug_names
from ccagent.llm import FakeLLMClient, add_llm_args, get_client, top_k_responder
from ccagent.llm.guard import spend_log
from ccagent.loop import run_selection
from ccagent.masking import Masker
from ccagent.strong import PRIMARY
from scripts.run_baselines import cell_line_tissue

OUT = Path("runs/agent")
OUTPUTS = ("_heldout_predictions.json", "_hypotheses.json", "_calls.jsonl", ".csv")  # csv last


def finished(stem: Path) -> bool:
    return all(Path(f"{stem}{suffix}").exists() for suffix in OUTPUTS)


def build_agent(client, ablation: str, seed: int, target: str) -> AgentChooser:
    annotations = pd.read_csv("data/reference/drug_annotations.csv")
    summaries = pd.read_csv("data/processed/cell_line_summaries.csv")
    masker = Masker.from_tables(
        pd.read_csv("data/reference/drugs.csv"), annotations,
        synonyms_by_drug(pd.read_csv("data/reference/drug_synonyms.csv"))) \
        if ablation == "masked_names" else Masker()
    literature = None if ablation == "no_literature" else \
        Literature(PubMed(), Exclusions.from_file(), drug_names(), class_terms(),
                   hide_lines=read_heldout())
    return AgentChooser(
        client=client, annotations=annotations,
        line_summaries=dict(zip(summaries["cell_line"], summaries["summary"], strict=True)),
        heldout=set(read_heldout()), literature=literature, tissue=cell_line_tissue(),
        masker=masker, config=AgentConfig(target=target, ablation=ablation, seed=seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--target", default="joint")
    parser.add_argument("--client", choices=["fake", "real"], default="fake")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--est-cost", type=float, default=3.0,
                        help="expected cost of one real run; a seed is skipped if it would "
                             "push logged spend past the budget")
    parser.add_argument("--overwrite", action="store_true",
                        help="rerun seeds whose outputs already exist (skipped by default)")
    add_llm_args(parser)
    args = parser.parse_args()
    ledger = None
    if args.client == "real":
        from dotenv import load_dotenv

        from ccagent.llm import settings
        from ccagent.llm.real import SpendLedger

        load_dotenv()
        ledger = SpendLedger(spend_log(args.key))
        print(f"model {settings.AGENT_MODEL}, effort {settings.AGENT_EFFORT}; key {args.key} "
              f"spent so far ${ledger.total():.2f} of ${ledger.budget_usd:.2f}", flush=True)

    truth, out_dir = visible_truth(), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"agent_{args.client}_{args.ablation}_{args.target}"
    found = []
    for seed in args.seeds:
        started = time.time()
        stem = out_dir / f"{name}_seed{seed}"
        if finished(stem) and not args.overwrite:
            print(f"seed {seed} skipped: already finished ({stem}.csv); --overwrite reruns it")
            continue
        if ledger is not None and ledger.total() + args.est_cost > ledger.budget_usd:
            print(f"seed {seed} skipped: ${ledger.total():.2f} spent, another run "
                  f"(~${args.est_cost:.2f}) would pass the budget")
            continue
        client = FakeLLMClient(responder=top_k_responder) if args.client == "fake" \
            else get_client("real", allow_llm=args.allow_llm, run_id=f"{name}_seed{seed}",
                            key=args.key)
        agent = build_agent(client, args.ablation, seed, args.target)
        log = run_selection(SimLab.from_files(), agent, seed=seed)
        predictions = agent.predict_heldout()  # after the last pick; genetics only
        log = add_strong_flags(log, truth)     # scoring reads the data only now
        Path(f"{stem}_heldout_predictions.json").write_text(json.dumps(
            {"predictions": predictions}, indent=1))
        Path(f"{stem}_hypotheses.json").write_text(json.dumps(agent.book.to_dict(), indent=1))
        with open(f"{stem}_calls.jsonl", "w") as f:
            for call in agent.calls:
                f.write(json.dumps(call) + "\n")
        log.to_csv(f"{stem}.csv", index=False)  # last: its presence marks the seed finished
        found.append(int(log[f"strong_{PRIMARY}"].sum()))
        cost = sum(c["cost_usd"] for c in agent.calls)
        print(f"seed {seed}: {found[-1]} strong, {len(agent.calls)} model calls, "
              f"${cost:.2f}, {time.time() - started:.0f}s", flush=True)
    if not found:
        return
    spread = float(np.std(found, ddof=1)) if len(found) > 1 else 0.0
    print(f"{name}: {np.mean(found):.1f} +/- {spread:.1f}  per seed {found}")


if __name__ == "__main__":
    main()
