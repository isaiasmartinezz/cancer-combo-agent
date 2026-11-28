"""Write the exact prompts the agent's model sees, for review and the report's appendix.

Offline: a short scripted run (4 rounds, seed 0) for each arm, with the top-k fake model, so the
hypotheses shown are placeholders, not model output. The held-out prompt needs a mechanistic
hypothesis, which the scripted model never makes, so one marked [example] is supplied.
Usage: python -m scripts.render_prompts [--out docs/prompt_examples]
"""
import argparse
from pathlib import Path

from ccagent import prompts
from ccagent.hypotheses import Hypothesis
from ccagent.lab import SimLab
from ccagent.llm import FakeLLMClient, top_k_responder
from ccagent.loop import run_selection
from ccagent.masking import Masker
from ccagent.targets import DESCRIPTIONS
from scripts.run_agent import build_agent

ARMS = ("none", "no_literature", "masked_names")
ROUNDS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/prompt_examples")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        agent = build_agent(FakeLLMClient(responder=top_k_responder), arm, 0, "joint")
        run_selection(SimLab.from_files(budget=10 * ROUNDS + 10), agent, seed=0)
        (out / f"{arm}_system.txt").write_text(agent.system)
        for call in agent.calls:
            if call["round"] == 2 and call["attempt"] == 1:
                (out / f"{arm}_round2_{call['step']}.txt").write_text(call["prompt"])
    example = Hypothesis(
        id="H7", statement="[example] A WEE1 inhibitor with gemcitabine hits p53-mutant lines.",
        drugs=("AZD1775", "GEMCITABINE"), level="mechanistic",
        predicted_responders=("A2058", "ES2", "SW-620"),
        responder_feature="TP53 mutant",
        status="supported", n_tests=4, score=0.72)
    agent = build_agent(FakeLLMClient(responder=top_k_responder), "none", 0, "joint")
    lines = {line: agent.line_summaries.get(line, "genetics unknown")
             for line in sorted(agent.heldout)}
    (out / "heldout_prediction.txt").write_text(
        prompts.heldout_prompt([example], lines, DESCRIPTIONS["joint"], Masker()))
    print(f"wrote {len(list(out.glob('*.txt')))} files to {out}")


if __name__ == "__main__":
    main()
