"""What each API key has spent, what each arm costs per seed, and what the rest of the plan
should cost at the measured rates. Offline: reads the spend logs only.

Usage: python -m scripts.spend_report
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from ccagent.llm import settings

# Cost of one seed before any real run (Nov 27 estimates), replaced by the measured mean once an
# arm has finished seeds. Keys are run-id prefixes ("agent_real_<ablation>_joint", "sheldon_joint").
ESTIMATE = {"agent_real_none_joint": 2.33, "agent_real_no_literature_joint": 1.82,
            "agent_real_masked_names_joint": 2.33, "agent_real_no_critique_joint": 1.50,
            "sheldon_joint": 9.00}
# The proposed plan: (key, arm, seeds). Edit as decisions change.
PLAN = [("1", "agent_real_none_joint", range(0, 5)),
        ("1", "agent_real_no_literature_joint", range(0, 5)),
        ("1", "agent_real_masked_names_joint", range(0, 5)),
        ("1", "agent_real_no_critique_joint", range(0, 5)),
        ("2", "agent_real_none_joint", range(5, 10)),
        ("2", "agent_real_no_literature_joint", range(5, 10)),
        ("2", "sheldon_joint", range(0, 2))]
PRACTICE_SEEDS = 100  # seeds from 100 up are for tuning and short checks
SEED = re.compile(r"^(?P<arm>.+)_seed(?P<seed>\d+)$")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_run(entries: list[dict]) -> dict[tuple[str, int], float]:
    out = defaultdict(float)
    for e in entries:
        m = SEED.match(e.get("run_id", ""))
        if m:
            out[(m["arm"], int(m["seed"]))] += e["cost_usd"]
    return dict(out)


def report(logs: dict[str, Path], cap: float = settings.BUDGET_USD, plan=PLAN) -> str:
    runs = {key: by_run(read(path)) for key, path in logs.items()}
    done = {(arm, seed) for r in runs.values() for arm, seed in r}
    per_arm = defaultdict(list)
    for r in runs.values():
        for (arm, seed), cost in r.items():
            if seed < PRACTICE_SEEDS:  # short checks run on practice seeds; not a full seed
                per_arm[arm].append(cost)
    rate = {arm: sum(c) / len(c) for arm, c in per_arm.items()}
    lines = []
    for key, path in logs.items():
        spent = sum(e["cost_usd"] for e in read(path))
        todo = [(arm, s) for k, arm, seeds in plan if k == key for s in seeds
                if (arm, s) not in done]
        ahead = sum(rate.get(arm, ESTIMATE.get(arm, 0.0)) for arm, _ in todo)
        lines.append(f"key {key}: spent ${spent:.2f} of ${cap:.2f}; plan left {len(todo)} seeds "
                     f"~${ahead:.2f} -> ~${spent + ahead:.2f} "
                     f"({'fits' if spent + ahead <= cap else 'OVER the cap'})")
    lines.append("arm: seeds run, cost per seed (estimate)")
    for arm in sorted(set(ESTIMATE) | set(rate)):
        n = len(per_arm.get(arm, []))
        measured = f"${rate[arm]:.2f}" if arm in rate else "-"
        lines.append(f"  {arm}: {n}, {measured} (${ESTIMATE.get(arm, 0):.2f})")
    return "\n".join(lines)


def main() -> None:
    print(report({key: Path(log) for key, (_, log) in settings.KEYS.items()}))


if __name__ == "__main__":
    main()
