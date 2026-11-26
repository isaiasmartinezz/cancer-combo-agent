"""Masked-drug-name ablation: the model sees "Drug 07" instead of every name a drug goes by.

If the agent does just as well with names hidden, its gains come from the mechanisms (drug class,
targets, literature content) rather than from recalling what it already knows about famous
drugs, or about this screen, from its training data.

Masking happens at the prompt boundary. Tables and text are masked on the way in, the model's
answers are unmasked on the way out, and everything stored (hypotheses, run log) uses real IDs.
Known limit: a brand name or synonym that is not in the names table can still slip through an
abstract. The default masker is the identity, so unmasked runs go through the same code path.
"""
import re

import numpy as np
import pandas as pd

MASK_SEED = 0  # code numbers are shuffled so they carry no alphabetical hint


class Masker:
    def __init__(self, names: dict[str, list[str]] | None = None, seed: int = MASK_SEED):
        """names: drug_id -> every name the drug goes by. None means no masking."""
        self.enabled = names is not None
        names = names or {}
        order = np.random.default_rng(seed).permutation(sorted(names))
        self.code = {drug: f"Drug {i + 1:02d}" for i, drug in enumerate(order)}
        self.drug = {code: drug for drug, code in self.code.items()}
        spellings = sorted({(n, d) for d, ns in names.items() for n in [d, *ns] if n},
                           key=lambda nd: -len(nd[0]))  # longest first: "SN-38" before "SN"
        self._spellings = [(re.compile(rf"(?<![\w-]){re.escape(n)}(?![\w-])", re.IGNORECASE),
                            self.code[d]) for n, d in spellings]

    @classmethod
    def from_tables(cls, drugs: pd.DataFrame, annotations: pd.DataFrame, **kwargs) -> "Masker":
        names: dict[str, set[str]] = {}
        for row in drugs.fillna("").itertuples():
            names.setdefault(row.drug_id, set()).update({row.drugcomb_name, row.generic_name})
        for row in annotations.fillna("").itertuples():
            names.setdefault(row.drug_id, set()).update({row.search_name, row.chembl_name})
        return cls({d: sorted(n - {"", "unknown"}) for d, n in names.items()}, **kwargs)

    def text(self, text: str) -> str:
        if not self.enabled:
            return text
        for pattern, code in self._spellings:
            text = pattern.sub(code, text)
        return text

    def name(self, drug: str) -> str:
        return self.code.get(drug, drug) if self.enabled else drug

    def unname(self, shown: str) -> str:
        return self.drug.get(shown, shown) if self.enabled else shown

    def table(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df
        out = df.copy()
        for col in ("drug_a", "drug_b"):
            if col in out:
                out[col] = out[col].map(self.name)
        return out
