"""Masked-drug-name ablation: the model sees "Drug 07" instead of every name a drug goes by.

If the agent does just as well with names hidden, its gains come from the mechanisms (drug class,
targets, literature content) rather than from recalling what it already knows about famous
drugs, or about this screen, from its training data.

Masking happens at the prompt boundary. Tables and text are masked on the way in, the model's
answers are unmasked on the way out, and everything stored (hypotheses, run log) uses real IDs.
Names come from the drug tables plus every PubChem synonym (`data/reference/drug_synonyms.csv`),
so brand names such as Taxol are masked too. A name PubChem does not list can still slip through.
The default masker is the identity, so unmasked runs go through the same code path.
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
        self._code_of: dict[str, str] = {}  # lowercased spelling -> code
        for drug, spellings in names.items():
            for spelling in [drug, *spellings]:
                if spelling:
                    self._code_of.setdefault(spelling.lower(), self.code[drug])
        # Index spellings by their first word, so each text is only checked against the few
        # spellings that could occur in it (there are thousands once synonyms are included).
        self._by_word: dict[str, list[str]] = {}
        for spelling in self._code_of:
            first = re.search(r"\w+", spelling)
            self._by_word.setdefault(first.group(0) if first else spelling, []).append(spelling)

    @classmethod
    def from_tables(cls, drugs: pd.DataFrame, annotations: pd.DataFrame,
                    synonyms: dict[str, list[str]] | None = None, **kwargs) -> "Masker":
        """Names from the drug tables, plus PubChem synonyms when given."""
        names: dict[str, set[str]] = {}
        for row in drugs.fillna("").itertuples():
            names.setdefault(row.drug_id, set()).update({row.drugcomb_name, row.generic_name})
        for row in annotations.fillna("").itertuples():
            names.setdefault(row.drug_id, set()).update({row.search_name, row.chembl_name})
        for drug, extra in (synonyms or {}).items():
            names.setdefault(drug, set()).update(extra)
        return cls({d: sorted(n - {"", "unknown"}) for d, n in names.items()}, **kwargs)

    def text(self, text: str) -> str:
        if not self.enabled:
            return text
        words = set(re.findall(r"\w+", text.lower()))
        candidates = [s for w in words for s in self._by_word.get(w, ())]
        if not candidates:
            return text
        candidates.sort(key=len, reverse=True)  # longest first: "SN-38" before "SN"
        pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, candidates)) +
                             r")(?!\w)", re.IGNORECASE)
        return pattern.sub(lambda m: self._code_of[m.group(0).lower()], text)

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
