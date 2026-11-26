"""Turning an experiment (two drugs + a cell line) into numbers for the response model.

Drugs: Morgan fingerprint of the structure (RDKit, radius 2, 256 bits), a multi-hot of the ChEMBL
targets, and the drug's own identity. The two drugs' vectors are added, so the order of a pair
cannot matter.
Cell lines: identity, tissue, and (optionally) the genetics from Cell Model Passports.
Target expression: per experiment, the smaller and larger of the two drugs' target-gene
expression in that line, which again does not depend on the order of the pair.
Unknown values become the column mean (expression) or 0 (mutations, copy number); `has_genetics`
tells the model which lines those are.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
FINGERPRINT_BITS = 256
FINGERPRINT_RADIUS = 2
ANNOTATIONS = Path("data/reference/drug_annotations.csv")
PROCESSED = Path("data/processed")


def drug_features(annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per drug: fingerprint bits, target multi-hot, identity one-hot."""
    annotations = annotations.set_index("drug_id")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    bits = []
    for smiles in annotations["smiles"]:
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) \
            and smiles != "unknown" else None
        bits.append(generator.GetFingerprintAsNumPy(molecule).astype(float) if molecule
                    else np.zeros(FINGERPRINT_BITS))
    fingerprints = pd.DataFrame(bits, index=annotations.index).add_prefix("fp_")
    targets = (annotations["targets"].fillna("unknown").str.get_dummies(sep="; ")
               .drop(columns=["unknown"], errors="ignore").add_prefix("target_"))
    identity = pd.get_dummies(annotations.index.to_series(), dtype=float).add_prefix("drug_")
    return pd.concat([fingerprints, targets, identity], axis=1).astype(float)


class FeatureSpace:
    """Fixed feature layout for a set of cell lines; built once per run."""

    def __init__(self, cell_lines, annotations: pd.DataFrame, tissue: dict | None = None,
                 genetics: pd.DataFrame | None = None,
                 target_expression: pd.DataFrame | None = None):
        self.drugs = drug_features(annotations)
        lines = sorted(set(cell_lines))
        table = pd.get_dummies(pd.Series(lines, index=lines), dtype=float).add_prefix("line_")
        if tissue:
            labels = pd.Series({line: tissue.get(line, "unknown") for line in lines})
            table = table.join(pd.get_dummies(labels, dtype=float).add_prefix("tissue_"))
        if genetics is not None:
            numeric = genetics.reindex(lines).select_dtypes("number").fillna(0.0)
            table = table.join(numeric.add_prefix("gene_"))
        self.lines = table
        self.expression = None
        if target_expression is not None:
            expression = target_expression.reindex(lines)
            self.expression = expression.fillna(expression.mean()).fillna(0.0)

    @classmethod
    def from_files(cls, cell_lines, tissue=None, with_genetics=True,
                   annotations_path: Path = ANNOTATIONS, processed: Path = PROCESSED):
        genetics = expression = None
        if with_genetics and (processed / "cell_line_features.csv").exists():
            genetics = pd.read_csv(processed / "cell_line_features.csv", index_col="cell_line")
            expression = pd.read_csv(processed / "target_expression.csv", index_col="cell_line")
        return cls(cell_lines, pd.read_csv(annotations_path, dtype=str), tissue=tissue,
                   genetics=genetics, target_expression=expression)

    def transform(self, experiments: pd.DataFrame) -> np.ndarray:
        first = self.drugs.loc[experiments["drug_a"]].to_numpy()
        second = self.drugs.loc[experiments["drug_b"]].to_numpy()
        parts = [first + second, self.lines.loc[experiments["cell_line"]].to_numpy()]
        if self.expression is not None:
            values = self.expression.to_numpy()
            row = self.expression.index.get_indexer(experiments["cell_line"])
            column_a = self.expression.columns.get_indexer(experiments["drug_a"])
            column_b = self.expression.columns.get_indexer(experiments["drug_b"])
            a = np.where(column_a >= 0, values[row, column_a], 0.0)
            b = np.where(column_b >= 0, values[row, column_b], 0.0)
            parts.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))
        return np.hstack(parts)
