"""Match the screen's cell-line names to Cell Model Passports (CMP) models.

Automatic rule: names equal after ignoring case and punctuation, exactly one CMP model, and the
tissue agrees. Anything else must be decided by hand in data/reference/cell_line_overrides.csv,
with a reason. An override names the CMP model it expects, so a renamed or removed CMP record
fails the build instead of silently changing a line's genetics. No match means "unknown".
"""
import re

import pandas as pd

MAP_COLS = ["cell_line", "tissue", "cmp_model_id", "cmp_model_name", "cmp_tissue",
            "cmp_cancer_type", "match", "unknown_genes", "note"]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def match_cell_lines(lines: pd.DataFrame, models: pd.DataFrame,
                     overrides: pd.DataFrame) -> pd.DataFrame:
    """lines: cell_line, tissue. models: CMP model list. Returns one row per line (MAP_COLS)."""
    models = models.set_index("model_id")
    by_name = models.assign(key=models["model_name"].map(normalize)).groupby("key")
    overrides = overrides.fillna("").set_index("cell_line")
    rows = []
    for line, tissue in zip(lines["cell_line"], lines["tissue"], strict=True):
        row = dict.fromkeys(MAP_COLS, "")
        row.update(cell_line=line, tissue=tissue)
        if line in overrides.index:
            o = overrides.loc[line]
            row.update(match="override" if o.cmp_model_id else "none",
                       unknown_genes=o.unknown_genes, note=o.reason)
            if o.cmp_model_id:
                found = models.loc[o.cmp_model_id, "model_name"] if o.cmp_model_id in models.index \
                    else None
                if found != o.expected_cmp_name:
                    raise ValueError(f"{line}: CMP {o.cmp_model_id} is {found!r}, "
                                     f"expected {o.expected_cmp_name!r}")
                row["cmp_model_id"] = o.cmp_model_id
        else:
            key = normalize(line)
            hits = by_name.get_group(key).index.tolist() if key in by_name.groups else []
            if len(hits) != 1:
                raise ValueError(f"{line}: {len(hits)} CMP name matches; add an override")
            if normalize(models.loc[hits[0], "tissue"]) != normalize(tissue):
                raise ValueError(f"{line}: tissue {tissue} vs CMP {models.loc[hits[0], 'tissue']}")
            row.update(cmp_model_id=hits[0], match="name")
        if row["cmp_model_id"]:
            m = models.loc[row["cmp_model_id"]]
            row.update(cmp_model_name=m["model_name"], cmp_tissue=m["tissue"],
                       cmp_cancer_type=m["cancer_type"])
        rows.append(row)
    return pd.DataFrame(rows, columns=MAP_COLS)
