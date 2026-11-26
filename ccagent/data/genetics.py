"""Cell-line genetics features from Cell Model Passports, for the response model and the agent.

- Mutations: CMP cancer-driver mutations (non-duplicate rows). The panel is every gene mutated in
  at least MIN_LINES of our lines, plus every drug-target gene mutated in any of them.
  1 / 0 per line, or unknown (NaN) when the line has no mutation data.
- Copy number: amplification / deletion of each drug-target gene (any CMP source).
- Target expression: per line and drug, the mean expression of the drug's target genes.
  CMP's `rsem_tpm` is already log2-scaled, so it is used as-is (checked: see LOG2_MAX).
- Summaries: one plain-text line per cell line for the language model.
Unknowns stay unknown: no CMP match, no data, or a gene listed in `unknown_genes`.
Uses genetics only, never screen outcomes, so it can include the held-out lines.
"""
import numpy as np
import pandas as pd

MIN_LINES = 3
LOG2_MAX = 25  # a value above this would mean rsem_tpm is not log2-scaled


def _genes(value: str) -> list[str]:
    return [] if value in ("", "unknown") or pd.isna(value) else value.split("; ")


def build_features(cell_map: pd.DataFrame, models: pd.DataFrame, mutations: pd.DataFrame,
                   cnv: pd.DataFrame, rnaseq: pd.DataFrame, drug_targets: dict[str, str]):
    """Returns (features, target_expression, summaries), each indexed by cell_line."""
    targets = {d: _genes(g) for d, g in drug_targets.items()}
    all_targets = sorted({g for gs in targets.values() for g in gs})
    ids = dict(zip(cell_map["cell_line"], cell_map["cmp_model_id"].fillna(""), strict=True))
    unknown = {line: set(_genes(u.replace(",", "; "))) for line, u in
               zip(cell_map["cell_line"], cell_map["unknown_genes"].fillna(""), strict=True)}
    ours = {i for i in ids.values() if i}
    models = models.set_index("model_id")

    mut = mutations[(mutations["cancer_driver"].astype(str).isin(["t", "True"]))
                    & (mutations["duplicate"].astype(str).isin(["f", "False"]))
                    & mutations["model_id"].isin(ours)]
    counts = mut.groupby("gene_symbol")["model_id"].nunique()
    recurrent = set(counts[counts >= MIN_LINES].index)
    panel = sorted(recurrent | (set(mut["gene_symbol"]) & set(all_targets)))

    keep = rnaseq["duplicate"].astype(str).isin(["f", "False"]) & rnaseq["model_id"].isin(ours)
    rna = rnaseq[keep]
    expr = rna.assign(v=rna["rsem_tpm"].astype(float)).groupby(["model_id", "gene_symbol"])["v"] \
        .mean().unstack()
    if expr.max().max() > LOG2_MAX:
        raise ValueError("rsem_tpm looks unscaled; expected log2 values")

    feats, texpr, summaries = [], {}, []
    for line, mid in ids.items():
        sequenced = bool(mid) and str(models.loc[mid, "mutation_data"]) == "True"
        msi = models.loc[mid, "msi_status"] if mid and pd.notna(models.loc[mid, "msi_status"]) \
            else "unknown"
        m = mut[mut["model_id"] == mid]
        mutated = set(m["gene_symbol"])
        c = cnv[cnv["model_id"] == mid].groupby("symbol")["cn_category"].agg(set).to_dict() \
            if mid else {}
        row = {"cell_line": line, "cmp_model_id": mid or "unknown",
               "has_genetics": int(sequenced), "msi_status": msi}
        for g in panel:
            row[f"mut_{g}"] = int(g in mutated) if sequenced else np.nan
        for g in all_targets:
            row[f"amp_{g}"] = int("Amplification" in c.get(g, set())) if c else np.nan
            row[f"del_{g}"] = int("Deletion" in c.get(g, set())) if c else np.nan
        for g in unknown[line]:
            for key in (f"mut_{g}", f"amp_{g}", f"del_{g}"):
                if key in row:
                    row[key] = np.nan
        feats.append(row)

        texpr[line] = {}
        for drug, gs in targets.items():
            have = [g for g in gs if g in expr.columns and g not in unknown[line]]
            texpr[line][drug] = expr.loc[mid, have].mean() if mid in expr.index and have else np.nan

        if not mid:
            text = "genetics unknown (no confident Cell Model Passports match)"
            summaries.append({"cell_line": line, "summary": text})
            continue
        hits = m[m["gene_symbol"].isin(panel) & ~m["gene_symbol"].isin(unknown[line])] \
            .drop_duplicates(["gene_symbol", "protein_mutation"])

        def change(p):
            return "(protein change not given)" if pd.isna(p) or p in ("p.?", "-", "") \
                else p.removeprefix("p.")

        muts = ", ".join(f"{g} {change(p)}" for g, p in
                         zip(hits["gene_symbol"], hits["protein_mutation"], strict=True))
        muts = muts or ("none in panel" if sequenced else "unknown")
        amps = sorted(g for g in all_targets if "Amplification" in c.get(g, set()))
        amp_text = (", ".join(amps) or "none") if c else "unknown"
        text = (f"{models.loc[mid, 'cancer_type']}; MSI: {msi}; driver mutations: {muts}; "
                f"amplified drug targets: {amp_text}")
        for g in sorted(unknown[line]):
            text += f"; {g}: unknown (engineered derivative)"
        summaries.append({"cell_line": line, "summary": text})

    target_expression = pd.DataFrame(texpr).T
    target_expression.index.name = "cell_line"
    return (pd.DataFrame(feats).set_index("cell_line"), target_expression,
            pd.DataFrame(summaries).set_index("cell_line"))
