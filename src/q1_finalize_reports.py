from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from shared_helpers import save_table, simple_wer


# ========== ERROR LABELS ==========
def primary_error_label(reference: str, prediction: str, wer: float) -> str:
    ref_tokens = str(reference).split()
    hyp_tokens = str(prediction).split()

    if bool(re.search(r"\d", str(reference))) != bool(re.search(r"\d", str(prediction))):
        return "number_format_mismatch"

    ratio = len(hyp_tokens) / max(1, len(ref_tokens))
    if ratio < 0.08:
        return "extreme_deletion"
    if ratio < 0.18:
        return "heavy_deletion"
    if ratio < 0.35:
        return "moderate_deletion"

    if wer > 0.9:
        return "lexical_substitution"
    return "mild_wording_difference"


def fallback_band_label(wer: float) -> str:
    if wer > 1.2:
        return "high_wer_band"
    if wer > 0.8:
        return "mid_wer_band"
    return "low_wer_band"


# ========== POST FIX ==========
def normalize_numbers_and_repetitions(text: str) -> str:
    number_map = {
        "एक": "1",
        "दो": "2",
        "तीन": "3",
        "चार": "4",
        "पांच": "5",
        "पाँच": "5",
        "छह": "6",
        "सात": "7",
        "आठ": "8",
        "नौ": "9",
        "दस": "10",
    }
    tokens = str(text).split()
    mapped = [number_map.get(t, t) for t in tokens]

    deduped = []
    for t in mapped:
        if not deduped or deduped[-1] != t:
            deduped.append(t)
    return " ".join(deduped)


# ========== FINALIZE REPORTS ==========
def build_final_q1_outputs(project_root: Path) -> None:
    reports = project_root / "asr_assignment" / "reports"
    artifacts = project_root / "asr_assignment" / "artifacts"

    pred = pd.read_csv(artifacts / "q1_predictions.csv")
    pred["wer"] = pred.apply(lambda x: simple_wer(str(x["reference"]), str(x["prediction"])), axis=1)
    errors = pred[pred["wer"] > 0].copy()

    if len(errors) == 0:
        raise RuntimeError("No error rows found in q1_predictions.csv")

    errors["error_type"] = errors.apply(lambda x: primary_error_label(x["reference"], x["prediction"], float(x["wer"])), axis=1)

    eligible = [k for k, v in Counter(errors["error_type"]).items() if v >= 3]
    if len(eligible) < 3:
        errors["error_type"] = errors["wer"].apply(fallback_band_label)

    # ========== 25 SYSTEMATIC SAMPLES ==========
    ordered = errors.sort_values(["wer", "id"], ascending=[False, True])
    stride = max(1, len(ordered) // 25)
    sampled = ordered.iloc[::stride].head(25).copy()
    if len(sampled) < 25:
        sampled = pd.concat([sampled, ordered.head(25 - len(sampled))], ignore_index=True)
    sampled["sampling_strategy"] = "systematic_stride_over_error_sorted_rows"
    save_table(reports / "q1_error_samples.csv", sampled)

    # ========== TAXONOMY (3-5 each) ==========
    counts = Counter(errors["error_type"])
    top3 = [c for c, n in counts.most_common() if n >= 3][:3]
    taxonomy = {}
    for cat in top3:
        chunk = errors[errors["error_type"] == cat].head(5)
        if len(chunk) < 3:
            continue
        taxonomy[cat] = [
            {
                "reference": str(r.reference),
                "prediction": str(r.prediction),
                "reasoning": f"classified as {cat} based on deletion ratio/number mismatch/wer band",
            }
            for r in chunk.itertuples(index=False)
        ]

    with (reports / "q1_error_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    # ========== TOP-3 FIX PROPOSALS ==========
    proposal = {
        "top_3_error_types": top3,
        "proposed_fixes": [
            {
                "type": top3[0] if len(top3) > 0 else "high_wer_band",
                "fix": "long-form chunking with overlap and merge-decoding to reduce deletion-heavy outputs",
            },
            {
                "type": top3[1] if len(top3) > 1 else "mid_wer_band",
                "fix": "domain phrase biasing and lexicon-constrained rescoring",
            },
            {
                "type": top3[2] if len(top3) > 2 else "low_wer_band",
                "fix": "post-decoding numeric normalization and repetition collapse",
            },
        ],
    }
    with (reports / "q1_top_fixes.json").open("w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)

    # ========== IMPLEMENT ONE FIX ==========
    target = errors.head(25).copy()
    target["fixed_prediction"] = target["prediction"].apply(normalize_numbers_and_repetitions)
    target["wer_before"] = target.apply(lambda x: simple_wer(str(x["reference"]), str(x["prediction"])), axis=1)
    target["wer_after"] = target.apply(lambda x: simple_wer(str(x["reference"]), str(x["fixed_prediction"])), axis=1)
    target["delta"] = target["wer_after"] - target["wer_before"]
    save_table(reports / "q1_fix_before_after.csv", target)

    summary = {
        "sampled_error_count": int(len(sampled)),
        "taxonomy_categories": list(taxonomy.keys()),
        "targeted_subset_size": int(len(target)),
        "avg_wer_before": float(target["wer_before"].mean()),
        "avg_wer_after": float(target["wer_after"].mean()),
    }
    with (reports / "q1_completion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    args = parser.parse_args()

    build_final_q1_outputs(args.project_root)
    print("Q1 final reports regenerated.")


if __name__ == "__main__":
    main()
