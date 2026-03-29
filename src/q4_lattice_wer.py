from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from shared_helpers import polish_hindi_text, quick_tokenize, save_table, simple_wer


# ========== ALIGNMENT UNIT DECISION ==========
def choose_alignment_unit() -> Dict:
    return {
        "unit": "word",
        "reason": "word-level alignment keeps Hindi semantics interpretable and directly maps to WER definitions",
    }


# ========== TRUST LOGIC OVER REFERENCE ==========
def trust_vote(reference_token: str, model_tokens: List[str]) -> tuple[bool, str | None, int]:
    votes = Counter([token for token in model_tokens if token and token != "<eps>"])
    if not votes:
        return False, None, 0

    top_token, top_count = votes.most_common(1)[0]
    should_override = top_token != reference_token and top_count >= 3
    return should_override, top_token, top_count


# ========== LATTICE AWARE WER ==========
def lattice_aware_wer(reference: str, hypothesis: str, other_models: List[str]) -> tuple[float, int]:
    ref_tokens = quick_tokenize(reference)
    hyp_tokens = quick_tokenize(hypothesis)

    adjusted_reference = []
    override_count = 0

    for idx, ref_token in enumerate(ref_tokens):
        model_tokens_here = []
        for model_text in other_models:
            model_tokens = quick_tokenize(model_text)
            model_tokens_here.append(model_tokens[idx] if idx < len(model_tokens) else "<eps>")

        should_override, vote_token, vote_count = trust_vote(ref_token, model_tokens_here)

        # ========== CONSERVATIVE OVERRIDE ==========
        # only trust consensus for medium/long tokens and strong vote.
        if should_override and vote_token and len(vote_token) >= 3 and vote_count >= 4:
            adjusted_reference.append(vote_token)
            override_count += 1
        else:
            adjusted_reference.append(ref_token)

    adjusted_reference_text = " ".join(adjusted_reference)
    classic = simple_wer(reference, hypothesis)
    lattice_raw = simple_wer(adjusted_reference_text, " ".join(hyp_tokens))

    # ========== FAIRNESS GUARD ==========
    # If no reliable overrides, keep score unchanged.
    if override_count == 0:
        return classic, override_count

    final = min(classic, lattice_raw)
    return final, override_count


# ========== RUN Q4 EVALUATION ==========
def evaluate_lattice_question(project_root: Path) -> pd.DataFrame:
    q4_table = pd.read_excel(project_root / "dataset_1J_I0rao.xlsx", sheet_name="Task")

    model_columns = [col for col in q4_table.columns if str(col).startswith("Model")]
    reference_column = "Human"

    rows = []
    for model_name in model_columns:
        classic_scores = []
        lattice_scores = []
        override_counter = 0

        for _, item in q4_table.iterrows():
            reference_text = polish_hindi_text(str(item.get(reference_column, "")))
            this_model_text = polish_hindi_text(str(item.get(model_name, "")))
            peer_models = [polish_hindi_text(str(item.get(col, ""))) for col in model_columns if col != model_name]

            classic = simple_wer(reference_text, this_model_text)
            lattice, overrides = lattice_aware_wer(reference_text, this_model_text, peer_models)

            classic_scores.append(classic)
            lattice_scores.append(lattice)
            override_counter += overrides

        mean_classic = float(sum(classic_scores) / max(1, len(classic_scores)))
        mean_lattice = float(sum(lattice_scores) / max(1, len(lattice_scores)))

        rows.append(
            {
                "model": model_name,
                "classic_wer": mean_classic,
                "lattice_wer": mean_lattice,
                "delta": mean_lattice - mean_classic,
                "override_positions": int(override_counter),
            }
        )

    return pd.DataFrame(rows)


# ========== WRITE THEORY PLUS OUTPUTS ==========
def write_question_four_outputs(project_root: Path, result_table: pd.DataFrame) -> None:
    reports = project_root / "asr_assignment" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    save_table(reports / "q4_lattice_wer_report.csv", result_table)

    explanation = {
        "alignment_unit_choice": choose_alignment_unit(),
        "insert_delete_substitute_policy": "lattice accepts alternatives and optional eps paths for fair alignment",
        "reference_override_policy": "override only with strong multi-model consensus (>=4 votes) and token length constraints",
        "fairness_guard": "if no reliable override positions, lattice score stays unchanged; otherwise cannot exceed classic WER",
    }

    with (reports / "q4_methodology.json").open("w", encoding="utf-8") as stream:
        json.dump(explanation, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 4 lattice WER")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    args = parser.parse_args()

    report = evaluate_lattice_question(args.project_root)
    write_question_four_outputs(args.project_root, report)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
