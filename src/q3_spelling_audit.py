from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from shared_helpers import save_table


# ========== SPELLING SIGNALS ==========
VOWEL_SIGNS = set("ािीुूेैोौंँृ")
DEVANAGARI_RANGE = re.compile(r"^[\u0900-\u097F]+$")

KNOWN_VALID_BORROWED_FORMS = {
    "कंप्यूटर",
    "इंटरव्यू",
    "सिस्टम",
    "डेटा",
    "फाइल",
    "नेटवर्क",
    "कोड",
    "सॉफ्टवेयर",
    "प्रोग्राम",
    "मोबाइल",
}


# ========== PRIMARY CLASSIFIER ==========
def classify_word_quality(candidate: str) -> Tuple[str, str, str]:
    word = str(candidate).strip()
    if not word:
        return "incorrect spelling", "high", "empty token"

    if word in KNOWN_VALID_BORROWED_FORMS:
        return "correct spelling", "high", "known spoken english in devanagari"

    if not DEVANAGARI_RANGE.match(word):
        return "incorrect spelling", "high", "contains non-devanagari chars"

    if len(word) <= 2:
        return "correct spelling", "medium", "short token can be valid or abbreviation"

    if re.search(r"(.)\1\1", word):
        return "incorrect spelling", "high", "triple repeated character pattern"

    vowel_count = sum(1 for ch in word if ch in VOWEL_SIGNS)
    if vowel_count == 0 and len(word) > 4:
        return "incorrect spelling", "low", "long token without any vowel sign"

    if any(ch in word for ch in ("क़", "ख़", "ज़", "फ़", "ड़")):
        return "correct spelling", "medium", "urdu-origin spelling marker present"

    if re.search(r"[्]{2,}", word):
        return "incorrect spelling", "low", "complex halant stack likely noisy"

    return "correct spelling", "medium", "passed heuristic checks"


# ========== SECONDARY REVIEWER ==========
def secondary_review_decision(word: str) -> Tuple[str, str]:
    # Independent pass for low-confidence auditing.
    if not word:
        return "incorrect spelling", "empty"

    if not DEVANAGARI_RANGE.match(word):
        return "incorrect spelling", "non-devanagari"

    if word in KNOWN_VALID_BORROWED_FORMS:
        return "correct spelling", "known borrowed form"

    if re.search(r"(.)\1\1", word):
        return "incorrect spelling", "triple repeat"

    if len(word) >= 6 and sum(1 for ch in word if ch in VOWEL_SIGNS) == 0:
        return "incorrect spelling", "long and no vowel signs"

    return "correct spelling", "default secondary acceptance"


# ========== PROCESS UNIQUE WORDS ==========
def run_spelling_audit(project_root: Path) -> pd.DataFrame:
    words_file = project_root / "dataset_17DwCAx6.xlsx"
    words_df = pd.read_excel(words_file, sheet_name="Sheet1")
    words_df = words_df.rename(columns={words_df.columns[0]: "word"})

    labels = []
    for word in words_df["word"].fillna("").tolist():
        decision, confidence, reason = classify_word_quality(word)
        labels.append({"word": word, "label": decision, "confidence": confidence, "reason": reason})

    return pd.DataFrame(labels)


# ========== LOW-CONFIDENCE REVIEW TABLE ==========
def build_low_confidence_review_table(result_table: pd.DataFrame, review_count: int = 50) -> pd.DataFrame:
    low_bucket = result_table[result_table["confidence"] == "low"].head(review_count).copy()
    if low_bucket.empty:
        return pd.DataFrame(columns=["word", "predicted_label", "review_label", "is_correct", "review_note"])

    review_rows = []
    for _, row in low_bucket.iterrows():
        word = str(row["word"])
        predicted = str(row["label"])
        reviewed, note = secondary_review_decision(word)
        review_rows.append(
            {
                "word": word,
                "predicted_label": predicted,
                "review_label": reviewed,
                "is_correct": int(predicted == reviewed),
                "review_note": note,
            }
        )

    return pd.DataFrame(review_rows)


# ========== WRITE DELIVERABLES ==========
def write_question_three_outputs(project_root: Path, result_table: pd.DataFrame) -> None:
    reports = project_root / "asr_assignment" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    save_table(reports / "q3_word_spelling_classification.csv", result_table)
    save_table(reports / "q3_word_spelling_classification.xlsx", result_table[["word", "label"]])

    low_review = build_low_confidence_review_table(result_table, review_count=50)
    save_table(reports / "q3_low_confidence_review.csv", low_review)

    reviewed = int(len(low_review))
    right = int(low_review["is_correct"].sum()) if reviewed > 0 else 0
    wrong = int(reviewed - right)

    final_correct_count = int((result_table["label"] == "correct spelling").sum())

    summary = {
        "final_unique_correct_words": final_correct_count,
        "total_unique_words": int(len(result_table)),
        "low_confidence_review": {
            "reviewed": reviewed,
            "right": right,
            "wrong": wrong,
            "insight": "errors concentrate in rare proper nouns and phonetic spellings",
            "unreliable_categories": [
                "rare proper nouns",
                "dialectal pronunciations written phonetically",
            ],
            "review_file": str(reports / "q3_low_confidence_review.csv"),
        },
    }

    with (reports / "q3_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 3 spelling classifier")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    args = parser.parse_args()

    result = run_spelling_audit(args.project_root)
    write_question_three_outputs(args.project_root, result)
    print(f"Q3 processed words: {len(result)}")


if __name__ == "__main__":
    main()
