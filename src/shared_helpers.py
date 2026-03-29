from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


# ========== FRIENDLY PATHS ==========
@dataclass
class FriendlyProjectPaths:
    root: Path

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def audio(self) -> Path:
        return self.downloads / "audio"

    @property
    def transcriptions(self) -> Path:
        return self.downloads / "transcriptions"

    @property
    def metadata(self) -> Path:
        return self.downloads / "metadata"

    @property
    def artifacts(self) -> Path:
        return self.root / "asr_assignment" / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "asr_assignment" / "reports"


# ========== TINY FILE HELPERS ==========
def load_json_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json_file(file_path: Path, payload: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def save_table(file_path: Path, table: pd.DataFrame) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix.lower() == ".csv":
        table.to_csv(file_path, index=False)
    else:
        table.to_excel(file_path, index=False)


# ========== TEXT NORMALIZATION ==========
def polish_hindi_text(raw_text: str) -> str:
    cleaned = (raw_text or "").strip().lower()
    cleaned = re.sub(r"[\u200c\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def quick_tokenize(sentence: str) -> List[str]:
    return [piece for piece in re.split(r"\s+", polish_hindi_text(sentence)) if piece]


def simple_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = quick_tokenize(reference)
    hyp_tokens = quick_tokenize(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1] / max(1, len(ref_tokens))


# ========== DATASET TABLE LOADER ==========
def load_main_dataset_sheet(root_folder: Path) -> pd.DataFrame:
    data_file = root_folder / "dataset_1bujiO2N.xlsx"
    return pd.read_excel(data_file, sheet_name="data")


def find_existing_files(folder: Path, suffix: str) -> Iterable[Path]:
    return folder.glob(f"*{suffix}")
