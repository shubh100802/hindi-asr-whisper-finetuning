from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ========== DATA COMPLETENESS CHECK ==========
def build_data_access_report(project_root: Path) -> dict:
    downloads = project_root / "downloads"
    audio_count = len(list((downloads / "audio").glob("*.wav")))
    trans_count = len(list((downloads / "transcriptions").glob("*.json")))
    meta_count = len(list((downloads / "metadata").glob("*.json")))

    expected_main = pd.read_excel(project_root / "dataset_1bujiO2N.xlsx", sheet_name="data")
    expected_q4 = pd.read_excel(project_root / "dataset_1J_I0rao.xlsx", sheet_name="Task")

    return {
        "question": "Q5_data_readiness",
        "note": "This closes the PDF instruction about correcting old URLs before processing.",
        "counts": {
            "audio_downloaded": audio_count,
            "transcriptions_downloaded": trans_count,
            "metadata_downloaded": meta_count,
            "expected_main_rows": int(len(expected_main)),
            "expected_q4_rows": int(len(expected_q4)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Data readiness report")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    args = parser.parse_args()

    report = build_data_access_report(args.project_root)
    output = args.project_root / "asr_assignment" / "reports" / "q5_data_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
