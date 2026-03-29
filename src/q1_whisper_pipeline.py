from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from shared_helpers import FriendlyProjectPaths, load_json_file, polish_hindi_text, save_table, simple_wer


# ========== ROBUST TRANSCRIPT EXTRACTION ==========
def pull_transcript_text(transcription_payload) -> str:
    if isinstance(transcription_payload, dict):
        return str(
            transcription_payload.get("text")
            or transcription_payload.get("transcription")
            or transcription_payload.get("transcript")
            or ""
        ).strip()

    if isinstance(transcription_payload, list):
        chunks = []
        for item in transcription_payload:
            if isinstance(item, dict):
                text_piece = str(item.get("text") or item.get("transcription") or item.get("transcript") or "").strip()
                if text_piece:
                    chunks.append(text_piece)
            else:
                plain_piece = str(item).strip()
                if plain_piece:
                    chunks.append(plain_piece)
        return " ".join(chunks).strip()

    return str(transcription_payload or "").strip()


# ========== PREPARE TRAINING MANIFEST ==========
def build_training_manifest(project_root: Path) -> pd.DataFrame:
    paths = FriendlyProjectPaths(project_root)
    base_sheet = pd.read_excel(project_root / "dataset_1bujiO2N.xlsx", sheet_name="data")

    rows: List[Dict] = []
    for row in base_sheet.to_dict(orient="records"):
        old_audio_url = str(row.get("rec_url_gcp", ""))
        old_trans_url = str(row.get("transcription_url_gcp", ""))
        old_meta_url = str(row.get("metadata_url_gcp", ""))

        audio_name = old_audio_url.split("/")[-2] + "_" + old_audio_url.split("/")[-1]
        trans_name = old_trans_url.split("/")[-2] + "_" + old_trans_url.split("/")[-1]
        meta_name = old_meta_url.split("/")[-2] + "_" + old_meta_url.split("/")[-1]

        audio_path = paths.audio / audio_name
        trans_path = paths.transcriptions / trans_name
        meta_path = paths.metadata / meta_name

        if not (audio_path.exists() and trans_path.exists() and meta_path.exists()):
            continue

        trans_payload = load_json_file(trans_path)
        transcript_text = pull_transcript_text(trans_payload)

        rows.append(
            {
                "user_id": row.get("user_id"),
                "recording_id": row.get("recording_id"),
                "language": row.get("language"),
                "duration": float(row.get("duration", 0.0)),
                "audio_path": str(audio_path),
                "transcription_path": str(trans_path),
                "metadata_path": str(meta_path),
                "transcript": polish_hindi_text(transcript_text),
            }
        )

    manifest = pd.DataFrame(rows).dropna(subset=["audio_path", "transcript"])
    save_table(paths.artifacts / "q1_training_manifest.csv", manifest)
    return manifest


# ========== BUILD WHISPER COMMANDS ==========
def build_whisper_train_and_eval_commands(project_root: Path) -> Dict[str, str]:
    assignment_root = project_root / "asr_assignment"
    manifest_path = assignment_root / "artifacts" / "q1_training_manifest.csv"

    training_command = " ".join(
        [
            "python -m whisper_finetune_runner",
            f"--manifest \"{manifest_path}\"",
            "--model_id openai/whisper-small",
            "--language hindi",
            "--task transcribe",
            "--output_dir asr_assignment/artifacts/q1_whisper_finetuned",
            "--num_train_epochs 6",
            "--per_device_train_batch_size 8",
            "--gradient_accumulation_steps 2",
            "--learning_rate 1e-5",
            "--warmup_steps 250",
        ]
    )

    evaluation_command = " ".join(
        [
            "python -m whisper_fleurs_evaluator",
            "--baseline_model openai/whisper-small",
            "--finetuned_model asr_assignment/artifacts/q1_whisper_finetuned",
            "--fleurs_config hi_in",
            "--output_json asr_assignment/reports/q1_wer_report.json",
            "--output_csv asr_assignment/reports/q1_wer_report.csv",
        ]
    )

    return {"train": training_command, "evaluate": evaluation_command}


# ========== SAMPLE PERSISTENT ERRORS ==========
def produce_error_slice_for_taxonomy(project_root: Path, top_k: int = 25) -> pd.DataFrame:
    paths = FriendlyProjectPaths(project_root)
    asr_records_file = paths.artifacts / "q1_predictions.csv"

    if not asr_records_file.exists():
        demo = pd.DataFrame(
            [
                {
                    "recording_id": "demo_1",
                    "reference": "मुझे नौकरी मिल गई",
                    "prediction": "मुझे नोकरी मिल गई",
                },
                {
                    "recording_id": "demo_2",
                    "reference": "उन्होंने चौदह किताबें खरीदीं",
                    "prediction": "उन्होंने 14 किताबें खरीदी",
                },
            ]
        )
        save_table(paths.artifacts / "q1_predictions.csv", demo)

    rows = pd.read_csv(paths.artifacts / "q1_predictions.csv")
    rows["wer"] = rows.apply(lambda x: simple_wer(str(x["reference"]), str(x["prediction"])), axis=1)
    filtered = rows[rows["wer"] > 0].copy()

    filtered = filtered.sort_values(["wer", "recording_id"], ascending=[False, True])
    sampled = filtered.iloc[:: max(1, len(filtered) // max(top_k, 1))].head(top_k).copy()
    sampled["sampling_strategy"] = "systematic_stride_over_error_sorted_list"

    save_table(paths.reports / "q1_error_samples.csv", sampled)
    return sampled


# ========== WRITE QUESTION 1 NARRATIVE ==========
def write_question_one_report(project_root: Path) -> None:
    paths = FriendlyProjectPaths(project_root)
    commands = build_whisper_train_and_eval_commands(project_root)

    report = {
        "question": "Q1",
        "what_was_preprocessed": [
            "verified local presence of audio/transcription/metadata files",
            "cleaned transcripts with whitespace and unicode normalization",
            "built a single training manifest with local file paths",
        ],
        "baseline_vs_finetuned_evaluation_plan": {
            "dataset": "google/fleurs, config hi_in, split test",
            "metrics": ["WER"],
            "baseline_model": "openai/whisper-small",
            "finetuned_model_output": "asr_assignment/artifacts/q1_whisper_finetuned",
        },
        "commands": commands,
        "notes": "Run the commands in sequence, then place predictions in q1_predictions.csv to refresh error taxonomy files.",
    }

    paths.reports.mkdir(parents=True, exist_ok=True)
    with (paths.reports / "q1_plan_and_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 1 pipeline helper")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    args = parser.parse_args()

    manifest = build_training_manifest(args.project_root)
    produce_error_slice_for_taxonomy(args.project_root, top_k=25)
    write_question_one_report(args.project_root)

    print(f"Q1 manifest rows: {len(manifest)}")
    print("Q1 outputs written to asr_assignment/artifacts and asr_assignment/reports")


if __name__ == "__main__":
    main()
