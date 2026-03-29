from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shared_helpers import polish_hindi_text, save_table, simple_wer


# ========== LOAD FLEURS ==========
def load_fleurs_test_split(config_name: str, max_samples: int | None = None):
    from datasets import load_dataset

    split = load_dataset("google/fleurs", config_name, split="test", trust_remote_code=True)
    if max_samples is not None and max_samples > 0:
        split = split.select(range(min(max_samples, len(split))))
    return split


# ========== SIMPLE RESAMPLING ==========
def resample_to_16k(audio, sample_rate: int):
    import numpy as np

    if sample_rate == 16000:
        return audio.astype("float32")
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(duration * 16000))
    src_idx = np.linspace(0, len(audio) - 1, num=len(audio), dtype="float32")
    dst_idx = np.linspace(0, len(audio) - 1, num=target_len, dtype="float32")
    return np.interp(dst_idx, src_idx, audio).astype("float32")


# ========== RUN MODEL TRANSCRIPTION ==========
def transcribe_batch(model_path: str, fleurs_rows):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")

    outputs = []
    for row in fleurs_rows:
        audio = row["audio"]["array"]
        sample_rate = int(row["audio"]["sampling_rate"])
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = resample_to_16k(audio, sample_rate)
        if len(audio) > 30 * 16000:
            audio = audio[: 30 * 16000]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=220)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        outputs.append(
            {
                "id": row.get("id"),
                "reference": polish_hindi_text(str(row.get("transcription", ""))),
                "prediction": polish_hindi_text(str(text)),
            }
        )
    return pd.DataFrame(outputs)


# ========== EVALUATE BOTH MODELS ==========
def evaluate_two_models(args: argparse.Namespace) -> pd.DataFrame:
    max_samples = None if args.max_eval_samples <= 0 else args.max_eval_samples
    fleurs = load_fleurs_test_split(args.fleurs_config, max_samples=max_samples)

    baseline_rows = transcribe_batch(args.baseline_model, fleurs)
    baseline_rows["wer"] = baseline_rows.apply(lambda x: simple_wer(x["reference"], x["prediction"]), axis=1)
    baseline_wer = float(baseline_rows["wer"].mean())

    tuned_rows = transcribe_batch(args.finetuned_model, fleurs)
    tuned_rows["wer"] = tuned_rows.apply(lambda x: simple_wer(x["reference"], x["prediction"]), axis=1)
    tuned_wer = float(tuned_rows["wer"].mean())

    report = pd.DataFrame(
        [
            {"model": "whisper-small-baseline", "wer": baseline_wer, "eval_samples": len(baseline_rows)},
            {"model": "whisper-small-finetuned", "wer": tuned_wer, "eval_samples": len(tuned_rows)},
        ]
    )

    artifact_folder = (args.output_csv.parent.parent / "artifacts")
    artifact_folder.mkdir(parents=True, exist_ok=True)
    tuned_rows.to_csv(artifact_folder / "q1_predictions.csv", index=False)
    baseline_rows.to_csv(artifact_folder / "q1_baseline_predictions.csv", index=False)
    return report


# ========== COMMAND LINE ==========
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--fleurs_config", type=str, default="hi_in")
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_two_models(args)
    save_table(args.output_csv, report)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(report.to_dict(orient="records"), stream, ensure_ascii=False, indent=2)

    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
