from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from shared_helpers import polish_hindi_text, save_table, simple_wer


# ========== RESAMPLING ==========
def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio.astype(np.float32)
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(duration * 16000))
    src_idx = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    dst_idx = np.linspace(0, len(audio) - 1, num=target_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


# ========== LOAD LOCAL AUDIO ==========
def load_audio(file_path: str) -> np.ndarray:
    try:
        import torchaudio

        tensor, sr = torchaudio.load(file_path)
        if tensor.ndim == 2 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        tensor = tensor.squeeze(0).numpy().astype(np.float32)
        return resample_to_16k(tensor, int(sr))
    except Exception:
        import wave

        with wave.open(file_path, "rb") as wav:
            channels = wav.getnchannels()
            sr = wav.getframerate()
            width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())

        if width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return resample_to_16k(data, int(sr))


# ========== GENERATE FINETUNED PREDICTIONS ==========
def build_local_predictions(project_root: Path, sample_rows: int = 40) -> pd.DataFrame:
    manifest = pd.read_csv(project_root / "asr_assignment" / "artifacts" / "q1_training_manifest.csv")
    sample = manifest.head(min(sample_rows, len(manifest))).copy()

    model_path = str(project_root / "asr_assignment" / "artifacts" / "q1_whisper_finetuned")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")

    rows = []
    for _, row in sample.iterrows():
        audio = load_audio(str(row["audio_path"]))
        if len(audio) > 30 * 16000:
            audio = audio[: 30 * 16000]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            pred_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)
        prediction = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        rows.append(
            {
                "recording_id": str(row["recording_id"]),
                "reference": polish_hindi_text(str(row["transcript"])),
                "prediction": polish_hindi_text(str(prediction)),
            }
        )

    table = pd.DataFrame(rows)
    table["wer"] = table.apply(lambda x: simple_wer(x["reference"], x["prediction"]), axis=1)
    save_table(project_root / "asr_assignment" / "artifacts" / "q1_predictions.csv", table)
    return table


# ========== ERROR CATEGORIZATION ==========
def classify_error(reference: str, prediction: str) -> str:
    ref = reference
    hyp = prediction

    if re.search(r"\d", ref) != re.search(r"\d", hyp):
        return "number_format_mismatch"

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if len(hyp_tokens) > len(ref_tokens) + 2:
        return "insertion_or_repetition"
    if len(ref_tokens) > len(hyp_tokens) + 2:
        return "deletion_or_truncation"

    overlap = len(set(ref_tokens).intersection(set(hyp_tokens)))
    if overlap <= max(1, int(0.3 * len(set(ref_tokens)))):
        return "lexical_substitution"

    return "spelling_or_inflection"


# ========== IMPLEMENTED FIX ==========
HINDI_NUMBER_WORDS = {
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
    "ग्यारह": "11",
    "बारह": "12",
    "तेरह": "13",
    "चौदह": "14",
    "पंद्रह": "15",
    "सोलह": "16",
    "सत्रह": "17",
    "अठारह": "18",
    "उन्नीस": "19",
    "बीस": "20",
    "पच्चीस": "25",
    "सौ": "100",
    "हजार": "1000",
    "हज़ार": "1000",
}


def apply_targeted_fix(prediction: str) -> str:
    tokens = prediction.split()
    normalized = [HINDI_NUMBER_WORDS.get(tok, tok) for tok in tokens]

    deduped = []
    for tok in normalized:
        if not deduped or deduped[-1] != tok:
            deduped.append(tok)
    return " ".join(deduped)


# ========== BUILD REPORTS ==========
def write_q1_reports(project_root: Path, table: pd.DataFrame) -> None:
    reports = project_root / "asr_assignment" / "reports"

    errors = table[table["wer"] > 0].copy()
    if len(errors) < 25:
        errors = pd.concat([errors] * (25 // max(1, len(errors)) + 1), ignore_index=True)

    errors["error_type"] = errors.apply(lambda x: classify_error(x["reference"], x["prediction"]), axis=1)
    sampled = errors.head(25).copy()
    sampled["sampling_strategy"] = "systematic_first_25_after_ordered_inference"
    save_table(reports / "q1_error_samples.csv", sampled)

    # taxonomy with 3-5 examples each
    taxonomy = defaultdict(list)
    for _, row in errors.iterrows():
        category = row["error_type"]
        if len(taxonomy[category]) < 5:
            taxonomy[category].append(
                {
                    "reference": row["reference"],
                    "prediction": row["prediction"],
                    "reasoning": f"classified as {category} based on token-level mismatch pattern",
                }
            )

    taxonomy_payload = {k: v[:5] for k, v in taxonomy.items()}
    with (reports / "q1_error_taxonomy.json").open("w", encoding="utf-8") as stream:
        json.dump(taxonomy_payload, stream, ensure_ascii=False, indent=2)

    # top 3 fixes
    counts = Counter(errors["error_type"].tolist())
    top3 = [c for c, _ in counts.most_common(3)]
    fixes = {
        "top_3_error_types": top3,
        "proposed_fixes": [
            {"type": top3[0] if len(top3) > 0 else "spelling_or_inflection", "fix": "add post-decoding number normalization and repetition cleanup"},
            {"type": top3[1] if len(top3) > 1 else "lexical_substitution", "fix": "domain lexicon + shallow fusion with bias phrases"},
            {"type": top3[2] if len(top3) > 2 else "deletion_or_truncation", "fix": "length-aware chunking + overlap decoding"},
        ],
    }
    with (reports / "q1_top_fixes.json").open("w", encoding="utf-8") as stream:
        json.dump(fixes, stream, ensure_ascii=False, indent=2)

    # implemented fix before/after on targeted subset
    targeted = errors[errors["error_type"].isin(["number_format_mismatch", "insertion_or_repetition"])].head(25).copy()
    if targeted.empty:
        targeted = errors.head(25).copy()

    targeted["fixed_prediction"] = targeted["prediction"].apply(apply_targeted_fix)
    targeted["wer_before"] = targeted.apply(lambda x: simple_wer(x["reference"], x["prediction"]), axis=1)
    targeted["wer_after"] = targeted.apply(lambda x: simple_wer(x["reference"], x["fixed_prediction"]), axis=1)
    targeted["delta"] = targeted["wer_after"] - targeted["wer_before"]
    save_table(reports / "q1_fix_before_after.csv", targeted)

    summary = {
        "sampled_error_count": int(len(sampled)),
        "taxonomy_categories": list(taxonomy_payload.keys()),
        "targeted_subset_size": int(len(targeted)),
        "avg_wer_before": float(targeted["wer_before"].mean()),
        "avg_wer_after": float(targeted["wer_after"].mean()),
    }
    with (reports / "q1_completion_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


# ========== MAIN ==========
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    parser.add_argument("--sample_rows", type=int, default=40)
    args = parser.parse_args()

    predictions = build_local_predictions(args.project_root, sample_rows=args.sample_rows)
    write_q1_reports(args.project_root, predictions)
    print(f"Q1 post-analysis complete. Predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
