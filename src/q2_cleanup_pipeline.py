from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from shared_helpers import FriendlyProjectPaths, polish_hindi_text, save_table, simple_wer


# ========== HINDI NUMBER LEXICON ==========
BASIC_NUMBERS = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "पच्चीस": 25,
    "तीस": 30,
    "चालीस": 40,
    "पचास": 50,
    "साठ": 60,
    "सत्तर": 70,
    "अस्सी": 80,
    "नब्बे": 90,
    "सौ": 100,
    "हजार": 1000,
    "हज़ार": 1000,
}

ENGLISH_HINTS = {
    "इंटरव्यू",
    "जॉब",
    "कंप्यूटर",
    "प्रॉब्लम",
    "फाइल",
    "सिस्टम",
    "नेटवर्क",
    "कोड",
    "डाटा",
    "डेटा",
    "सॉफ्टवेयर",
    "प्रोग्राम",
    "मोबाइल",
}


# ========== AUDIO LOADER ==========
def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio.astype(np.float32)
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(duration * 16000))
    src_idx = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    dst_idx = np.linspace(0, len(audio) - 1, num=target_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


def load_audio(file_path: str) -> np.ndarray:
    try:
        import torchaudio

        tensor, sr = torchaudio.load(file_path)
        if tensor.ndim == 2 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        wave = tensor.squeeze(0).numpy().astype(np.float32)
        return resample_to_16k(wave, int(sr))
    except Exception:
        import wave

        with wave.open(file_path, "rb") as wav:
            channels = wav.getnchannels()
            sr = wav.getframerate()
            width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())

        if width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return resample_to_16k(data, int(sr))


# ========== RAW ASR GENERATION ==========
def create_raw_asr_table(project_root: Path, sample_size: int = 60) -> pd.DataFrame:
    paths = FriendlyProjectPaths(project_root)
    manifest = pd.read_csv(paths.artifacts / "q1_training_manifest.csv")
    picked = manifest.sample(n=min(sample_size, len(manifest)), random_state=42).copy()

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")

    raw_predictions: List[str] = []
    for _, row in picked.iterrows():
        audio = load_audio(str(row["audio_path"]))
        if len(audio) > 30 * 16000:
            audio = audio[: 30 * 16000]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            pred_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=220)
        guess = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        raw_predictions.append(polish_hindi_text(str(guess)))

    picked["raw_asr_text"] = raw_predictions
    save_table(paths.artifacts / "q2_raw_asr_outputs.csv", picked)
    return picked


# ========== NUMBER NORMALIZATION ==========
def turn_hindi_number_words_into_digits(sentence: str) -> Tuple[str, List[Dict]]:
    tokens = sentence.split()
    converted: List[str] = []
    traces: List[Dict] = []

    # ========== EDGE CASE PROTECTION ==========
    # Keep conversational idioms intact.
    protected_pattern = re.compile(r"दो[-\s]चार|चार[-\s]पांच|चार[-\s]पाँच|एक\s+बार")
    if protected_pattern.search(sentence):
        return sentence, [{"edge_case": "idiom_or_fixed_phrase_kept", "text": sentence}]

    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]

        if token in {"सौ", "हजार", "हज़ार"} and converted and converted[-1].isdigit():
            base_value = int(converted.pop())
            merged = str(base_value * BASIC_NUMBERS[token])
            converted.append(merged)
            traces.append({"from": f"{base_value} {token}", "to": merged, "reason": "multiplier_merge"})
            cursor += 1
            continue

        if token in BASIC_NUMBERS:
            mapped = str(BASIC_NUMBERS[token])
            converted.append(mapped)
            traces.append({"from": token, "to": mapped, "reason": "direct_map"})
        else:
            converted.append(token)

        cursor += 1

    return " ".join(converted), traces


# ========== ENGLISH WORD TAGGING ==========
def tag_spoken_english_tokens(sentence: str) -> str:
    pieces = sentence.split()
    tagged: List[str] = []
    for piece in pieces:
        if piece in ENGLISH_HINTS:
            tagged.append(f"[EN]{piece}[/EN]")
        else:
            tagged.append(piece)
    return " ".join(tagged)


# ========== RUN CLEANUP PIPELINE ==========
def run_cleanup_pipeline(project_root: Path) -> pd.DataFrame:
    paths = FriendlyProjectPaths(project_root)
    input_table = pd.read_csv(paths.artifacts / "q2_raw_asr_outputs.csv")

    normalized_texts = []
    normalization_debug = []
    english_tagged = []
    for text in input_table["raw_asr_text"].fillna("").tolist():
        converted_text, traces = turn_hindi_number_words_into_digits(polish_hindi_text(text))
        normalized_texts.append(converted_text)
        normalization_debug.append(traces)
        english_tagged.append(tag_spoken_english_tokens(converted_text))

    input_table["number_normalized_text"] = normalized_texts
    input_table["english_tagged_text"] = english_tagged
    input_table["normalization_trace"] = [json.dumps(t, ensure_ascii=False) for t in normalization_debug]

    input_table["wer_raw"] = input_table.apply(lambda x: simple_wer(str(x["transcript"]), str(x["raw_asr_text"])), axis=1)
    input_table["wer_clean"] = input_table.apply(
        lambda x: simple_wer(str(x["transcript"]), str(x["number_normalized_text"])), axis=1
    )

    save_table(paths.reports / "q2_cleanup_results.csv", input_table)
    return input_table


# ========== BUILD REQUIRED EXAMPLES ==========
def write_question_two_examples(project_root: Path, cleanup_table: pd.DataFrame) -> None:
    paths = FriendlyProjectPaths(project_root)

    good = cleanup_table[cleanup_table["raw_asr_text"] != cleanup_table["number_normalized_text"]].head(5)

    edge = cleanup_table[cleanup_table["normalization_trace"].str.contains("idiom_or_fixed_phrase_kept", na=False)].head(3)
    if len(edge) < 3:
        fallback = cleanup_table[cleanup_table["number_normalized_text"].str.contains("एक बार|दो चार|चार पांच|चार पाँच", na=False)]
        edge = pd.concat([edge, fallback]).head(3)

    payload = {
        "number_normalization_examples": good[["raw_asr_text", "number_normalized_text", "normalization_trace"]].to_dict(
            orient="records"
        ),
        "edge_case_examples": edge[["raw_asr_text", "number_normalized_text", "normalization_trace"]].to_dict(
            orient="records"
        ),
        "average_wer_raw": float(cleanup_table["wer_raw"].mean()),
        "average_wer_clean": float(cleanup_table["wer_clean"].mean()),
    }

    output_file = paths.reports / "q2_examples_and_metrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 2 cleanup pipeline")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    parser.add_argument("--sample_size", type=int, default=60)
    args = parser.parse_args()

    create_raw_asr_table(args.project_root, sample_size=args.sample_size)
    cleanup = run_cleanup_pipeline(args.project_root)
    write_question_two_examples(args.project_root, cleanup)
    print(f"Q2 processed rows: {len(cleanup)}")


if __name__ == "__main__":
    main()
