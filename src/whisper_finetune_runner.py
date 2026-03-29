from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ========== AUDIO LOADING ==========
def load_audio_as_mono_16k(file_path: str) -> np.ndarray:
    waveform = None
    sample_rate = None

    try:
        import torchaudio

        tensor, sr = torchaudio.load(file_path)
        if tensor.ndim == 2 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        tensor = tensor.squeeze(0)
        if sr != 16000:
            tensor = torchaudio.functional.resample(tensor, sr, 16000)
        waveform = tensor.numpy().astype(np.float32)
    except Exception:
        pass

    if waveform is None:
        try:
            import soundfile as sf

            data, sr = sf.read(file_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = data.astype(np.float32)
            sample_rate = int(sr)
        except Exception:
            pass

    if waveform is None:
        import wave

        with wave.open(file_path, "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
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
        waveform = data.astype(np.float32)

    if sample_rate and sample_rate != 16000:
        duration = len(waveform) / float(sample_rate)
        target_len = max(1, int(duration * 16000))
        src_idx = np.linspace(0, len(waveform) - 1, num=len(waveform), dtype=np.float32)
        dst_idx = np.linspace(0, len(waveform) - 1, num=target_len, dtype=np.float32)
        waveform = np.interp(dst_idx, src_idx, waveform).astype(np.float32)

    return waveform


# ========== DATASET ==========
class FriendlyWhisperDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, processor, max_seconds: int = 30, max_label_tokens: int = 448):
        self.rows = manifest.to_dict(orient="records")
        self.processor = processor
        self.max_audio_samples = max_seconds * 16000
        self.max_label_tokens = max_label_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        audio = load_audio_as_mono_16k(str(row["audio_path"]))
        if len(audio) > self.max_audio_samples:
            audio = audio[: self.max_audio_samples]

        input_features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(
            str(row["transcript"]), truncation=True, max_length=self.max_label_tokens
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ========== COLLATOR ==========
@dataclass
class FriendlySpeechCollator:
    processor: object

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# ========== FRIENDLY TRAINING ENTRY ==========
def run_training_job(arguments: argparse.Namespace) -> None:
    try:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
    except Exception as error:
        raise RuntimeError("Please install transformers, datasets, accelerate, evaluate, jiwer") from error

    manifest = pd.read_csv(arguments.manifest)
    if len(manifest) == 0:
        raise RuntimeError("Manifest is empty, cannot train")

    processor = WhisperProcessor.from_pretrained(arguments.model_id, language=arguments.language, task=arguments.task)
    processor.tokenizer.set_prefix_tokens(language=arguments.language, task=arguments.task)
    model = WhisperForConditionalGeneration.from_pretrained(arguments.model_id)

    
    

    train_dataset = FriendlyWhisperDataset(manifest, processor, max_seconds=30, max_label_tokens=448)
    collator = FriendlySpeechCollator(processor=processor)

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=arguments.output_dir,
        per_device_train_batch_size=arguments.per_device_train_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        warmup_steps=arguments.warmup_steps,
        num_train_epochs=arguments.num_train_epochs,
        max_steps=arguments.max_steps,
        logging_steps=5,
        save_strategy="no",
        fp16=False,
        bf16=False,
        push_to_hub=False,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        args=trainer_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(arguments.output_dir)
    processor.save_pretrained(arguments.output_dir)


# ========== COMMAND LINE ==========
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training_job(args)
    print("Whisper fine-tuning completed.")


if __name__ == "__main__":
    main()
