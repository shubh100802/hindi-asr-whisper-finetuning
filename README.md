# hindi-asr-whisper-finetuning

End-to-end implementation for the Hindi ASR assignment (Q1-Q4):
- Q1: Whisper-small fine-tuning + baseline vs fine-tuned WER + error analysis
- Q2: Raw ASR cleanup pipeline (number normalization + English word tagging)
- Q3: Spelling quality classification with confidence and low-confidence review
- Q4: Lattice-based WER evaluation and method explanation

## 1) Prerequisites

- Python 3.10+
- Git
- Recommended for faster runs: NVIDIA GPU (CPU works but is slow)

## 2) Clone and setup

```bash
git clone https://github.com/shubh100802/hindi-asr-whisper-finetuning.git
cd hindi-asr-whisper-finetuning
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Required data files (place at repo root)

Put these files in the repository root:
- `dataset_1bujiO2N.xlsx`
- `dataset_17DwCAx6.xlsx`
- `dataset_1J_I0rao.xlsx`
- `dataset_1JItJnil.xlsx`

Create dataset folders under repo root:
- `downloads/audio`
- `downloads/transcriptions`
- `downloads/metadata`

The code expects downloaded local dataset files in these folders.

## 4) Project structure

```text
src/
  shared_helpers.py
  q1_whisper_pipeline.py
  whisper_finetune_runner.py
  whisper_fleurs_evaluator.py
  q1_finalize_reports.py
  q2_cleanup_pipeline.py
  q3_spelling_audit.py
  q4_lattice_wer.py
  q5_data_readiness.py
  run_all_questions.py
reports/        # generated outputs
artifacts/      # generated intermediate files/models
```

## 5) Quick run (all non-heavy steps)

```bash
python src/run_all_questions.py --project_root . --skip_q1_training
```

This runs:
- Q1 preprocessing only
- Q2 full cleanup pipeline
- Q3 spelling pipeline
- Q4 lattice scoring
- Q5 data readiness report

## 6) Q1 full run (fine-tune + FLEURS eval)

### 6.1 Build Q1 manifest

```bash
python src/q1_whisper_pipeline.py --project_root .
```

### 6.2 Fine-tune Whisper-small

```bash
python -m src.whisper_finetune_runner \
  --manifest ./artifacts/q1_training_manifest.csv \
  --model_id openai/whisper-small \
  --language hindi \
  --task transcribe \
  --output_dir ./artifacts/q1_whisper_finetuned \
  --num_train_epochs 1 \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --warmup_steps 10
```

### 6.3 Evaluate baseline vs fine-tuned on FLEURS (Hindi)

```bash
python src/whisper_fleurs_evaluator.py \
  --baseline_model openai/whisper-small \
  --finetuned_model ./artifacts/q1_whisper_finetuned \
  --fleurs_config hi_in \
  --max_eval_samples -1 \
  --output_json ./reports/q1_wer_report.json \
  --output_csv ./reports/q1_wer_report.csv
```

Notes:
- `--max_eval_samples -1` means full Hindi test split.
- On CPU this can take a long time.

### 6.4 Generate Q1 final analysis artifacts

```bash
python src/q1_finalize_reports.py --project_root .
```

## 7) Q2 run

```bash
python src/q2_cleanup_pipeline.py --project_root . --sample_size 104
```

Outputs:
- `reports/q2_cleanup_results.csv`
- `reports/q2_examples_and_metrics.json`

## 8) Q3 run

```bash
python src/q3_spelling_audit.py --project_root .
```

Outputs:
- `reports/q3_word_spelling_classification.csv`
- `reports/q3_word_spelling_classification.xlsx`
- `reports/q3_low_confidence_review.csv`
- `reports/q3_summary.json`

## 9) Q4 run

```bash
python src/q4_lattice_wer.py --project_root .
```

Outputs:
- `reports/q4_lattice_wer_report.csv`
- `reports/q4_methodology.json`

## 10) All key outputs checklist

A checklist file is generated at:
- `reports/SUBMISSION_CHECKLIST.md`

## 11) Troubleshooting

- If ASR evaluation is too slow on CPU, first test with smaller sample:
  - `--max_eval_samples 20`
- If a run gets stuck, stop Python processes and rerun:
  - PowerShell: `Get-Process | ? ProcessName -like 'python*' | Stop-Process -Force`
- If Devanagari text displays incorrectly in terminal, ensure UTF-8 output is enabled.

## 12) What to commit

Commit only source and lightweight project files:
- `src/`
- `requirements.txt`
- `README.md`
- `.gitignore`

Do **not** commit:
- `downloads/` raw data
- trained model weights
- heavy report csv/xlsx artifacts unless explicitly needed
