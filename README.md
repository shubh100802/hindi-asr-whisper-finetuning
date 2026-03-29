# hindi-asr-whisper-finetuning

Production-style, reproducible Hindi ASR assignment implementation covering:
- Q1: Whisper-small fine-tuning + FLEURS evaluation + error taxonomy + fix analysis
- Q2: Raw ASR cleanup pipeline (number normalization + English word tagging)
- Q3: Spelling quality classification with confidence + low-confidence audit
- Q4: Lattice-based evaluation with fair WER handling

## What this repo gives you
- Clean `src/` scripts for each question
- End-to-end execution flow
- Structured outputs in `reports/`
- Reproducible command list for fresh machines

Architecture: [docs/architecture.md](./docs/architecture.md)

## Table of Contents
- [1. Requirements](#1-requirements)
- [2. Setup](#2-setup)
- [3. Dataset Layout](#3-dataset-layout)
- [4. Run Paths](#4-run-paths)
- [5. Question-wise Commands](#5-question-wise-commands)
- [6. Expected Outputs](#6-expected-outputs)
- [7. Troubleshooting](#7-troubleshooting)
- [8. What to Commit](#8-what-to-commit)

## 1. Requirements
- Python 3.10+
- Git
- Optional but recommended: NVIDIA GPU (CPU works but is much slower)

## 2. Setup

```bash
git clone https://github.com/shubh100802/hindi-asr-whisper-finetuning.git
cd hindi-asr-whisper-finetuning
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Dataset Layout
Put these XLSX files in repo root:
- `dataset_1bujiO2N.xlsx`
- `dataset_17DwCAx6.xlsx`
- `dataset_1J_I0rao.xlsx`
- `dataset_1JItJnil.xlsx`

Create this local data structure:

```text
downloads/
  audio/
  transcriptions/
  metadata/
```

## 4. Run Paths

### A) Fast validation run (non-heavy)

```bash
python src/run_all_questions.py --project_root . --skip_q1_training
```

### B) Full assignment run
Run Q1-heavy path + Q2/Q3/Q4 scripts manually (recommended for control):
1. Q1 preprocessing
2. Q1 fine-tuning
3. Q1 FLEURS eval
4. Q1 final report shaping
5. Q2, Q3, Q4 scripts

## 5. Question-wise Commands

### Q1: Preprocess + Fine-tune + Evaluate + Finalize

```bash
python src/q1_whisper_pipeline.py --project_root .
```

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

```bash
python src/whisper_fleurs_evaluator.py \
  --baseline_model openai/whisper-small \
  --finetuned_model ./artifacts/q1_whisper_finetuned \
  --fleurs_config hi_in \
  --max_eval_samples -1 \
  --output_json ./reports/q1_wer_report.json \
  --output_csv ./reports/q1_wer_report.csv
```

```bash
python src/q1_finalize_reports.py --project_root .
```

Notes:
- Use `--max_eval_samples -1` for full Hindi FLEURS test split.
- On CPU this can take a long time.

### Q2: Cleanup Pipeline

```bash
python src/q2_cleanup_pipeline.py --project_root . --sample_size 104
```

### Q3: Spelling Audit

```bash
python src/q3_spelling_audit.py --project_root .
```

### Q4: Lattice WER

```bash
python src/q4_lattice_wer.py --project_root .
```

## 6. Expected Outputs

Primary files in `reports/`:
- `q1_wer_report.csv`, `q1_wer_report.json`
- `q1_error_samples.csv`, `q1_error_taxonomy.json`
- `q1_top_fixes.json`, `q1_fix_before_after.csv`, `q1_completion_summary.json`
- `q2_cleanup_results.csv`, `q2_examples_and_metrics.json`
- `q3_word_spelling_classification.csv`, `q3_word_spelling_classification.xlsx`
- `q3_low_confidence_review.csv`, `q3_summary.json`
- `q4_lattice_wer_report.csv`, `q4_methodology.json`
- `SUBMISSION_CHECKLIST.md`

Primary files in `artifacts/`:
- `q1_training_manifest.csv`
- `q1_predictions.csv`
- `q1_whisper_finetuned/` (trained checkpoint)

## 7. Troubleshooting
- Slow evaluation on CPU:
  - Start with `--max_eval_samples 20` to smoke-test.
- Stuck Python processes:
  - PowerShell: `Get-Process | ? ProcessName -like 'python*' | Stop-Process -Force`
- Devanagari display issue in terminal:
  - Use UTF-8 terminal/profile.
- Hugging Face download limits:
  - Set `HF_TOKEN` for better reliability.

## 8. What to Commit
Commit:
- `src/`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `docs/architecture.md`

Do not commit:
- `downloads/`
- large model checkpoints
- heavy generated reports unless explicitly required
