# System Architecture

```mermaid
flowchart TD
    A[Local Dataset Files\nXLSX + downloads/] --> B[q1_whisper_pipeline.py\nManifest + preprocessing]
    B --> C[whisper_finetune_runner.py\nWhisper-small fine-tuning]
    C --> D[whisper_fleurs_evaluator.py\nBaseline vs Fine-tuned WER]
    D --> E[q1_finalize_reports.py\nError samples + taxonomy + fixes]

    A --> F[q2_cleanup_pipeline.py\nRaw ASR + number normalization + EN tagging]
    A --> G[q3_spelling_audit.py\nWord correctness + confidence review]
    A --> H[q4_lattice_wer.py\nLattice WER vs classic WER]

    E --> R[reports/]
    F --> R
    G --> R
    H --> R

    B --> T[artifacts/]
    C --> T
    D --> T

    R --> S[SUBMISSION_CHECKLIST.md\nFinal handoff mapping]
```

## Notes
- `reports/` contains submission-ready outputs.
- `artifacts/` stores training manifests, predictions, and checkpoints.
- Q1 is the heaviest path (fine-tune + FLEURS evaluation).
