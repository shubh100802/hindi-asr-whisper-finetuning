from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ========== FRIENDLY RUNNER ==========
def run_step(step_name: str, command: list[str], working_folder: Path) -> None:
    print(f"\n========== RUNNING {step_name} ==========")
    subprocess.run(command, cwd=str(working_folder), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run assignment pipelines")
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve())
    parser.add_argument("--skip_q1_training", action="store_true")
    args = parser.parse_args()

    src_folder = args.project_root / "asr_assignment" / "src"

    run_step("Q1_PREP", [sys.executable, "q1_whisper_pipeline.py", "--project_root", str(args.project_root)], src_folder)
    run_step("Q2", [sys.executable, "q2_cleanup_pipeline.py", "--project_root", str(args.project_root)], src_folder)
    run_step("Q3", [sys.executable, "q3_spelling_audit.py", "--project_root", str(args.project_root)], src_folder)
    run_step("Q4", [sys.executable, "q4_lattice_wer.py", "--project_root", str(args.project_root)], src_folder)
    run_step("Q5", [sys.executable, "q5_data_readiness.py", "--project_root", str(args.project_root)], src_folder)

    if args.skip_q1_training:
        print("\n========== Q1 TRAIN/EVAL SKIPPED ==========")
        print("Use whisper_finetune_runner.py and whisper_fleurs_evaluator.py when you are ready.")


if __name__ == "__main__":
    main()
