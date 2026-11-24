from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pamap2_activity_recognition.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a PAMAP2 activity model.")
    parser.add_argument("--data-root", required=True, help="Path to the extracted PAMAP2 dataset.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics and model files.")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--test-subject", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        test_subject=args.test_subject,
    )
    print(summary)


if __name__ == "__main__":
    main()
