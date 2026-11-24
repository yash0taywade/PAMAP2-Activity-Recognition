from __future__ import annotations

from pathlib import Path
import json

from .data import load_any_dataset, describe_missingness, profile_dataset
from .features import build_windowed_dataset
from .model import train_classifier, evaluate_classifier, save_model, save_metrics


def split_by_subject(windowed, test_subject: int = 9):
    train_mask = windowed.subjects != test_subject
    test_mask = windowed.subjects == test_subject
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Subject split failed. Train windows: {train_mask.sum()}, test windows: {test_mask.sum()}."
        )
    return (
        windowed.X[train_mask],
        windowed.y[train_mask],
        windowed.X[test_mask],
        windowed.y[test_mask],
    )


def run_pipeline(
    data_root: str | Path,
    output_dir: str | Path = "outputs",
    window_seconds: float | None = None,
    step_seconds: float | None = None,
    test_subject: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_any_dataset(data_root)
    sequences = loaded.sequences
    window_seconds = loaded.window_seconds if window_seconds is None else window_seconds
    step_seconds = loaded.step_seconds if step_seconds is None else step_seconds
    test_subject = loaded.test_subject if test_subject is None else test_subject

    raw_stats = describe_missingness(sequences)
    dataset_profile = profile_dataset(sequences)
    windowed = build_windowed_dataset(
        sequences,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        sampling_rate_hz=loaded.sampling_rate_hz,
    )

    X_train, y_train, X_test, y_test = split_by_subject(windowed, test_subject=test_subject)
    model = train_classifier(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    save_model(model, output_dir / "pamap2_extratrees.joblib")
    save_metrics(metrics, output_dir / "metrics.json")
    (output_dir / "data_profile.json").write_text(json.dumps(dataset_profile, indent=2, sort_keys=True))

    summary = {
        "dataset": loaded.name,
        "num_subjects": len({seq.subject_id for seq in sequences}),
        "num_windows_train": int(X_train.shape[0]),
        "num_windows_test": int(X_test.shape[0]),
        "raw_stats": raw_stats,
        "profile": dataset_profile,
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "test_subject": test_subject,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
