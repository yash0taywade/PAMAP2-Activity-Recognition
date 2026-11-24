from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SubjectSequence:
    subject_id: int
    timestamps: np.ndarray
    labels: np.ndarray
    signals: np.ndarray
    raw_missing_values: int
    raw_total_values: int
    raw_label_zero_rows: int


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    sequences: list[SubjectSequence]
    sampling_rate_hz: int
    window_seconds: float
    step_seconds: float
    test_subject: int | None


def find_subject_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if root.is_file():
        root = root.parent
    if (root / "PAMAP2_Dataset" / "Protocol").exists():
        root = root / "PAMAP2_Dataset" / "Protocol"
    elif (root / "Protocol").exists():
        root = root / "Protocol"
    files = sorted(root.rglob("subject*.dat"))
    if not files:
        raise FileNotFoundError(
            f"No PAMAP2 subject*.dat files found under {root}. "
            "Point --data-root at the extracted dataset directory."
        )
    return files


def _normalize_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if root.is_file():
        return root.parent
    return root


def subject_id_from_path(path: Path) -> int:
    match = re.search(r"subject(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not infer subject id from {path}")
    return int(match.group(1))


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    filled = values.astype(np.float32, copy=True)
    x = np.arange(filled.shape[0])
    for col in range(filled.shape[1]):
        column = filled[:, col]
        mask = np.isfinite(column)
        if mask.sum() == 0:
            filled[:, col] = 0.0
            continue
        if mask.sum() == 1:
            filled[:, col] = column[mask][0]
            continue
        filled[:, col] = np.interp(x, x[mask], column[mask]).astype(np.float32)
    return filled


def load_subject_sequence(path: str | Path, keep_label_zero: bool = False) -> SubjectSequence:
    file_path = Path(path)
    data = np.loadtxt(file_path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Unexpected column count in {file_path}: {data.shape[1]}")

    timestamps = data[:, 0]
    labels = data[:, 1].astype(np.int32)
    raw_signals = data[:, 2:]
    raw_missing_values = int(np.isnan(raw_signals).sum())
    raw_total_values = int(raw_signals.size)
    raw_label_zero_rows = int((labels == 0).sum())
    signals = raw_signals

    valid = np.isfinite(labels)
    if not keep_label_zero:
        valid &= labels > 0

    timestamps = timestamps[valid]
    labels = labels[valid]
    signals = signals[valid]
    signals = interpolate_missing(signals)

    return SubjectSequence(
        subject_id=subject_id_from_path(file_path),
        timestamps=timestamps,
        labels=labels,
        signals=signals,
        raw_missing_values=raw_missing_values,
        raw_total_values=raw_total_values,
        raw_label_zero_rows=raw_label_zero_rows,
    )


def load_dataset(data_root: str | Path, keep_label_zero: bool = False) -> list[SubjectSequence]:
    loaded = load_any_dataset(data_root, keep_label_zero=keep_label_zero)
    return loaded.sequences


def detect_dataset_type(data_root: str | Path) -> str:
    root = _normalize_root(data_root)
    if (root / "PAMAP2_Dataset" / "Protocol").exists() or (root / "Protocol").exists():
        return "pamap2"
    if list(root.rglob("subject*.dat")):
        return "pamap2"
    if list(root.rglob("Inertial Signals")) or list(root.rglob("body_acc_x_train.txt")):
        return "uci_har"
    raise FileNotFoundError(
        f"Could not find a supported dataset under {root}. "
        "Expected PAMAP2 subject*.dat files or UCI HAR inertial-signal files."
    )


def _load_uci_har_split(split_dir: Path) -> list[SubjectSequence]:
    inertial_dir = split_dir / "Inertial Signals"
    if not inertial_dir.exists():
        raise FileNotFoundError(f"Missing inertial signals directory: {inertial_dir}")

    channel_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    def load_txt(name: str) -> np.ndarray:
        return np.loadtxt(inertial_dir / f"{name}.txt", dtype=np.float32)

    channel_data = [load_txt(name) for name in channel_names]
    labels = np.loadtxt(split_dir / "y.txt", dtype=np.int32)
    subjects = np.loadtxt(split_dir / "subject.txt", dtype=np.int32)

    n_samples = labels.shape[0]
    sequences: list[SubjectSequence] = []
    for idx in range(n_samples):
        signals = np.column_stack([channel[idx] for channel in channel_data]).astype(np.float32)
        seq_labels = np.full(signals.shape[0], labels[idx], dtype=np.int32)
        sequences.append(
            SubjectSequence(
                subject_id=int(subjects[idx]),
                timestamps=np.arange(signals.shape[0], dtype=np.float32),
                labels=seq_labels,
                signals=signals,
                raw_missing_values=0,
                raw_total_values=int(signals.size),
                raw_label_zero_rows=int((seq_labels == 0).sum()),
            )
        )
    return sequences


def load_any_dataset(data_root: str | Path, keep_label_zero: bool = False) -> LoadedDataset:
    root = _normalize_root(data_root)
    dataset_type = detect_dataset_type(root)

    if dataset_type == "pamap2":
        sequences = [
            load_subject_sequence(path, keep_label_zero=keep_label_zero)
            for path in find_subject_files(root)
        ]
        return LoadedDataset(
            name="pamap2",
            sequences=sequences,
            sampling_rate_hz=100,
            window_seconds=5.0,
            step_seconds=2.5,
            test_subject=108,
        )

    train_sequences = _load_uci_har_split(root / "train")
    test_sequences = _load_uci_har_split(root / "test")
    sequences = train_sequences + test_sequences
    subjects = sorted({seq.subject_id for seq in sequences})
    return LoadedDataset(
        name="uci_har",
        sequences=sequences,
        sampling_rate_hz=1,
        window_seconds=128.0,
        step_seconds=128.0,
        test_subject=subjects[-1] if subjects else None,
    )


def describe_missingness(sequences: Iterable[SubjectSequence]) -> dict[str, float]:
    total_values = 0
    missing_values = 0
    label_zero_rows = 0
    for seq in sequences:
        total_values += seq.raw_total_values
        missing_values += seq.raw_missing_values
        label_zero_rows += seq.raw_label_zero_rows
    return {
        "missing_fraction": (missing_values / total_values) if total_values else 0.0,
        "missing_values": float(missing_values),
        "total_values": float(total_values),
        "label_zero_rows": float(label_zero_rows),
    }


def profile_dataset(sequences: Iterable[SubjectSequence]) -> dict:
    subject_rows: dict[str, int] = {}
    subject_missing: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    total_rows = 0
    total_signal_values = 0
    total_missing_values = 0

    for seq in sequences:
        subject_key = str(seq.subject_id)
        subject_rows[subject_key] = subject_rows.get(subject_key, 0) + int(seq.labels.shape[0])
        subject_missing[subject_key] = subject_missing.get(subject_key, 0) + int(seq.raw_missing_values)
        total_rows += int(seq.labels.shape[0])
        total_signal_values += int(seq.raw_total_values)
        total_missing_values += int(seq.raw_missing_values)

        unique_labels, counts = np.unique(seq.labels, return_counts=True)
        for label, count in zip(unique_labels, counts, strict=False):
            key = str(int(label))
            label_counts[key] = label_counts.get(key, 0) + int(count)

    return {
        "total_rows": total_rows,
        "total_signal_values": total_signal_values,
        "total_missing_values": total_missing_values,
        "missing_fraction": (total_missing_values / total_signal_values) if total_signal_values else 0.0,
        "subject_rows": dict(sorted(subject_rows.items(), key=lambda item: int(item[0]))),
        "subject_missing_values": dict(sorted(subject_missing.items(), key=lambda item: int(item[0]))),
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: int(item[0]))),
    }
