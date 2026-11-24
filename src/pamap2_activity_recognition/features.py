from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import SubjectSequence


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    feature_names: list[str]


def contiguous_label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if labels.size == 0:
        return runs

    start = 0
    current = int(labels[0])
    for idx in range(1, labels.size):
        label = int(labels[idx])
        if label != current:
            runs.append((start, idx, current))
            start = idx
            current = label
    runs.append((start, labels.size, current))
    return runs


def compute_window_features(window: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stats = [
        ("mean", np.mean(window, axis=0)),
        ("std", np.std(window, axis=0)),
        ("min", np.min(window, axis=0)),
        ("max", np.max(window, axis=0)),
        ("median", np.median(window, axis=0)),
        ("q25", np.quantile(window, 0.25, axis=0)),
        ("q75", np.quantile(window, 0.75, axis=0)),
        ("energy", np.mean(np.square(window), axis=0)),
    ]

    features = [values.astype(np.float32) for _, values in stats]
    names: list[str] = []
    for stat_name, _ in stats:
        for col in range(window.shape[1]):
            names.append(f"{stat_name}_c{col:02d}")
    return np.concatenate(features, axis=0), names


def window_subject(
    sequence: SubjectSequence,
    window_size: int,
    step_size: int,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subject_parts: list[int] = []

    for start, end, label in contiguous_label_runs(sequence.labels):
        run = sequence.signals[start:end]
        if run.shape[0] < window_size:
            continue

        last_start = end - start - window_size
        for offset in range(0, last_start + 1, step_size):
            window = run[offset : offset + window_size]
            feats, _ = compute_window_features(window)
            X_parts.append(feats)
            y_parts.append(label)
            subject_parts.append(sequence.subject_id)

    return X_parts, y_parts, subject_parts


def build_windowed_dataset(
    sequences: Iterable[SubjectSequence],
    window_seconds: float = 5.0,
    step_seconds: float = 2.5,
    sampling_rate_hz: int = 100,
) -> WindowedDataset:
    window_size = int(window_seconds * sampling_rate_hz)
    step_size = max(1, int(step_seconds * sampling_rate_hz))

    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subject_parts: list[int] = []
    feature_names: list[str] | None = None

    for sequence in sequences:
        seq_X, seq_y, seq_subjects = window_subject(sequence, window_size, step_size)
        if seq_X and feature_names is None:
            _, feature_names = compute_window_features(sequence.signals[:window_size])
        X_parts.extend(seq_X)
        y_parts.extend(seq_y)
        subject_parts.extend(seq_subjects)

    if not X_parts:
        raise ValueError("No windows were produced. Check the window size and input data.")

    return WindowedDataset(
        X=np.vstack(X_parts).astype(np.float32),
        y=np.asarray(y_parts, dtype=np.int32),
        subjects=np.asarray(subject_parts, dtype=np.int32),
        feature_names=feature_names or [],
    )

