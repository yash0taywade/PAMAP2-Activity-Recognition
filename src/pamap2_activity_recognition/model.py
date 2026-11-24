from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np


@dataclass(frozen=True)
class SplitMetrics:
    accuracy: float
    balanced_accuracy: float
    report: dict


@dataclass
class NearestCentroidModel:
    classes_: np.ndarray
    centroids_: np.ndarray
    feature_mean_: np.ndarray
    feature_scale_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.feature_mean_) / self.feature_scale_
        distances = np.linalg.norm(Xn[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> NearestCentroidModel:
    feature_mean = X_train.mean(axis=0)
    feature_scale = X_train.std(axis=0)
    feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
    Xn = (X_train - feature_mean) / feature_scale
    classes = np.unique(y_train)
    centroids = np.vstack([Xn[y_train == cls].mean(axis=0) for cls in classes])
    return NearestCentroidModel(
        classes_=classes,
        centroids_=centroids,
        feature_mean_=feature_mean,
        feature_scale_=feature_scale,
    )


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report: dict[str, dict[str, float] | float] = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for cls in classes:
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        support = float(np.sum(y_true == cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        report[str(int(cls))] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    accuracy = float(np.mean(y_true == y_pred))
    macro_avg = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": float(np.sum(supports)),
    }
    weighted_avg = {
        "precision": float(np.average(precisions, weights=supports)) if supports and np.sum(supports) else 0.0,
        "recall": float(np.average(recalls, weights=supports)) if supports and np.sum(supports) else 0.0,
        "f1-score": float(np.average(f1s, weights=supports)) if supports and np.sum(supports) else 0.0,
        "support": float(np.sum(supports)),
    }
    report["accuracy"] = accuracy
    report["macro avg"] = macro_avg
    report["weighted avg"] = weighted_avg
    return report


def evaluate_classifier(model: NearestCentroidModel, X_test: np.ndarray, y_test: np.ndarray) -> SplitMetrics:
    predictions = model.predict(X_test)
    classes = np.unique(np.concatenate([y_test, predictions]))
    recalls = []
    for cls in classes:
        tp = np.sum((y_test == cls) & (predictions == cls))
        fn = np.sum((y_test == cls) & (predictions != cls))
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
    return SplitMetrics(
        accuracy=float(np.mean(y_test == predictions)),
        balanced_accuracy=float(np.mean(recalls)) if recalls else 0.0,
        report=_classification_report(y_test, predictions),
    )


def save_model(model, path: str | Path) -> None:
    with Path(path).open("wb") as f:
        pickle.dump(model, f)


def save_metrics(metrics: SplitMetrics, path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "report": metrics.report,
            },
            indent=2,
            sort_keys=True,
        )
    )
