"""Train and evaluate models on the real UCI Parkinson's voice dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "parkinsons.data"
DEFAULT_MODEL_PATH = PROJECT_DIR / "submission" / "parkinsons_models.joblib"
TARGET_COLUMN = "status"
IDENTIFIER_COLUMNS = ("name",)


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the supplied UCI CSV, raising a helpful error when it is missing."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download the official UCI ZIP and "
            "extract parkinsons.data into the project's data folder; see "
            "data/README.md."
        )
    frame = pd.read_csv(data_path)
    required_columns = {TARGET_COLUMN, *IDENTIFIER_COLUMNS}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    return frame


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove identifiers and separate numeric voice measurements from status."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target column 'status' contains missing values.")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
    observed_classes = set(target.unique())
    if not observed_classes.issubset({0, 1}) or len(observed_classes) != 2:
        raise ValueError(
            "Expected both binary target classes 0 and 1 in the real dataset."
        )

    features = frame.drop(
        columns=[TARGET_COLUMN, *IDENTIFIER_COLUMNS], errors="ignore"
    ).apply(pd.to_numeric, errors="raise")
    if features.empty or features.shape[1] == 0:
        raise ValueError("No numeric voice-measurement feature columns were found.")
    return features, target


def build_models() -> dict[str, Pipeline]:
    """Create reproducible pipelines that learn preprocessing from training rows."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, Any]:
    """Return holdout metrics; ROC-AUC is omitted if the holdout has one class."""
    predictions = model.predict(features)
    result: dict[str, Any] = {
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "accuracy": accuracy_score(target, predictions),
        "classification_report": classification_report(
            target, predictions, labels=[0, 1], zero_division=0
        ),
    }
    if target.nunique() == 2:
        result["roc_auc"] = roc_auc_score(target, model.predict_proba(features)[:, 1])
    return result


def train_and_evaluate(
    frame: pd.DataFrame,
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.Series, dict[str, Any]]:
    """Split the real data, fit both models, and evaluate on held-out rows."""
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    models = build_models()
    results: dict[str, Any] = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(model, x_test, y_test)
    return models, x_test, y_test, results


def save_models(
    models: dict[str, Pipeline],
    feature_columns: list[str],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save fitted pipelines and their expected input feature order."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models, "feature_columns": feature_columns},
        model_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Parkinson's voice classifiers on the real UCI dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the downloaded parkinsons.data file.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path for the fitted joblib model bundle.",
    )
    args = parser.parse_args()

    frame = load_dataset(args.data)
    features, _ = split_features_target(frame)
    models, _, _, results = train_and_evaluate(frame)

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(
            "Confusion matrix (rows=true, columns=predicted; labels 0, 1):\n"
            f"{metrics['confusion_matrix']}"
        )
        for metric_name in ("precision", "recall", "f1", "accuracy", "roc_auc"):
            if metric_name in metrics:
                print(f"{metric_name}: {metrics[metric_name]:.3f}")
        print(metrics["classification_report"])

    save_models(models, list(features.columns), args.model_output)
    print(f"Saved fitted pipelines to: {args.model_output}")


if __name__ == "__main__":
    main()