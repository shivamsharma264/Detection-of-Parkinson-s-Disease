"""Software tests on explicitly synthetic fixtures, not clinical records."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train_model import build_models, split_features_target


@pytest.fixture
def synthetic_voice_fixture() -> pd.DataFrame:
    """Deterministic toy rows used only to exercise preprocessing and prediction."""
    return pd.DataFrame(
        {
            "name": [f"synthetic-row-{index}" for index in range(40)],
            "voice_measure_a": [float(index) for index in range(40)],
            "voice_measure_b": [
                float("nan") if index == 0 else float(index % 7)
                for index in range(40)
            ],
            "status": [index % 2 for index in range(40)],
        }
    )


def test_feature_target_separation_excludes_identifier_and_target(
    synthetic_voice_fixture: pd.DataFrame,
) -> None:
    features, target = split_features_target(synthetic_voice_fixture)

    assert list(features.columns) == ["voice_measure_a", "voice_measure_b"]
    assert "name" not in features.columns
    assert target.name == "status"
    assert set(target.unique()) == {0, 1}


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest"])
def test_pipeline_fits_and_predicts_with_synthetic_fixture(
    synthetic_voice_fixture: pd.DataFrame, model_name: str
) -> None:
    features, target = split_features_target(synthetic_voice_fixture)
    model = build_models()[model_name]

    assert isinstance(model, Pipeline)
    model.fit(features, target)
    predictions = model.predict(features.iloc[:5])

    assert predictions.shape == (5,)
    assert set(predictions).issubset({0, 1})


def test_logistic_scaler_is_fitted_on_training_rows_only(
    synthetic_voice_fixture: pd.DataFrame,
) -> None:
    features, target = split_features_target(synthetic_voice_fixture)
    training_features = features.iloc[:20]
    model = build_models()["Logistic Regression"]
    model.fit(training_features, target.iloc[:20])

    assert model.named_steps["scaler"].n_samples_seen_ == len(training_features)
    assert list(model.feature_names_in_) == list(training_features.columns)


def test_feature_target_split_rejects_missing_target() -> None:
    frame = pd.DataFrame({"name": ["synthetic"], "measurement": [1.0]})

    with pytest.raises(ValueError, match="target column"):
        split_features_target(frame)
