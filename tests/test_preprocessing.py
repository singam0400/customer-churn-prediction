
import pandas as pd
import pytest

from src.preprocessing import (
    clean_data,
    split_features_target,
    build_preprocessor,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4"],
        "tenure": [1, 12, 36, 60],
        "MonthlyCharges": [29.85, 56.95, 70.35, 89.10],
        "TotalCharges": ["29.85", "684.35", "2500.50", "5400.00"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "Churn": ["Yes", "No", "No", "Yes"],
    })


def test_clean_data_encodes_target(sample_data):
    cleaned = clean_data(sample_data)

    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert cleaned["Churn"].dtype.kind in "iu"


def test_clean_data_converts_total_charges(sample_data):
    cleaned = clean_data(sample_data)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_clean_data_creates_tenure_group(sample_data):
    cleaned = clean_data(sample_data)

    assert "tenure_group" in cleaned.columns
    assert cleaned["tenure_group"].notna().all()


def test_split_features_target_excludes_target_and_id(sample_data):
    cleaned = clean_data(sample_data)
    X, y = split_features_target(cleaned)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert len(X) == len(y)


def test_preprocessor_transforms_features(sample_data):
    cleaned = clean_data(sample_data)
    X, _ = split_features_target(cleaned)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_clean_data_rejects_invalid_target(sample_data):
    sample_data.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Unexpected target labels"):
        clean_data(sample_data)
