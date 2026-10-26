
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco churn data and encode the target."""

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Remove surrounding whitespace from string columns.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges to numeric; blank strings become NaN.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Remove rows with missing target values.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Encode target: No = 0, Yes = 1.
    target_mapping = {"No": 0, "Yes": 1}
    invalid_targets = set(df[TARGET_COLUMN].unique()) - set(target_mapping)

    if invalid_targets:
        raise ValueError(f"Unexpected target labels: {invalid_targets}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_mapping).astype(int)

    # Add tenure groups if tenure is available.
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, 60, float("inf")],
            labels=[
                "0-6_months",
                "7-12_months",
                "13-24_months",
                "25-48_months",
                "49-60_months",
                "61+_months",
            ],
        ).astype("object")

    return df


def split_features_target(df: pd.DataFrame):
    """Separate features and target, excluding the customer identifier."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_columns.append(ID_COLUMN)

    X = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
