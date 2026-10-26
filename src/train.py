
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    clean_data,
    split_features_target,
    build_preprocessor,
)


def train_and_evaluate(
    df: pd.DataFrame,
    experiment_name: str = "customer-churn-prediction",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Train, evaluate, and log churn classification models to MLflow."""

    df_clean = clean_data(df)
    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    mlflow.set_experiment(experiment_name)

    trained_models = {}
    results = []

    for name, classifier in models.items():
        pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model_type": name,
                "test_size": test_size,
                "random_state": random_state,
                "feature_engineering": "tenure_group",
            })

            if name == "Logistic Regression":
                mlflow.log_param("max_iter", 1000)
            else:
                mlflow.log_params({
                    "n_estimators": 200,
                    "min_samples_leaf": 2,
                })

            mlflow.log_metrics(metrics)

            # Log confusion matrix as a CSV artifact.
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            cm_df = pd.DataFrame(
                cm,
                index=["Actual_No_Churn", "Actual_Churn"],
                columns=["Predicted_No_Churn", "Predicted_Churn"],
            )

            cm_path = f"/tmp/confusion_matrix_{name.replace(' ', '_')}.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path)

            print(f"Logged MLflow run: {name}")
            print(classification_report(
                y_test,
                y_pred,
                target_names=["No Churn", "Churn"],
                zero_division=0,
            ))

        trained_models[name] = pipeline

        results.append({
            "model": name,
            **metrics,
        })

    results_df = pd.DataFrame(results).sort_values(
        by="f1", ascending=False
    ).reset_index(drop=True)

    return {
        "models": trained_models,
        "results": results_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
