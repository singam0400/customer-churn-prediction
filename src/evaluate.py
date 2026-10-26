
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted binary classification pipeline."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"],
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_df,
        "classification_report": pd.DataFrame(report).transpose(),
    }
