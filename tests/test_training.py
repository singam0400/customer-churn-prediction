import pandas as pd

from src.train import train_and_evaluate


def test_training_pipeline_returns_expected_models():
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(40)],
        "tenure": [i % 60 + 1 for i in range(40)],
        "MonthlyCharges": [30.0 + (i % 20) * 2 for i in range(40)],
        "TotalCharges": [str(100.0 + i * 50) for i in range(40)],
        "Contract": [
            ["Month-to-month", "One year", "Two year"][i % 3]
            for i in range(40)
        ],
        "InternetService": [
            ["DSL", "Fiber optic", "No"][i % 3]
            for i in range(40)
        ],
        "Churn": ["Yes" if i % 2 else "No" for i in range(40)],
    })

    output = train_and_evaluate(
        df,
        experiment_name="customer-churn-test",
        test_size=0.25,
        random_state=42,
    )

    assert "Logistic Regression" in output["models"]
    assert "Random Forest" in output["models"]
    assert not output["results"].empty


def test_trained_models_produce_valid_predictions():
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(40)],
        "tenure": [i % 60 + 1 for i in range(40)],
        "MonthlyCharges": [30.0 + (i % 20) * 2 for i in range(40)],
        "TotalCharges": [str(100.0 + i * 50) for i in range(40)],
        "Contract": [
            ["Month-to-month", "One year", "Two year"][i % 3]
            for i in range(40)
        ],
        "InternetService": [
            ["DSL", "Fiber optic", "No"][i % 3]
            for i in range(40)
        ],
        "Churn": ["Yes" if i % 2 else "No" for i in range(40)],
    })

    output = train_and_evaluate(
        df,
        experiment_name="customer-churn-prediction-test",
        test_size=0.25,
        random_state=42,
    )

    for model in output["models"].values():
        predictions = model.predict(output["X_test"])

        assert len(predictions) == len(output["y_test"])
        assert set(predictions).issubset({0, 1})
