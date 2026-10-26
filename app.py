from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from src.preprocessing import clean_data, split_features_target


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "churn_pipeline.joblib"

model = joblib.load(MODEL_PATH)


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    """Predict churn from a customer's account and service details."""

    raw_customer = pd.DataFrame([{
        "customerID": "inference_customer",
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        # Placeholder required by the shared cleaning function.
        "Churn": "No",
    }])

    cleaned = clean_data(raw_customer)
    features, _ = split_features_target(cleaned)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    result = "Likely to churn" if prediction == 1 else "Likely to stay"

    return result, round(probability, 4)


yes_no = ["Yes", "No"]
service_options = ["No", "Yes", "No internet service"]
line_options = ["No", "Yes", "No phone service"]

demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender", value="Female"),
        gr.Radio([0, 1], label="Senior Citizen", value=0),
        gr.Dropdown(yes_no, label="Partner", value="Yes"),
        gr.Dropdown(yes_no, label="Dependents", value="No"),
        gr.Slider(0, 72, step=1, value=12, label="Tenure (months)"),
        gr.Dropdown(yes_no, label="Phone Service", value="Yes"),
        gr.Dropdown(line_options, label="Multiple Lines", value="No"),
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service",
            value="DSL",
        ),
        gr.Dropdown(service_options, label="Online Security", value="No"),
        gr.Dropdown(service_options, label="Online Backup", value="No"),
        gr.Dropdown(service_options, label="Device Protection", value="No"),
        gr.Dropdown(service_options, label="Tech Support", value="No"),
        gr.Dropdown(service_options, label="Streaming TV", value="No"),
        gr.Dropdown(service_options, label="Streaming Movies", value="No"),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month",
        ),
        gr.Dropdown(yes_no, label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        ),
        gr.Number(label="Monthly Charges", value=70.0, minimum=0),
        gr.Number(label="Total Charges", value=840.0, minimum=0),
    ],
    outputs=[
        gr.Textbox(label="Churn Prediction"),
        gr.Number(label="Estimated Churn Probability"),
    ],
    title="Customer Churn Prediction",
    description=(
        "Enter customer account and service details to estimate churn risk. "
        "The probability is a model estimate, not a guarantee."
    ),
)


if __name__ == "__main__":
    demo.launch()
