import gradio as gr
import pandas as pd
import numpy as np
import joblib
from openvino.runtime import Core

# Load OpenVINO model
core = Core()
model_ov = core.read_model("openvino_model/xgb_churn_model_hb.xml")
compiled_model = core.compile_model(model_ov, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load scaler and feature list
scaler = joblib.load("scaler.pkl")
feature_names = scaler.feature_names_in_

# Define user interface inputs
input_labels = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

def predict_churn(*args):
    # Map inputs to dataframe
    input_dict = dict(zip(input_labels, args))
    input_df = pd.DataFrame([input_dict])

    # Convert numeric columns
    # input_df["SeniorCitizen"] = int(input_df["SeniorCitizen"])
    # input_df["tenure"] = int(input_df["tenure"])
    # input_df["MonthlyCharges"] = float(input_df["MonthlyCharges"])
    # input_df["TotalCharges"] = float(input_df["TotalCharges"])

    input_df["SeniorCitizen"] = int(input_df["SeniorCitizen"].iloc[0])
    input_df["tenure"] = int(input_df["tenure"].iloc[0])
    input_df["MonthlyCharges"] = float(input_df["MonthlyCharges"].iloc[0])
    input_df["TotalCharges"] = float(input_df["TotalCharges"].iloc[0])


    # One-hot encode categoricals
    input_df = pd.get_dummies(input_df)

    # Add missing dummy columns all at once
    missing_cols = {
        col: [0] for col in feature_names if col not in input_df.columns
    }
    if missing_cols:
        input_df = pd.concat([input_df, pd.DataFrame(missing_cols)], axis=1)

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Scale input
    scaled = scaler.transform(input_df).astype(np.float32)

    # Run prediction
    result = compiled_model([scaled])[output_layer]
    prediction = int(result[0] > 0.5)
    return "Churn" if prediction == 1 else "No Churn"

# Build the Gradio UI
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Checkbox(label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, step=1, label="Tenure (months)"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["No", "Yes", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Online Security"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Online Backup"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Device Protection"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Tech Support"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs="text",
    title="Customer Churn Prediction",
    description="Predict whether a customer will churn based on their service details.",
)

iface.launch()
