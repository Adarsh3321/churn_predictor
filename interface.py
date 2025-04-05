# import numpy as np
# import pandas as pd
# import joblib
# from openvino.runtime import Core

# # 1. Load the scaler
# scaler = joblib.load("scaler.pkl")

# # 2. Load and preprocess the dataset
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv").dropna()
# df_encoded = pd.get_dummies(df, drop_first=True)
# X = df_encoded.drop("Churn_Yes", axis=1)

# # 3. Scale one sample input
# # sample_input = scaler.transform([X.iloc[0].values]).astype(np.float32)
# sample_input = scaler.transform(pd.DataFrame([X.iloc[0]], columns=X.columns)).astype(np.float32)


# # 4. Load OpenVINO model
# core = Core()
# model_ov = core.read_model("openvino_model/xgb_churn_model_hb.xml")
# compiled_model = core.compile_model(model_ov, "CPU")

# # 5. Get input/output layer names
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)

# # 6. Run inference
# result = compiled_model([sample_input])[output_layer]

# # 7. Output prediction
# print("Prediction:", result)







import numpy as np
import pandas as pd
import joblib
import gradio as gr
from openvino.runtime import Core

# Load OpenVINO model
core = Core()
model_ov = core.read_model("openvino_model/xgb_churn_model_hb.xml")
compiled_model = core.compile_model(model_ov, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load scaler and feature names
scaler = joblib.load("scaler.pkl")
feature_names = list(pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv").dropna().drop("customerID", axis=1).select_dtypes(include=[np.number]).columns)

def predict_churn(*inputs):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    scaled = scaler.transform(input_df).astype(np.float32)
    prediction = compiled_model([scaled])[output_layer]
    return int(prediction[0])  # 0 or 1

# Create interface dynamically based on feature names
inputs = [gr.Number(label=feat) for feat in feature_names]
output = gr.Label(num_top_classes=2)

# Launch Gradio interface
gr.Interface(fn=predict_churn, inputs=inputs, outputs=output, title="Churn Prediction", description="Enter the values below to predict if a customer will churn.").launch()
