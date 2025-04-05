# # import joblib
# # import pandas as pd
# # from onnxmltools.convert import convert_xgboost
# # from skl2onnx.common.data_types import FloatTensorType

# # # Load model and sample data
# # model = joblib.load("xgb_churn_model.pkl")
# # df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# # df = df.dropna()
# # df_encoded = pd.get_dummies(df, drop_first=True)
# # X = df_encoded.drop("Churn_Yes", axis=1)

# # # Define input shape for ONNX conversion
# # initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# # # Convert model to ONNX format
# # onnx_model = convert_xgboost(model, initial_types=initial_type)

# # # Save ONNX model to file
# # with open("xgb_churn_model.onnx", "wb") as f:
# #     f.write(onnx_model.SerializeToString())

# # print("✅ Model successfully converted to ONNX and saved as 'xgb_churn_model.onnx'")



# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler
# from hummingbird.ml import convert

# # Load model and data
# model = joblib.load("xgb_churn_model.pkl")
# scaler = joblib.load("scaler.pkl")
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv").dropna()
# df_encoded = pd.get_dummies(df, drop_first=True)
# X = df_encoded.drop("Churn_Yes", axis=1)

# # Convert with Hummingbird to ONNX (NN backend)
# model_hb = convert(model, 'onnx', X.values.astype(np.float32))

# # Save to ONNX file
# with open("xgb_churn_model_hb.onnx", "wb") as f:
#     f.write(model_hb.model.SerializeToString())

# print("✅ Model converted to ONNX using Hummingbird and saved as 'xgb_churn_model_hb.onnx'")




import pandas as pd
import numpy as np
import joblib
from hummingbird.ml import convert
import torch

# Load model and data
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load and preprocess data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv").dropna()
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Churn_Yes", axis=1)

# Scale data like training
X_scaled = scaler.transform(X)

# Convert XGBoost to PyTorch model
model_hb = convert(model, backend="torch", test_input=X_scaled.astype(np.float32))

# Create dummy input for export
dummy_input = torch.from_numpy(X_scaled[:1].astype(np.float32))

# Export to ONNX
torch.onnx.export(
    model_hb.model,               # The PyTorch model
    dummy_input,                  # An example input
    "xgb_churn_model_hb.onnx",    # Output file
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Successfully exported Hummingbird (PyTorch) model to ONNX!")
