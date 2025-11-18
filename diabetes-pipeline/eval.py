# eval.py
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

model_path = "/opt/ml/processing/model/model.joblib"
scaler_path = "/opt/ml/processing/model/scaler.joblib"
test_path = "/opt/ml/processing/test/test.csv"
output_dir = "/opt/ml/processing/output"
os.makedirs(output_dir, exist_ok=True)

# Load artifacts
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
test_df = pd.read_csv(test_path)

X_test = test_df.drop("Outcome", axis=1)
y_test = test_df["Outcome"]
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# Create evaluation.json
report_dict = {
    "accuracy": acc,
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}

with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump(report_dict, f)

print("✅ Evaluation complete. Accuracy:", acc)