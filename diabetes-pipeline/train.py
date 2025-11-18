# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

train_path = "/opt/ml/input/data/training/train.csv"
model_dir = "/opt/ml/model"

os.makedirs(model_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(train_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_scaled, y)

# Save model and scaler
joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

print("✅ Model trained and saved at:", model_dir)