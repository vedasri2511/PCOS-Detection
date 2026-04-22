import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = "C:/Users/sneha/OneDrive/Desktop/samhitha/mini_project/emotional_monitoring_dataset_with_target.csv"
df = pd.read_csv(file_path)

# Select relevant features
selected_features = [
    "HeartRate", "SkinConductance", "EEG", "Temperature", "PupilDiameter", 
    "FrownIntensity", "CortisolLevel", "EmotionalState", "CognitiveState"
]
target_variable = "EngagementLevel"

# Encode categorical variables (EmotionalState & CognitiveState)
label_encoder = LabelEncoder()
df["EmotionalState"] = label_encoder.fit_transform(df["EmotionalState"])
df["CognitiveState"] = label_encoder.fit_transform(df["CognitiveState"])

# Handle missing values (fill with median)
df[selected_features] = df[selected_features].apply(pd.to_numeric, errors='coerce')  # Convert all to numeric
df[selected_features] = df[selected_features].fillna(df[selected_features].median()) # Fill NaN with median

# Convert Engagement Level (1,2,3) → (0,1,2)
df[target_variable] = df[target_variable] - 1

# Split dataset into train & test
X = df[selected_features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print Metrics
print(f"Model: XGBoost")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Saving the model and scaler
joblib.dump(xgb_model, 'stress_model.pkl')  # Save the trained model
joblib.dump(scaler, 'scaler_stress.pkl')  # Save the scaler used for scaling features
