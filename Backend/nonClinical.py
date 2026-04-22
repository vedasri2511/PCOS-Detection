import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load dataset
file_path = "C:/Users/sneha/OneDrive/Desktop/samhitha/mini_project/PCOS_extended_dataset.csv"
df = pd.read_csv(file_path)

# Step 2: Strip spaces from column names
df.columns = df.columns.str.strip()

# Step 3: Select non-clinical features + target
selected_features = [
    "AGE", "Weight (Kg)", "Height(Cm)", "BMI", "CYCLE_LENGTH",
    "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
    "Hair loss(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", "StressLevel"
]
target_column = "PCOS"

# Step 4: Add StressLevel column if not present
if "StressLevel" not in df.columns:
    df["StressLevel"] = np.random.randint(0, 3, df.shape[0])  # Placeholder for testing

# Step 5: Drop missing data
df = df[selected_features + [target_column]].dropna()

# Step 6: Convert categorical Y/N columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].map({"Y": 1, "N": 0})

# Step 7: Split features and labels
X = df[selected_features]
y = df[target_column]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train XGBoost classifier
model = XGBClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 11: Evaluate
y_pred = model.predict(X_test_scaled)
print("\u2705 Accuracy:", accuracy_score(y_test, y_pred))
print("\u2705 Classification Report:\n", classification_report(y_test, y_pred))

# Step 12: Save the model and scaler
joblib.dump(model, 'non_clinical_model.pkl')       # Save the trained model
joblib.dump(scaler, 'scaler_nonclinical.pkl')      # Save the feature scaler
