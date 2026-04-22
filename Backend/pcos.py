# File: train_pcos_model.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load Dataset ===
df = pd.read_csv("C:/Users/sneha/OneDrive/Desktop/samhitha/mini_project/models/PCOS_extended_dataset.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# === Encode Categorical Columns ===
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

# === Split X and y ===
X = df.drop(columns=['PCOS'])   # 🔥 Fix: Don't scale 'PCOS'
y = df['PCOS']

# === Scale Features ===
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Balance using SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === Train Model ===
model = XGBClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# === Evaluation ===
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# === Save Model and Scaler ===
joblib.dump(model, 'pcos_model.pkl')
joblib.dump(scaler, 'scaler_pcos.pkl')
