from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# === Load Models and Scalers ===
scaler_stress = joblib.load('scaler_stress.pkl')
nonclinical_model = joblib.load('non_clinical_model.pkl')
scaler_nonclinical = joblib.load('scaler_nonclinical.pkl')
pcos_model = joblib.load('pcos_model.pkl')
scaler_pcos = joblib.load('scaler_pcos.pkl')

# === Load Datasets ===
df_stress = pd.read_csv("../Data/emotional_monitoring_dataset_with_target.csv")
df_stress.replace("?", np.nan, inplace=True)
df_stress = df_stress.apply(pd.to_numeric, errors='coerce')
stress_means = df_stress.mean()

df_pcos = pd.read_csv("../Data/PCOS_extended_dataset.csv")
if 'PCOS' in df_pcos.columns:
    df_pcos_features = df_pcos.drop(columns=['PCOS'])
else:
    df_pcos_features = df_pcos.copy()

df_pcos_features = df_pcos_features.apply(pd.to_numeric, errors='coerce')
df_pcos_features.fillna(df_pcos_features.mean(numeric_only=True), inplace=True)
df_pcos_features.drop_duplicates(inplace=True)
categorical_cols = df_pcos_features.select_dtypes(include='object').columns
df_pcos_features[categorical_cols] = df_pcos_features[categorical_cols].apply(
    lambda x: x.astype('category').cat.codes
)

pcos_feature_list = df_pcos_features.columns.tolist()

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json
    clinical = data.get('clinical', {})
    non_clinical = data.get('non_clinical', {})
    gsr_input = data.get('gsr_features', {})

    print(f"[GSR INPUT RECEIVED] => {gsr_input}")

    # === Rule-Based Stress Prediction ===
    if "SkinConductance" not in gsr_input:
        return jsonify({"error": "SkinConductance value missing in GSR input"}), 400

    sc = gsr_input["SkinConductance"]
    if sc < 0.25:
        stress_numeric = 0  # Low
    elif sc < 0.5:
        stress_numeric = 1  # Mid
    else:
        stress_numeric = 2  # High

    stress_label = ["Low", "Mid", "High"][stress_numeric]
    stress_score = [0, 0.5, 1][stress_numeric]  # Used for ensemble

    print(f"[RULE-BASED STRESS] => SC: {sc} | Level: {stress_label} | Numeric: {stress_numeric}")

    # === Non-Clinical Model Path ===
    non_clinical["StressLevel"] = stress_numeric
    nonclinical_features = [
        "AGE", "WeightKg", "HeightCm", "BMI", "CYCLE_LENGTH",
        "WeightGain", "HairGrowth", "SkinDarkening",
        "HairLoss", "FastFood", "RegExercise", "StressLevel"
    ]
    nc_input_filled = [non_clinical.get(f, 0) for f in nonclinical_features]
    nc_scaled = scaler_nonclinical.transform([nc_input_filled])
    nc_pred = int(nonclinical_model.predict(nc_scaled)[0])
    nc_conf = round(nonclinical_model.predict_proba(nc_scaled)[0][1] * 100, 2)

    # === Decide Which Model to Use ===
    if len(clinical.keys()) < 2:
        print("[USING NON-CLINICAL MODEL]")
        model_used = "non_clinical_model"
        pcos_prob = nc_conf
    else:
        print("[USING FULL PCOS MODEL]")
        model_used = "full_pcos_model"
        combined_input = {**non_clinical, **clinical}
        combined_input["StressLevel"] = stress_numeric

        input_vector = []
        for col in pcos_feature_list:
            if col in combined_input:
                input_vector.append(combined_input[col])
            else:
                print(f"[MISSING FEATURE] {col} set to 0")
                input_vector.append(0)

        input_scaled = scaler_pcos.transform([input_vector])
        input_df_scaled = pd.DataFrame(input_scaled, columns=pcos_feature_list)

        pcos_pred = int(pcos_model.predict(input_df_scaled)[0])
        pcos_prob = round(pcos_model.predict_proba(input_df_scaled)[0][1] * 100, 2)

    # === Final Risk Score (Weighted Average with Stress) ===
    final_score = 0.8 * (pcos_prob / 100) + 0.2 * stress_score
    final_prob = round(final_score * 100, 2)

    if final_prob < 30:
        risk = "Low"
    elif final_prob < 70:
        risk = "Mid"
    else:
        risk = "High"

    print(f"[PCOS PROB] = {pcos_prob}%, [STRESS SCORE] = {stress_score}")
    print(f"[ENSEMBLE SCORE] => Final Score: {final_prob}% | Risk: {risk}")

    return jsonify({
        "stress_level": str(stress_label),
        "pcos_probability": float(pcos_prob),
        "final_combined_score": float(final_prob),
        "pcos_risk": str(risk),
        "model_used": str(model_used)
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)