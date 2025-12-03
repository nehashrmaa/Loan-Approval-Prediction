import os
import sys
from flask import Flask, render_template, request
import joblib
import pandas as pd

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# -----------------------------
# Load Model + Artifacts
# -----------------------------
def load_model():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_path = os.path.join(project_root, "models")
        
        model = joblib.load(os.path.join(models_path, "loan_model.pkl"))
        scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
        label_encoders = joblib.load(os.path.join(models_path, "label_encoders.pkl"))
        feature_names = joblib.load(os.path.join(models_path, "feature_names.pkl"))
        
        print("Model loaded successfully.")
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


model, scaler, label_encoders, feature_names = load_model()


# -----------------------------
# Home Route
# -----------------------------
@app.route('/')
def home():
    if model is None:
        return render_template('error.html', message="Model not loaded. Please run train.py")

    # Default sample (for prefill)
    sample_data = {
        'no_of_dependents': 2,
        'education': 'Graduate',
        'self_employed': 'No',
        'income_annum': 7000000,
        'loan_amount': 3000000,
        'loan_term': 12,
        'cibil_score': 750,
        'residential_assets_value': 3000000,
        'commercial_assets_value': 2000000,
        'luxury_assets_value': 10000000,
        'bank_asset_value': 5000000
    }

    education_options = ["Graduate", "Not Graduate"]
    employment_options = ["Yes", "No"]

    return render_template(
        'index.html',
        sample=sample_data,
        education_options=education_options,
        employment_options=employment_options,
        features=feature_names
    )


# -----------------------------
# Predict Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded", 500

    try:
        # Collect input data in correct feature order
        input_data = {}
        for feature in feature_names:
            input_data[feature] = request.form.get(feature)

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Normalize + encode categorical
        for col in ["education", "self_employed"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                allowed = label_encoders[col].classes_

                if df[col].iloc[0] not in allowed:
                    return render_template(
                        "error.html",
                        message=f"Invalid value '{df[col].iloc[0]}' for {col}. Allowed values: {list(allowed)}"
                    )

                df[col] = label_encoders[col].transform(df[col])

        # Convert numeric types
        numeric_cols = [f for f in feature_names if f not in ['education', 'self_employed']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Apply scaling
        scaled_input = scaler.transform(df[feature_names])

        # Prediction
        pred_class = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]

        pred_label = label_encoders['loan_status'].inverse_transform([pred_class])[0]
        confidence = f"{proba[pred_class] * 100:.1f}%"

        result = {
            'prediction': pred_label,
            'confidence': confidence,
            'approved': pred_label == 'approved'
        }

        return render_template('result.html', result=result, input_data=input_data)

    except Exception as e:
        return render_template('error.html', message=f"Error occurred: {e}")


# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    print("Loan Approval System")
    print("Server running at: http://localhost:5000")
    app.run(debug=True, port=5000)
