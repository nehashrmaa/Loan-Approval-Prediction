import joblib
import os

print("Checking saved model files...")
print("=" * 50)

models_dir = "models"

# Verify models directory exists
if not os.path.exists(models_dir):
    print("Error: 'models/' folder not found.")
    exit()

print(f"Found '{models_dir}/' folder.\n")

# List files in models directory
print("Files in models/ folder:")
for i, file in enumerate(os.listdir(models_dir), start=1):
    file_path = os.path.join(models_dir, file)
    file_size = os.path.getsize(file_path)
    print(f"{i:2}. {file:25} ({file_size:,} bytes)")

print("\nLoading and validating model components:")

try:
    # Load feature names
    features = joblib.load("models/feature_names.pkl")
    print(f"\nFeature names loaded ({len(features)} features):")
    for i, feat in enumerate(features, start=1):
        print(f"   {i:2}. {feat}")

    # Load target variable name
    target = joblib.load("models/target_name.pkl")
    print(f"\nTarget variable: '{target}'")

    # Load label encoders
    encoders = joblib.load("models/label_encoders.pkl")
    print(f"\nLabel encoders loaded for: {list(encoders.keys())}")

    # Display encoder mappings
    for col, encoder in encoders.items():
        print(f"\n   Encoding for '{col}':")
        for cls, code in zip(encoder.classes_, encoder.transform(encoder.classes_)):
            print(f"      {cls} -> {code}")

    # Load scaler
    scaler = joblib.load("models/scaler.pkl")
    print("\nScaler loaded successfully.")

    # Load trained model
    model = joblib.load("models/loan_model.pkl")
    print(f"\nModel loaded: {type(model).__name__}")
    print(f"Number of trees: {model.n_estimators}")

    print("\n" + "=" * 50)
    print("All model files are valid.")
    print("=" * 50)

except Exception as e:
    print(f"\nError while loading model components: {e}")

print("\nReady for next step: Flask app integration.")
