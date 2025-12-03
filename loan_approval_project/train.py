import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("=" * 60)
print("🚀 LOAN APPROVAL MODEL TRAINING - SIMPLE VERSION")
print("=" * 60)

# Load data from current folder
try:
    df = pd.read_csv('data/loan_data.csv')
    print("✅ Dataset loaded successfully!")
except:
    print("❌ ERROR: Could not load data/loan_data.csv")
    print("Make sure you're in loan_approval_project folder")
    exit()

# Clean column names
df.columns = df.columns.str.strip()
print("✅ Column names cleaned")

print(f"\n📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Show distribution
print(f"\n🎯 Loan Status Distribution:")
print(df['loan_status'].value_counts())

# Encode categorical
print("\n🔤 Encoding categorical variables...")
label_encoders = {}
# Clean + normalize categorical text
for col in ['education', 'self_employed', 'loan_status']:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

# Encode categorical
label_encoders = {}
for col in ['education', 'self_employed', 'loan_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✅ {col} encoded (classes = {le.classes_})")

    label_encoders[col] = le
    print(f"   ✅ {col} encoded")

# Features and target
target = 'loan_status'
features = [col for col in df.columns if col not in ['loan_id', target]]

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("\n🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {accuracy:.2%}")

# Save
print("\n💾 Saving model files...")
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/loan_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(features, 'models/feature_names.pkl')
joblib.dump(target, 'models/target_name.pkl')

print("✅ Model saved successfully!")
print("\n📁 Files created in 'models/' folder:")
for file in os.listdir('models'):
    print(f"   - {file}")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)