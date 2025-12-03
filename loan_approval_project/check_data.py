import pandas as pd
import os

print("Checking dataset...")
print("=" * 50)

# Path to dataset
file_path = "data/loan_data.csv"

# Check if dataset exists
if not os.path.exists(file_path):
    print("Error: loan_data.csv not found in the data/ folder.")
    print("Expected location: project_root/data/loan_data.csv")
    exit()

print("Dataset located successfully.\n")

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Column list
    print("Column Names:")
    print("-" * 30)
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    # Preview first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"{col}: {count} missing")

except Exception as e:
    print(f"Error while loading dataset: {e}")

print("\n" + "=" * 50)
print("Dataset check complete.")
