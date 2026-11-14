
import pandas as pd
import numpy as np
from pathlib import Path

# Locate project folders
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "confirmed_planets.csv"
output_file = project_root / "data" / "cleaned_planets.csv"

def preprocess_data():

    # Load NASA dataset
    df = pd.read_csv(input_file)

    print("Original dataset shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates(subset=["pl_name"])

    # Convert numerical columns
    numeric_columns = [
        "disc_year",
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "pl_eqt",
        "st_teff"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Save cleaned dataset
    df.to_csv(output_file, index=False)

    print("Data preprocessing completed!")
    print("Cleaned dataset shape:", df.shape)
    print("Missing values:")
    print(df[numeric_columns].isnull().sum())
    print("Saved to:", output_file)


if __name__ == "__main__":
    preprocess_data()
