
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "cleaned_planets.csv"

# Features used to compare planets
FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_eqt",
    "st_teff"
]


def train_recommendation_model():

    # Load cleaned NASA data
    df = pd.read_csv(DATA_FILE)

    # Keep only available numerical features
    available_features = [
        col for col in FEATURES
        if col in df.columns and df[col].notna().any()
    ]

    if len(available_features) < 2:
        raise ValueError("Not enough numerical features available.")

    # Prepare feature data
    X = df[available_features].copy()

    # Fill missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train K-Means model
    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X_scaled)

    print("K-Means model trained successfully!")
    print("Total planets:", len(df))
    print("Number of clusters:", df["cluster"].nunique())
    print("Features used:", available_features)

    return df, X_scaled, available_features


def recommend_similar_planets(planet_name, df, X_scaled, top_n=5):

    # Find selected planet
    matches = df.index[df["pl_name"] == planet_name].tolist()

    if not matches:
        print("Planet not found!")
        return pd.DataFrame()

    selected_index = matches[0]

    # Calculate similarity distances
    distances = euclidean_distances(
        X_scaled[selected_index].reshape(1, -1),
        X_scaled
    )[0]

    results = df.copy()
    results["similarity_distance"] = distances

    # Recommend planets from the same cluster
    selected_cluster = df.loc[selected_index, "cluster"]

    recommendations = results[
        (results["cluster"] == selected_cluster)
        & (results.index != selected_index)
    ]

    recommendations = recommendations.sort_values(
        by="similarity_distance"
    )

    return recommendations.head(top_n)


if __name__ == "__main__":

    planets, X_scaled, features = train_recommendation_model()

    # Test with the first planet in the dataset
    sample_planet = planets.iloc[0]["pl_name"]

    print("\nSelected planet:", sample_planet)

    recommendations = recommend_similar_planets(
        sample_planet,
        planets,
        X_scaled
    )

    print("\nRecommended similar planets:")
    print(recommendations[["pl_name", "cluster", "similarity_distance"]])
