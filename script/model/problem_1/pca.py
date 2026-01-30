from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features.csv"
OUTPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_pca.csv"

ID_COLS = [
    "season",
    "week",
    "celebrity_name",
    "ballroom_partner",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])

    id_df = df[ID_COLS].copy() if set(ID_COLS).issubset(df.columns) else pd.DataFrame()
    feature_df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

    categorical_cols = feature_df.select_dtypes(include=["object", "string"]).columns
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(categorical_cols)),
            ("numeric", numeric_pipeline, list(numeric_cols)),
        ],
        remainder="drop",
    )

    pca = PCA(n_components=0.85, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("pca", pca)])

    transformed = pipeline.fit_transform(feature_df)
    pca_cols = [f"pca_{i + 1}" for i in range(transformed.shape[1])]
    pca_df = pd.DataFrame(transformed, columns=pca_cols)

    output_df = pd.concat([id_df.reset_index(drop=True), pca_df], axis=1)
    output_df.to_csv(OUTPUT_PATH, index=False)



if __name__ == "__main__":
    main()
