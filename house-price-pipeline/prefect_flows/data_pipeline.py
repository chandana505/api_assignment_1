# prefect_flows/data_pipeline.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from datetime import datetime

# Setup folders for logs & artifacts
def setup_folders():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_folder = f"logs/data/{timestamp}"
    artifacts_folder = f"artifacts/data/{timestamp}"
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)
    return logs_folder, artifacts_folder

# Data Ingestion
@task
def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Loaded data from {path}, shape: {df.shape}")
    return df

# Preprocessing
@task
def preprocess(df: pd.DataFrame):
    print("Preprocessing started...")

    print("Summary statistics:\n", df.describe(include='all'))

    print("Missing values per column:\n", df.isnull().sum())

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Normalize numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"Preprocessing completed. Shape: {df.shape}")
    return df

# Exploratory Data Analysis
@task
def exploratory_data_analysis(df: pd.DataFrame, artifacts_folder: str):
    print("EDA started...")

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    corr = df_encoded.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    corr_path = os.path.join(artifacts_folder, "eda_correlation.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"Saved correlation heatmap at {corr_path}")

    target = 'SalePrice'
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target)

    top_corr_features = corr[target].abs().sort_values(ascending=False)[1:11].index.tolist()  # top 10 features
    for col in top_corr_features:
        plt.figure()
        sns.scatterplot(x=df_encoded[col], y=df_encoded[target])
        scatter_path = os.path.join(artifacts_folder, f"{col}_vs_{target}.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"Saved scatterplot {col} vs {target} at {scatter_path}")

    X = df_encoded[top_corr_features]
    y = df_encoded[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=top_corr_features).sort_values(ascending=False)

    fi_path = os.path.join(artifacts_folder, "feature_importance.png")
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved feature importance plot at {fi_path}")

    print("EDA completed.")

@flow(name="data-pipeline-flow", task_runner=ConcurrentTaskRunner())
def data_pipeline_flow():
    logs_folder, artifacts_folder = setup_folders()
    df = load_data("data/train.csv")
    df = preprocess(df)
    exploratory_data_analysis(df, artifacts_folder)
    print("Data pipeline completed successfully!")

if __name__ == "__main__":
    data_pipeline_flow()
