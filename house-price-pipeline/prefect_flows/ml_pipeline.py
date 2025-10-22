# prefect_flows/ml_pipeline.py

import pandas as pd
import os
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
from prefect import task, flow, get_run_logger

# ---------- Setup timestamped directories ----------
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs/ml", run_time)
ARTIFACT_DIR = os.path.join("artifacts/ml", run_time)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------- Setup logging ----------
log_file = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ---------- Tasks ----------

@task
def load_processed_data(file_path="data/train.csv") -> pd.DataFrame:
    """Load processed data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded with shape: {data.shape}")
    return data

@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset without removing the target column."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    if "ocean_proximity" in df.columns:
        df["ocean_proximity"] = df["ocean_proximity"].map({
            "NEAR BAY": 1,
            "INLAND": 0,
            "NEAR OCEAN": 2,
            "ISLAND": 3,
            "1H OCEAN": 4
        }).fillna(0)

    if "total_rooms" in df.columns and "households" in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / df["households"]

    logging.info(f"Preprocessing done. Columns now: {df.columns.tolist()}")
    return df

@task
def split_data(df: pd.DataFrame, target_col="SalePrice"):
    """Split dataset into train and test sets with proper imputation."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found! Available columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="median")
    X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_categorical = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

    X_imputed = pd.concat([X_numeric, X_categorical], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=42
    )
    logging.info(f"Split done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Save imputers
    joblib.dump(num_imputer, os.path.join(ARTIFACT_DIR, "num_imputer.pkl"))
    joblib.dump(cat_imputer, os.path.join(ARTIFACT_DIR, "cat_imputer.pkl"))
    logging.info(f"Imputers saved in {ARTIFACT_DIR}")

    return X_train, X_test, y_train, y_test

@task
def train_models(X_train, y_train):
    """Train Linear Regression and Random Forest models with proper encoding."""
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    X_train_processed = preprocessor.fit_transform(X_train)

    lr = LinearRegression()
    lr.fit(X_train_processed, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_processed, y_train)

    # Save models and preprocessor
    joblib.dump(lr, os.path.join(ARTIFACT_DIR, "linear_regression.pkl"))
    joblib.dump(rf, os.path.join(ARTIFACT_DIR, "random_forest.pkl"))
    joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))

    logging.info(f"Models and preprocessor saved in {ARTIFACT_DIR}")
    return lr, rf, preprocessor

@task
def evaluate_models(models, preprocessor, X_test, y_test):
    """Evaluate trained models, save metrics, and log them to Prefect Cloud."""
    logger = get_run_logger()
    X_test_processed = preprocessor.transform(X_test)
    metrics = {}

    for model, name in zip(models, ["LinearRegression", "RandomForest"]):
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        metrics[name] = {"MSE": mse, "R2": r2, "MAE": mae, "RMSE": rmse}

        # Log to Prefect Cloud
        logger.info(f"{name} Metrics: MSE={mse:.2f}, R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Save metrics locally
    metrics_path_txt = os.path.join(ARTIFACT_DIR, "evaluation_metrics.txt")
    with open(metrics_path_txt, "w") as f:
        for model_name, metric_values in metrics.items():
            f.write(f"{model_name}:\n")
            for k, v in metric_values.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    return metrics

# ---------- Flow ----------

@flow(name="ml-pipeline-flow")
def ml_pipeline_flow():
    df = load_processed_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    lr_model, rf_model, preprocessor = train_models(X_train, y_train)
    metrics = evaluate_models([lr_model, rf_model], preprocessor, X_test, y_test)
    logging.info(f"Pipeline finished. Metrics: {metrics}")

# Run the flow
if __name__ == "__main__":
    ml_pipeline_flow()
