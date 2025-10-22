# prefect_flows/ml_pipeline.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from prefect import task, flow

# ---------- Tasks ----------

@task
def load_processed_data(file_path="data/train.csv") -> pd.DataFrame:
    """Load processed data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")
    data = pd.read_csv(file_path)
    print(f"Data loaded with shape: {data.shape}")
    return data

@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset without removing the target column."""
    # Fill missing numerical values with median
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode 'ocean_proximity' categorical feature if exists
    if "ocean_proximity" in df.columns:
        df["ocean_proximity"] = df["ocean_proximity"].map({
            "NEAR BAY": 1,
            "INLAND": 0,
            "NEAR OCEAN": 2,
            "ISLAND": 3,
            "1H OCEAN": 4
        }).fillna(0)

    # Create rooms_per_household feature if possible
    if "total_rooms" in df.columns and "households" in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / df["households"]

    print(f"Preprocessing done. Columns now: {df.columns.tolist()}")
    return df

@task
def split_data(df: pd.DataFrame, target_col="SalePrice"):
    """Split dataset into train and test sets with proper imputation."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found! Available columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Impute numeric columns with median
    num_imputer = SimpleImputer(strategy="median")
    X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

    # Impute categorical columns with mode (most frequent)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_categorical = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

    # Combine numeric + categorical
    X_imputed = pd.concat([X_numeric, X_categorical], axis=1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    print(f"Split done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

@task
def train_models(X_train, y_train, save_dir="artifacts/ml"):
    """Train Linear Regression and Random Forest models with proper encoding."""
    os.makedirs(save_dir, exist_ok=True)

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    # One-hot encode categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # keep numeric columns as they are
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_processed, y_train)
    joblib.dump(lr, os.path.join(save_dir, "linear_regression.pkl"))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_processed, y_train)
    joblib.dump(rf, os.path.join(save_dir, "random_forest.pkl"))

    # Save preprocessor for test data
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))

    print(f"Models trained and saved in {save_dir}")
    return lr, rf, preprocessor

@task
def evaluate_models(models, preprocessor, X_test, y_test, save_dir="artifacts/ml"):
    """Evaluate trained models and save metrics."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = {}

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    for model, name in zip(models, ["LinearRegression", "RandomForest"]):
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"MSE": mse, "R2": r2}

    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        for model_name, metric_values in metrics.items():
            f.write(f"{model_name}:\n")
            for k, v in metric_values.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"Evaluation metrics saved in {metrics_path}")
    return metrics

# ---------- Flow ----------

@flow(name="ml-pipeline-flow")
def ml_pipeline_flow():
    """Main ML pipeline flow."""
    df = load_processed_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    lr_model, rf_model, preprocessor = train_models(X_train, y_train)
    metrics = evaluate_models([lr_model, rf_model], preprocessor, X_test, y_test)
    print("Pipeline finished. Metrics:", metrics)

# Run the flow
if __name__ == "__main__":
    ml_pipeline_flow()
