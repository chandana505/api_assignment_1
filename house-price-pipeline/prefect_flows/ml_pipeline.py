from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
from datetime import datetime

@task
def load_processed_data(path='data/train.csv'):
    df = pd.read_csv(path)
    df = df.fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df)
    print(f'📥 Loaded and preprocessed data from {path}')
    return df

@task
def split_data(df):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f'⚙️ Split data: Train={X_train.shape}, Test={X_test.shape}')
    return X_train, X_test, y_train, y_test

@task
def train_models(X_train, y_train, models_folder):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_path = os.path.join(models_folder, 'linear_regression.pkl')
    rf_path = os.path.join(models_folder, 'random_forest.pkl')
    joblib.dump(lr, lr_path)
    joblib.dump(rf, rf_path)

    print(f'💾 Saved Linear Regression model at {lr_path}')
    print(f'💾 Saved Random Forest model at {rf_path}')
    return lr, rf
@task
def evaluate_models(lr, rf, X_test, y_test, metrics_folder):
    preds_lr = lr.predict(X_test)
    preds_rf = rf.predict(X_test)

    metrics = {
        'Linear Regression': {
            'RMSE': np.sqrt(mean_squared_error(y_test, preds_lr)),
            'R2': r2_score(y_test, preds_lr)
        },
        'Random Forest': {
            'RMSE': np.sqrt(mean_squared_error(y_test, preds_rf)),
            'R2': r2_score(y_test, preds_rf)
        }
    }

    metrics_file = os.path.join(metrics_folder, 'evaluation_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:  # <-- fix applied
        f.write('📊 Model Evaluation Results:\n')
        for model, vals in metrics.items():
            f.write(f'{model}: RMSE={vals["RMSE"]:.2f}, R2={vals["R2"]:.2f}\n')

    print(f'📄 Saved evaluation metrics at {metrics_file}')
    return metrics

@task
def log_to_aws(metrics):
    # Simulated AWS logging
    print('\n🪣 Logging metrics to AWS SageMaker Experiments (simulated)...')
    for model, vals in metrics.items():
        print(f'Logged {model} -> RMSE={vals["RMSE"]:.2f}, R2={vals["R2"]:.2f}')
    print('✅ Metrics successfully logged!')

@flow
def ml_pipeline_flow():
    # ------------------------
    # Timestamped folders for this run
    # ------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_folder = f'logs/ml/{timestamp}'
    models_folder = f'artifacts/ml/{timestamp}/models'
    metrics_folder = f'artifacts/ml/{timestamp}/metrics'
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    print(f'🗂️ ML pipeline run folders created:\nLogs: {logs_folder}\nModels: {models_folder}\nMetrics: {metrics_folder}')

    # ------------------------
    # Run pipeline tasks
    # ------------------------
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)
    lr, rf = train_models(X_train, y_train, models_folder)
    metrics = evaluate_models(lr, rf, X_test, y_test, metrics_folder)
    log_to_aws(metrics)

    print('🎯 ML Pipeline execution completed successfully!')

if __name__ == '__main__':
    ml_pipeline_flow()
