from prefect import flow, task
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

@task
def load_data(path='data/train.csv'):
    df = pd.read_csv(path)
    print(f'📥 Loaded data from {path}')
    return df

@task
def preprocess(df):
    df = df.fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df)
    print(f'⚙️ Data preprocessing completed. Shape: {df.shape}')
    return df

@task
def exploratory_data_analysis(df, artifacts_folder):
    corr = df.corr()
    output_file = os.path.join(artifacts_folder, 'eda_correlation.png')
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(output_file)
    print(f'📊 Saved correlation heatmap at {output_file}')
    return corr

@flow
def data_pipeline_flow():
    # ------------------------
    # Timestamped folders for this run
    # ------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_folder = f'logs/data/{timestamp}'
    artifacts_folder = f'artifacts/data/{timestamp}'
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(artifacts_folder, exist_ok=True)
    print(f'🗂️ Data pipeline run folders created:\nLogs: {logs_folder}\nArtifacts: {artifacts_folder}')

    # ------------------------
    # Run pipeline tasks
    # ------------------------
    df = load_data()
    df_processed = preprocess(df)
    exploratory_data_analysis(df_processed, artifacts_folder)

    print('✅ Data pipeline completed successfully!')

if __name__ == '__main__':
    data_pipeline_flow()
