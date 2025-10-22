from prefect import flow, task
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@task
def load_data(path):
    df = pd.read_csv(path)
    return df

@task
def preprocess(df):
    df = df.fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df)
    return df

@task
def exploratory_data_analysis(df):
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('eda_correlation.png')
    return corr

@flow
def data_pipeline_flow():
    df = load_data('data/train.csv')
    df_processed = preprocess(df)
    exploratory_data_analysis(df_processed)
    print('✅ Data pipeline completed successfully!')

if __name__ == '__main__':
    data_pipeline_flow()
