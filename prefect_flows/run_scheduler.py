import time
from prefect_flows.data_pipeline import data_pipeline_flow

while True:
    print("Running flow...")
    data_pipeline_flow()
    print("Flow completed. Sleeping for 2 minutes...")
    time.sleep(120)  # 120 seconds = 2 minutes
