import asyncio
from prefect.client.orchestration import get_client
from collections import Counter

async def fetch_prefect_details():
    async with get_client() as client:
        # 1️⃣ Get all flows
        flows = await client.read_flows()
        if not flows:
            print("No flows found in the workspace.")
            return

        for flow in flows:
            print("="*80)
            print(f"Flow Name: {flow.name}")
            print(f"Flow ID: {flow.id}")

            # 2️⃣ Get deployments for this flow
            deployments = await client.read_deployments()
            flow_deployments = [d for d in deployments if d.flow_id == flow.id]

            if flow_deployments:
                for deployment in flow_deployments:
                    print(f"\n  Deployment Name: {deployment.name}")
                    print(f"  Deployment ID: {deployment.id}")
                    print(f"  Deployment Schedule: {deployment.schedule}")

                    # 3️⃣ Last 5 runs info
                    flow_runs = await client.read_flow_runs(deployment_id=deployment.id, limit=5)
                    if flow_runs:
                        print("    Last 5 Flow Runs:")
                        states_counter = Counter()
                        for run in flow_runs:
                            state = run.state_type
                            states_counter[state] += 1
                            print(f"      Run ID: {run.id}, State: {state}, Start: {run.start_time}, End: {run.end_time}")
                        # 4️⃣ Summary of run states
                        print(f"    Flow Run State Summary: {dict(states_counter)}")
                    else:
                        print("    No flow runs yet.")
            else:
                print("  No deployments for this flow.")

        print("="*80)
        print("All flows & deployments details fetched successfully!")

# Run asynchronously
if __name__ == "__main__":
    asyncio.run(fetch_prefect_details())
