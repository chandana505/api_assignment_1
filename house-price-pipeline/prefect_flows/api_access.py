# prefect_flows/api_access.py

import asyncio
from prefect.client.orchestration import get_client

async def fetch_prefect_details():
    async with get_client() as client:
        # Fetch all flows
        flows = await client.read_flows()
        for flow in flows:
            print("\n--- Flow Details ---")
            print(f"Flow Name       : {flow.name}")
            print(f"Flow ID         : {flow.id}")
            print(f"Flow Tags       : {flow.tags}")
            print(f"Flow Created At : {flow.created}")
            print(f"Flow Updated At : {flow.updated}")

            # Fetch deployments for this flow
            deployments = await client.read_deployments()
            flow_deployments = [d for d in deployments if d.flow_id == flow.id]

            if flow_deployments:
                for dep in flow_deployments:
                    print("\n--- Deployment Details ---")
                    print(f"Deployment Name       : {dep.name}")
                    print(f"Deployment ID         : {dep.id}")
                    print(f"Deployment Tags       : {dep.tags}")
                    print(f"Deployment Created At : {dep.created}")
                    print(f"Deployment Updated At : {dep.updated}")
            else:
                print("No deployments found for this flow.")

# Run the async function
if __name__ == "__main__":
    asyncio.run(fetch_prefect_details())
