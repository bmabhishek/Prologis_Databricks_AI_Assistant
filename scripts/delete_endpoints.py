"""
DELETE SageMaker endpoints to stop incurring charges.
Run this after your demo is recorded!

Run: python scripts/delete_endpoints.py
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
REG_EP = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT")
CLF_EP = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT")

sm = boto3.client("sagemaker", region_name=REGION)


def delete(endpoint_name):
    if not endpoint_name:
        print(f"  (no endpoint name set)")
        return
    try:
        # Delete endpoint
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"  ✅ Deleted endpoint: {endpoint_name}")
        # Also delete the endpoint config and model to be tidy
        try:
            sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
            print(f"  ✅ Deleted endpoint config: {endpoint_name}")
        except Exception as e:
            print(f"  (endpoint config delete: {e})")
    except sm.exceptions.ClientError as e:
        print(f"  ⚠️  Could not delete {endpoint_name}: {e}")


if __name__ == "__main__":
    print("Deleting SageMaker endpoints...")
    print(f"\nRegression: {REG_EP}")
    delete(REG_EP)
    print(f"\nClassification: {CLF_EP}")
    delete(CLF_EP)
    print("\nDone. Verify in AWS console: SageMaker → Inference → Endpoints (should be empty).")
