"""
Deploy both ML models to Amazon SageMaker as hosted endpoints.

Uses Python 3.9 + sklearn 1.2.2 (matches the SageMaker sklearn 1.2-1 container).
Passes inference code via source_dir + entry_point so SageMaker re-tars the
model bundle correctly with the inference module registered.

Run: python scripts/deploy_sagemaker.py
"""
import os
import sys
import tarfile
import time
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv, set_key
from sagemaker.sklearn.model import SKLearnModel

load_dotenv()

ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / ".env"

REGION = os.getenv("AWS_REGION", "us-east-1")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("SAGEMAKER_BUCKET")

if not ROLE_ARN or not BUCKET:
    sys.exit("ERROR: Set SAGEMAKER_ROLE_ARN and SAGEMAKER_BUCKET in .env first.")

session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=session, default_bucket=BUCKET)


def package_model_only(model_dir: Path, archive_name: str) -> Path:
    """Tar JUST model.joblib (no code/). SageMaker SDK will repack it
    with our inference script when source_dir + entry_point are set."""
    out = model_dir / archive_name
    with tarfile.open(out, "w:gz") as tar:
        tar.add(model_dir / "model.joblib", arcname="model.joblib")
    print(f"  Packaged: {out}")
    return out


def upload_to_s3(local_path: Path, s3_key: str) -> str:
    s3 = session.client("s3")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    uri = f"s3://{BUCKET}/{s3_key}"
    print(f"  Uploaded to: {uri}")
    return uri


def deploy_model(model_data: str, source_dir: Path, endpoint_name: str,
                 instance_type: str = "ml.t2.medium"):
    print(f"  Deploying endpoint '{endpoint_name}' (this takes 5-8 min)...")
    # source_dir = local directory containing inference.py
    # entry_point = filename within source_dir
    # The SDK uploads source_dir to S3 and configures the container to load it.
    model = SKLearnModel(
        model_data=model_data,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir=str(source_dir),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    print(f"  ✅ Endpoint live: {endpoint_name}")
    return predictor


def main():
    timestamp = int(time.time())

    print("\n=== REGRESSION MODEL (Random Forest, California Housing) ===")
    reg_dir = ROOT / "ml" / "regression"
    if not (reg_dir / "model.joblib").exists():
        sys.exit("ERROR: Run `python ml/regression/train.py` first.")
    if not (reg_dir / "inference.py").exists():
        sys.exit("ERROR: ml/regression/inference.py is missing.")
    reg_tar = package_model_only(reg_dir, "model.tar.gz")
    reg_s3 = upload_to_s3(reg_tar, f"models/regression/model-{timestamp}.tar.gz")
    reg_endpoint_name = f"housing-rf-{timestamp}"
    deploy_model(
        model_data=reg_s3,
        source_dir=reg_dir,  # contains inference.py
        endpoint_name=reg_endpoint_name,
    )
    set_key(str(ENV_FILE), "SAGEMAKER_REGRESSION_ENDPOINT", reg_endpoint_name)

    print("\n=== CLASSIFICATION MODEL (Logistic Regression, Bank Marketing) ===")
    clf_dir = ROOT / "ml" / "classification"
    if not (clf_dir / "model.joblib").exists():
        sys.exit("ERROR: Run `python ml/classification/train.py` first.")
    if not (clf_dir / "inference.py").exists():
        sys.exit("ERROR: ml/classification/inference.py is missing.")
    clf_tar = package_model_only(clf_dir, "model.tar.gz")
    clf_s3 = upload_to_s3(clf_tar, f"models/classification/model-{timestamp}.tar.gz")
    clf_endpoint_name = f"bank-lr-{timestamp}"
    deploy_model(
        model_data=clf_s3,
        source_dir=clf_dir,
        endpoint_name=clf_endpoint_name,
    )
    set_key(str(ENV_FILE), "SAGEMAKER_CLASSIFICATION_ENDPOINT", clf_endpoint_name)

    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"Regression endpoint    : {reg_endpoint_name}")
    print(f"Classification endpoint: {clf_endpoint_name}")
    print("\n⚠️  To delete endpoints later: python scripts/delete_endpoints.py")


if __name__ == "__main__":
    main()
