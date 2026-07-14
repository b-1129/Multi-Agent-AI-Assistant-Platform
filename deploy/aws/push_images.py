"""
Build and push Docker images to ECR.

Usage (before running deploy.py):
    python deploy/aws/push_images.py --tag abc1234

What it does:
    1. Authenticate Docker with ECR (aws ecr get-login-password)
    2. Build the 'api' target from Dockerfile.prod
    3. Build the 'mcp-server' target from Dockerfile.prod
    4. Tag and push both to ECR

Prerequisites:
    - Docker running locally
    - AWS credentials configured
    - ECR repos exist (run deploy.py first, or they're created here)
"""

import argparse
import os
import subprocess
import sys

import boto3


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=check, text=True)
    return result


def get_ecr_login(region: str) -> tuple[str, str]:
    """Returns (registry_url, docker_password)."""
    ecr = boto3.client("ecr", region_name=region)
    token = ecr.get_authorization_token()
    import base64
    auth = base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode()
    _, password = auth.split(":", 1)
    registry = token["authorizationData"][0]["proxyEndpoint"]
    return registry, password


def push_images(tag: str, region: str, account_id: str, dry_run: bool) -> None:
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    images = [
        {
            "target": "api",
            "repo":   f"{registry}/agent-platform-api",
            "tag":    tag,
        },
        {
            "target": "mcp-server",
            "repo":   f"{registry}/agent-platform-mcp",
            "tag":    tag,
        },
    ]

    if not dry_run:
        # ECR auth
        print("Authenticating with ECR...")
        _, password = get_ecr_login(region)
        run(["docker", "login", "--username", "AWS", "--password-stdin", registry],
            check=False)

    for img in images:
        local_tag = f"{img['repo']}:{img['tag']}"
        latest_tag = f"{img['repo']}:latest"

        if dry_run:
            print(f"[dry-run] Would build: --target {img['target']} -> {local_tag}")
            print(f"[dry-run] Would push: {local_tag}")
            print(f"[dry-run] Would push: {latest_tag}")
            continue

        # Build
        run([
            "docker", "build",
            "-f", os.path.join(project_root, "Dockerfile.prod"),
            "--target", img["target"],
            "-t", local_tag,
            "-t", latest_tag,
            project_root,
        ])

        # Push
        run(["docker", "push", local_tag])
        run(["docker", "push", latest_tag])
        print(f"Pushed: {local_tag}")

    print(f"\nImages pushed with tag: {tag}")
    print("Run next:")
    print(f"  python deploy/aws/deploy.py --image-tag {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker images to ECR")
    parser.add_argument("--tag",        default="latest",                     help="Image tag")
    parser.add_argument("--region",     default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--account-id", default=os.environ.get("AWS_ACCOUNT_ID", ""))
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    account_id = args.account_id
    if not account_id and not args.dry_run:
        sts = boto3.client("sts", region_name=args.region)
        account_id = sts.get_caller_identity()["Account"]

    push_images(args.tag, args.region, account_id or "123456789012", args.dry_run)
