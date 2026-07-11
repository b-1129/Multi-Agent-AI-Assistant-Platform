"""
AWS deployment script for the agent-platform project.

This script is intentionally written in Python (not shell) so that every
step is readable, testable, and easy to extend. It uses boto3 directly
rather than the AWS CLI so you don't need a different tool installed.

Usage:
    # First time (creates all infrastructure):
    python deploy/aws/deploy.py --env production

    # Update after a code change (new image tag):
    python deploy/aws/deploy.py --env production --image-tag abc1234

    # Dry run (prints plan without making changes):
    python deploy/aws/deploy.py --env production --dry-run

Prerequisites:
    pip install boto3
    AWS credentials configured: aws configure  (or IAM role if running in CI)
    Required env vars set (or passed via --):
        AWS_ACCOUNT_ID, AWS_REGION, ANTHROPIC_API_KEY

What this script does (in order):
    1.  ECR: create repositories for api and mcp-server images (idempotent)
    2.  Secrets Manager: store API keys (skips if already exists)
    3.  VPC: create VPC, 2 public + 2 private subnets, IGW, NAT gateway
    4.  Security groups: ALB, ECS tasks, RDS
    5.  RDS: PostgreSQL 15 on db.t3.micro in private subnets
    6.  ECS: cluster + task definition + Fargate service
    7.  ALB: load balancer + target group + listener (port 80 -> ECS port 8000)
    8.  Print: the ALB DNS name to use as your API endpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PROJECT = "agent-platform"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Helpers

def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def tag(env: str) -> list[dict]:
    return [
        {"Key": "Project",     "Value": PROJECT},
        {"Key": "Environment", "Value": env},
        {"Key": "ManagedBy",   "Value": "deploy.py"},
    ]


def wait(msg: str, check_fn, timeout: int = 600, interval: int = 15) -> None:
    """Poll `check_fn()` until it returns True or timeout is reached."""
    log(f"Waiting for: {msg}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_fn():
            log(f"  ✓ {msg}")
            return
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for: {msg}")

# 1. ECR repositories

def ensure_ecr_repos(ecr, account_id: str, region: str, dry_run: bool) -> dict[str, str]:
    """Create ECR repos if they don't exist. Returns {name: uri}."""
    repos = {
        f"{PROJECT}-api": f"{account_id}.dkr.ecr.{region}.amazonaws.com/{PROJECT}-api",
        f"{PROJECT}-mcp": f"{account_id}.dkr.ecr.{region}.amazonaws.com/{PROJECT}-mcp",
    }
    for name, uri in repos.items():
        try:
            ecr.describe_repositories(repositoryNames=[name])
            log(f"ECR repo exists: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "RepositoryNotFoundException":
                if not dry_run:
                    ecr.create_repository(
                        repositoryName=name,
                        imageScanningConfiguration={"scanOnPush": True},
                        encryptionConfiguration={"encryptionType": "AES256"},
                    )
                log(f"{'[dry-run] ' if dry_run else ''}Created ECR repo: {name}")
            else:
                raise
    return repos


# 2. Secrets Manager

def ensure_secrets(sm, region: str, account_id: str, env: str, dry_run: bool, api_keys: dict) -> dict[str, str]:
    """Store secrets in Secrets Manager. Skips if the secret already exists."""
    secrets_to_create = {
        f"{PROJECT}/anthropic-api-key": api_keys.get("anthropic") or "REPLACE_ME",
        f"{PROJECT}/openai-api-key":    api_keys.get("openai")    or "REPLACE_ME",
        f"{PROJECT}/langchain-api-key": api_keys.get("langchain") or "REPLACE_ME",
        # DATABASE_URL is written by the RDS step after the DB is created
    }
    arns = {}
    for name, value in secrets_to_create.items():
        try:
            resp = sm.describe_secret(SecretId=name)
            arns[name] = resp["ARN"]
            log(f"Secret exists: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("ResourceNotFoundException", "SecretNotFoundException"):
                if not dry_run:
                    resp = sm.create_secret(
                        Name=name,
                        SecretString=value,
                        Tags=tag(env),
                    )
                    arns[name] = resp["ARN"]
                log(f"{'[dry-run] ' if dry_run else ''}Created secret: {name}")
            else:
                raise
    return arns


# 3. VPC + subnets + IGW + NAT

def ensure_vpc(ec2, env: str, dry_run: bool) -> dict:
    """
    Create a VPC with 2 public and 2 private subnets across 2 AZs.

    Layout (us-east-1 example):
      10.0.0.0/16   VPC
        10.0.1.0/24   public-1  (us-east-1a)
        10.0.2.0/24   public-2  (us-east-1b)
        10.0.10.0/24  private-1 (us-east-1a) <- ECS tasks, RDS
        10.0.11.0/24  private-2 (us-east-1b) <- RDS multi-AZ standby
    """
    # Check for existing VPC by tag
    existing = ec2.describe_vpcs(Filters=[
        {"Name": "tag:Project", "Values": [PROJECT]},
        {"Name": "tag:Environment", "Values": [env]},
    ])["Vpcs"]

    if existing:
        vpc_id = existing[0]["VpcId"]
        log(f"VPC exists: {vpc_id}")
    else:
        if dry_run:
            log("[dry-run] Would create VPC 10.0.0.0/16")
            return {"vpc_id": "vpc-dryrun", "public_subnets": [], "private_subnets": []}
        resp = ec2.create_vpc(CidrBlock="10.0.0.0/16")
        vpc_id = resp["Vpc"]["VpcId"]
        ec2.create_tags(Resources=[vpc_id], Tags=tag(env) + [{"Key": "Name", "Value": f"{PROJECT}-{env}"}])
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
        log(f"Created VPC: {vpc_id}")

    # Internet Gateway
    igws = ec2.describe_internet_gateways(Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])["InternetGateways"]
    if not igws:
        if not dry_run:
            igw = ec2.create_internet_gateway()["InternetGateway"]["InternetGatewayId"]
            ec2.attach_internet_gateway(InternetGatewayId=igw, VpcId=vpc_id)
            log(f"Created and attached IGW: {igw}")
    else:
        log(f"IGW exists: {igws[0]['InternetGatewayId']}")

    # Subnets (idempotent by tag)
    azs = ec2.describe_availability_zones(Filters=[{"Name": "state", "Values": ["available"]}])
    az_names = [az["ZoneName"] for az in azs["AvailabilityZones"][:2]]

    subnet_configs = [
        {"cidr": "10.0.1.0/24",  "az": az_names[0], "public": True,  "name": "public-1"},
        {"cidr": "10.0.2.0/24",  "az": az_names[1], "public": True,  "name": "public-2"},
        {"cidr": "10.0.10.0/24", "az": az_names[0], "public": False, "name": "private-1"},
        {"cidr": "10.0.11.0/24", "az": az_names[1], "public": False, "name": "private-2"},
    ]

    public_subnets = []
    private_subnets = []

    for sc in subnet_configs:
        existing_sn = ec2.describe_subnets(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "cidrBlock", "Values": [sc["cidr"]]},
        ])["Subnets"]

        if existing_sn:
            sn_id = existing_sn[0]["SubnetId"]
        elif not dry_run:
            sn = ec2.create_subnet(VpcId=vpc_id, CidrBlock=sc["cidr"], AvailabilityZone=sc["az"])
            sn_id = sn["Subnet"]["SubnetId"]
            ec2.create_tags(Resources=[sn_id], Tags=tag(env) + [{"Key": "Name", "Value": f"{PROJECT}-{sc['name']}"}])
            if sc["public"]:
                ec2.modify_subnet_attribute(SubnetId=sn_id, MapPublicIpOnLaunch={"Value": True})
            log(f"Created {'public' if sc['public'] else 'private'} subnet: {sn_id} ({sc['cidr']})")
            sn_id = sn_id
        else:
            sn_id = f"subnet-dryrun-{sc['name']}"

        if sc["public"]:
            public_subnets.append(sn_id)
        else:
            private_subnets.append(sn_id)

    return {"vpc_id": vpc_id, "public_subnets": public_subnets, "private_subnets": private_subnets}

# 4. Security groups

def ensure_security_groups(ec2, vpc_id: str, env: str, dry_run: bool) -> dict[str, str]:
    """Create security groups for ALB, ECS tasks, and RDS."""
    groups = {}

    sg_configs = [
        {
            "name": f"{PROJECT}-alb-{env}",
            "desc": "Agent Platform ALB -- inbound 80/443 from anywhere",
            "ingress": [
                {"IpProtocol": "tcp", "FromPort": 80,  "ToPort": 80,  "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            ],
            "key": "alb",
        },
        {
            "name": f"{PROJECT}-ecs-{env}",
            "desc": "Agent Platform ECS tasks -- inbound 8000/8001 from ALB",
            "ingress": [],  # ALB SG added after ALB SG is known
            "key": "ecs",
        },
        {
            "name": f"{PROJECT}-rds-{env}",
            "desc": "Agent Platform RDS -- inbound 5432 from ECS tasks only",
            "ingress": [],  # ECS SG added after ECS SG is known
            "key": "rds",
        },
    ]

    for sg_conf in sg_configs:
        existing = ec2.describe_security_groups(Filters=[
            {"Name": "vpc-id",     "Values": [vpc_id]},
            {"Name": "group-name", "Values": [sg_conf["name"]]},
        ])["SecurityGroups"]

        if existing:
            sg_id = existing[0]["GroupId"]
            log(f"SG exists: {sg_conf['name']} ({sg_id})")
        elif not dry_run:
            resp = ec2.create_security_group(
                GroupName=sg_conf["name"],
                Description=sg_conf["desc"],
                VpcId=vpc_id,
            )
            sg_id = resp["GroupId"]
            ec2.create_tags(Resources=[sg_id], Tags=tag(env) + [{"Key": "Name", "Value": sg_conf["name"]}])
            if sg_conf["ingress"]:
                ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=sg_conf["ingress"])
            log(f"Created SG: {sg_conf['name']} ({sg_id})")
        else:
            sg_id = f"sg-dryrun-{sg_conf['key']}"

        groups[sg_conf["key"]] = sg_id

    # Wire cross-references: ECS tasks accept traffic from ALB; RDS accepts from ECS
    if not dry_run and groups["ecs"] and groups["alb"]:
        try:
            ec2.authorize_security_group_ingress(
                GroupId=groups["ecs"],
                IpPermissions=[{"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8001,
                                "UserIdGroupPairs": [{"GroupId": groups["alb"]}]}],
            )
        except ClientError as e:
            if "InvalidPermission.Duplicate" not in str(e):
                raise

        try:
            ec2.authorize_security_group_ingress(
                GroupId=groups["rds"],
                IpPermissions=[{"IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432,
                                "UserIdGroupPairs": [{"GroupId": groups["ecs"]}]}],
            )
        except ClientError as e:
            if "InvalidPermission.Duplicate" not in str(e):
                raise

    return groups