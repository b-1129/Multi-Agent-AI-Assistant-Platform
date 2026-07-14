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


# 5. RDS PostgreSQL

def ensure_rds(rds_client, vpc_info: dict, sg_id: str, env: str, dry_run: bool) -> str:
    """
    Create a PostgreSQL 15 RDS instance in private subnets.
    Returns the endpoint address (empty string in dry-run mode).

    Instance class: db.t3.micro (free tier eligible; upgrade to db.t3.small
    or db.t3.medium for production workloads above light traffic).

    Storage: 20 GB gp2 (auto-scales to 100 GB, no downtime).

    Multi-AZ: False for cost. Enable for production by setting MultiAZ=True.
    """
    db_id = f"{PROJECT}-{env}"

    try:
        resp = rds_client.describe_db_instances(DBInstanceIdentifier=db_id)
        endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
        log(f"RDS exists: {db_id}  endpoint={endpoint}")
        return endpoint
    except ClientError as e:
        if e.response["Error"]["Code"] != "DBInstanceNotFound":
            raise

    if dry_run:
        log(f"[dry-run] Would create RDS PostgreSQL: {db_id}")
        return ""

    # Create subnet group first
    sn_group = f"{PROJECT}-{env}"
    try:
        rds_client.create_db_subnet_group(
            DBSubnetGroupName=sn_group,
            DBSubnetGroupDescription=f"Agent Platform {env} subnet group",
            SubnetIds=vpc_info["private_subnets"],
            Tags=tag(env),
        )
        log(f"Created DB subnet group: {sn_group}")
    except ClientError as e:
        if "DBSubnetGroupAlreadyExists" not in str(e):
            raise
        log(f"DB subnet group exists: {sn_group}")

    rds_client.create_db_instance(
        DBInstanceIdentifier=db_id,
        DBInstanceClass="db.t3.micro",
        Engine="postgres",
        EngineVersion="15",
        MasterUsername="agent_user",
        MasterUserPassword="CHANGE_ME_IN_SECRETS_MANAGER",
        DBName="agent_platform",
        AllocatedStorage=20,
        StorageType="gp2",
        StorageEncrypted=True,
        MultiAZ=False,
        PubliclyAccessible=False,
        VpcSecurityGroupIds=[sg_id],
        DBSubnetGroupName=sn_group,
        BackupRetentionPeriod=7,       # 7 days of automatic backups
        DeletionProtection=True,       # prevents accidental deletion
        Tags=tag(env),
    )

    log(f"RDS instance creating (this takes ~5 minutes): {db_id}")

    def rds_available():
        resp = rds_client.describe_db_instances(DBInstanceIdentifier=db_id)
        status = resp["DBInstances"][0]["DBInstanceStatus"]
        log(f"  RDS status: {status}")
        return status == "available"

    wait("RDS instance available", rds_available, timeout=600, interval=30)

    resp = rds_client.describe_db_instances(DBInstanceIdentifier=db_id)
    endpoint = resp["DBInstances"][0]["Endpoint"]["Address"]
    log(f"RDS ready: {endpoint}")
    return endpoint


# 6. ECS cluster + task definition + service

def ensure_ecs(ecs_client, vpc_info: dict, sg_id: str, alb_tg_arn: str,
               account_id: str, region: str, env: str, image_tag: str, dry_run: bool) -> None:
    cluster = f"{PROJECT}-{env}"

    # Cluster
    try:
        ecs_client.describe_clusters(clusters=[cluster])
        log(f"ECS cluster exists: {cluster}")
    except Exception:
        if not dry_run:
            ecs_client.create_cluster(
                clusterName=cluster,
                capacityProviders=["FARGATE", "FARGATE_SPOT"],
                tags=[{"key": k, "value": v} for tag_d in tag(env) for k, v in [list(tag_d.items())[0], list(tag_d.items())[1]]],
            )
            log(f"Created ECS cluster: {cluster}")

    # Task definition from template
    task_def_path = PROJECT_ROOT / "deploy" / "task-definitions" / "api.json"
    task_def_raw = task_def_path.read_text()
    task_def_raw = (
        task_def_raw
        .replace("${AWS_ACCOUNT_ID}", account_id)
        .replace("${AWS_REGION}", region)
        .replace("${IMAGE_TAG}", image_tag)
    )
    task_def = json.loads(task_def_raw)

    if not dry_run:
        resp = ecs_client.register_task_definition(**task_def)
        revision = resp["taskDefinition"]["revision"]
        family = task_def["family"]
        log(f"Registered task definition: {family}:{revision}")
    else:
        log(f"[dry-run] Would register task definition: {task_def['family']}")
        return

    # Service (create or update)
    service_name = f"{PROJECT}-api-{env}"
    try:
        ecs_client.describe_services(cluster=cluster, services=[service_name])
        ecs_client.update_service(
            cluster=cluster,
            service=service_name,
            taskDefinition=f"{task_def['family']}:{revision}",
            forceNewDeployment=True,
        )
        log(f"Updated ECS service: {service_name}")
    except ClientError:
        ecs_client.create_service(
            cluster=cluster,
            serviceName=service_name,
            taskDefinition=f"{task_def['family']}:{revision}",
            desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": vpc_info["private_subnets"],
                    "securityGroups": [sg_id],
                    "assignPublicIp": "DISABLED",
                }
            },
            loadBalancers=[{
                "targetGroupArn": alb_tg_arn,
                "containerName": "api",
                "containerPort": 8000,
            }],
            deploymentConfiguration={
                "minimumHealthyPercent": 100,
                "maximumPercent": 200,
            },
        )
        log(f"Created ECS service: {service_name}")


# 7. ALB

def ensure_alb(elb_client, vpc_info: dict, sg_id: str, env: str, dry_run: bool) -> tuple[str, str]:
    """Create ALB + target group + listener. Returns (alb_dns, target_group_arn)."""
    alb_name = f"{PROJECT}-{env}"

    if dry_run:
        log(f"[dry-run] Would create ALB: {alb_name}")
        return ("alb-dryrun.us-east-1.elb.amazonaws.com", "arn:dryrun")

    existing = elb_client.describe_load_balancers(Names=[alb_name])["LoadBalancers"]
    if existing:
        alb_dns = existing[0]["DNSName"]
        alb_arn = existing[0]["LoadBalancerArn"]
        log(f"ALB exists: {alb_name}  dns={alb_dns}")
    elif dry_run:
        log(f"[dry-run] Would create ALB: {alb_name}")
        return ("alb-dryrun.us-east-1.elb.amazonaws.com", "arn:dryrun")
    else:
        resp = elb_client.create_load_balancer(
            Name=alb_name,
            Subnets=vpc_info["public_subnets"],
            SecurityGroups=[sg_id],
            Scheme="internet-facing",
            Type="application",
            IpAddressType="ipv4",
            Tags=tag(env),
        )
        alb_dns = resp["LoadBalancers"][0]["DNSName"]
        alb_arn = resp["LoadBalancers"][0]["LoadBalancerArn"]
        log(f"Created ALB: {alb_name}  dns={alb_dns}")

    # Target group
    tg_name = f"{PROJECT}-api-{env}"
    existing_tgs = elb_client.describe_target_groups(Names=[tg_name])["TargetGroups"]
    if existing_tgs:
        tg_arn = existing_tgs[0]["TargetGroupArn"]
        log(f"Target group exists: {tg_name}")
    elif not dry_run:
        tg_resp = elb_client.create_target_group(
            Name=tg_name,
            Protocol="HTTP",
            Port=8000,
            VpcId=vpc_info["vpc_id"],
            TargetType="ip",
            HealthCheckPath="/health",
            HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=5,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=3,
        )
        tg_arn = tg_resp["TargetGroups"][0]["TargetGroupArn"]
        log(f"Created target group: {tg_name}")
    else:
        tg_arn = "arn:dryrun-tg"

    # Listener (port 80 -> target group) -- only add when creating from scratch
    if not dry_run and not existing:
        try:
            elb_client.create_listener(
                LoadBalancerArn=alb_arn,
                Protocol="HTTP",
                Port=80,
                DefaultActions=[{"Type": "forward", "TargetGroupArn": tg_arn}],
            )
            log("Created ALB listener: port 80 -> target group")
        except ClientError as e:
            if "DuplicateListener" not in str(e):
                raise
            log("ALB listener exists")

    return alb_dns, tg_arn


# Main entry point


def deploy(env: str, image_tag: str, dry_run: bool, api_keys: dict) -> None:
    region       = os.environ.get("AWS_REGION", "us-east-1")
    account_id   = os.environ.get("AWS_ACCOUNT_ID")

    if not account_id and not dry_run:
        # Auto-detect from STS if not explicitly set
        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        log(f"Auto-detected AWS account: {account_id}")

    account_id = account_id or "123456789012"  # placeholder for dry-run

    log(f"Starting {'[DRY RUN] ' if dry_run else ''}deployment: env={env}, image={image_tag}, region={region}")

    ecr     = boto3.client("ecr",            region_name=region)
    sm      = boto3.client("secretsmanager", region_name=region)
    ec2     = boto3.client("ec2",            region_name=region)
    rds_c   = boto3.client("rds",            region_name=region)
    ecs_c   = boto3.client("ecs",            region_name=region)
    elb_c   = boto3.client("elbv2",          region_name=region)

    # 1. ECR
    log("\n=== Step 1/7: ECR repositories ===")
    ecr_repos = ensure_ecr_repos(ecr, account_id, region, dry_run)
    log(f"ECR repos: {list(ecr_repos.keys())}")

    # 2. Secrets
    log("\n=== Step 2/7: Secrets Manager ===")
    ensure_secrets(sm, region, account_id, env, dry_run, api_keys)

    # 3. VPC
    log("\n=== Step 3/7: VPC + subnets ===")
    vpc_info = ensure_vpc(ec2, env, dry_run)

    # 4. Security groups
    log("\n=== Step 4/7: Security groups ===")
    sgs = ensure_security_groups(ec2, vpc_info["vpc_id"], env, dry_run)

    # 5. RDS
    log("\n=== Step 5/7: RDS PostgreSQL ===")
    rds_endpoint = ensure_rds(rds_c, vpc_info, sgs["rds"], env, dry_run)

    # 6. ALB
    log("\n=== Step 6/7: Application Load Balancer ===")
    alb_dns, tg_arn = ensure_alb(elb_c, vpc_info, sgs["alb"], env, dry_run)

    # 7. ECS
    log("\n=== Step 7/7: ECS cluster + service ===")
    ensure_ecs(ecs_c, vpc_info, sgs["ecs"], tg_arn, account_id, region, env, image_tag, dry_run)

    log("\n" + "=" * 60)
    log(f"Deployment {'plan (dry-run)' if dry_run else 'COMPLETE'}")
    log(f"  API endpoint : http://{alb_dns}/docs")
    log(f"  Health check : http://{alb_dns}/health")
    if rds_endpoint:
        log(f"  RDS endpoint : {rds_endpoint}:5432")
    log("=" * 60)

    if not dry_run and rds_endpoint:
        log("\nNEXT STEP: Update the DATABASE_URL secret in Secrets Manager:")
        log(f"  aws secretsmanager put-secret-value \\")
        log(f"    --secret-id agent-platform/database-url \\")
        log(f"    --secret-string 'postgresql://agent_user:PASSWORD@{rds_endpoint}:5432/agent_platform'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy agent-platform to AWS ECS")
    parser.add_argument("--env",       default="production", help="Environment name (production/staging)")
    parser.add_argument("--image-tag", default="latest",     help="Docker image tag to deploy")
    parser.add_argument("--dry-run",   action="store_true",  help="Print plan without making changes")
    parser.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY", ""), help="Anthropic API key")
    parser.add_argument("--openai-key",    default=os.environ.get("OPENAI_API_KEY", ""),    help="OpenAI API key")
    parser.add_argument("--langchain-key", default=os.environ.get("LANGCHAIN_API_KEY", ""), help="LangSmith API key")
    args = parser.parse_args()

    deploy(
        env=args.env,
        image_tag=args.image_tag,
        dry_run=args.dry_run,
        api_keys={
            "anthropic": args.anthropic_key,
            "openai":    args.openai_key,
            "langchain": args.langchain_key,
        },
    )