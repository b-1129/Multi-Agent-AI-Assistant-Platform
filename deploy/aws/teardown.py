"""
Teardown script: destroys all agent-platform AWS resources.

Runs in REVERSE order of deploy.py and respects AWS dependencies
(you can't delete a VPC until its subnets and security groups are gone).

Usage:
    python deploy/aws/teardown.py --env production

Safety:
    RDS has DeletionProtection=True -- you must disable it manually before
    teardown can delete it, which is intentional to prevent accidents.

    This script will print a warning and skip RDS if deletion protection
    is still enabled. You can disable it with:
        aws rds modify-db-instance \
            --db-instance-identifier agent-platform-production \
            --no-deletion-protection \
            --apply-immediately
"""

import argparse
import os
import time

import boto3
from botocore.exceptions import ClientError

PROJECT = "agent-platform"


def log(msg: str) -> None:
    print(f"[teardown] {msg}", flush=True)


def teardown(env: str, region: str, dry_run: bool) -> None:
    ec2   = boto3.client("ec2",   region_name=region)
    rds   = boto3.client("rds",   region_name=region)
    ecs   = boto3.client("ecs",   region_name=region)
    elb   = boto3.client("elbv2", region_name=region)

    cluster = f"{PROJECT}-{env}"

    # 1. ECS service (scale to 0 first, then delete)
    log("Step 1: ECS service")
    service_name = f"{PROJECT}-api-{env}"
    try:
        ecs.update_service(cluster=cluster, service=service_name, desiredCount=0)
        time.sleep(15)
        ecs.delete_service(cluster=cluster, service=service_name, force=True)
        log(f"  Deleted ECS service: {service_name}")
    except ClientError as e:
        log(f"  Skip ECS service ({e.response['Error']['Code']})")

    try:
        ecs.delete_cluster(cluster=cluster)
        log(f"  Deleted ECS cluster: {cluster}")
    except ClientError as e:
        log(f"  Skip ECS cluster ({e.response['Error']['Code']})")

    # 2. ALB + target group
    log("Step 2: ALB")
    alb_name = f"{PROJECT}-{env}"
    try:
        albs = elb.describe_load_balancers(Names=[alb_name])["LoadBalancers"]
        for alb in albs:
            listeners = elb.describe_listeners(LoadBalancerArn=alb["LoadBalancerArn"])["Listeners"]
            for listener in listeners:
                elb.delete_listener(ListenerArn=listener["ListenerArn"])
            elb.delete_load_balancer(LoadBalancerArn=alb["LoadBalancerArn"])
            log(f"  Deleted ALB: {alb_name}")
    except ClientError as e:
        log(f"  Skip ALB ({e.response['Error']['Code']})")

    try:
        tg_name = f"{PROJECT}-api-{env}"
        tgs = elb.describe_target_groups(Names=[tg_name])["TargetGroups"]
        for tg in tgs:
            elb.delete_target_group(TargetGroupArn=tg["TargetGroupArn"])
            log(f"  Deleted target group: {tg_name}")
    except ClientError:
        pass

    # 3. RDS
    log("Step 3: RDS")
    db_id = f"{PROJECT}-{env}"
    try:
        resp = rds.describe_db_instances(DBInstanceIdentifier=db_id)
        db = resp["DBInstances"][0]
        if db.get("DeletionProtection"):
            log(f"  SKIP RDS: deletion protection is enabled.")
            log(f"  To delete, first run:")
            log(f"    aws rds modify-db-instance --db-instance-identifier {db_id} --no-deletion-protection --apply-immediately")
        else:
            rds.delete_db_instance(
                DBInstanceIdentifier=db_id,
                SkipFinalSnapshot=False,
                FinalDBSnapshotIdentifier=f"{db_id}-final-snapshot",
            )
            log(f"  Deleting RDS: {db_id} (final snapshot will be created)")
    except ClientError as e:
        log(f"  Skip RDS ({e.response['Error']['Code']})")

    # 4. Security groups + subnets + VPC (leave for last -- dependencies)
    log("Step 4: VPC resources (security groups, subnets, VPC)")
    log("  (Manual step recommended -- use AWS Console to avoid dependency errors)")
    log("  Or run: aws ec2 describe-vpcs --filters Name=tag:Project,Values=agent-platform")

    log("\nTeardown complete (check AWS Console for any remaining resources).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tear down agent-platform AWS resources")
    parser.add_argument("--env",      default="production")
    parser.add_argument("--region",   default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        confirm = input(f"This will DELETE all {args.env} resources. Type '{args.env}' to confirm: ")
        if confirm != args.env:
            print("Aborted.")
            exit(1)

    teardown(args.env, args.region, args.dry_run)