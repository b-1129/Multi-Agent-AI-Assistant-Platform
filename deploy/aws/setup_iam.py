"""
IAM setup: creates the two IAM roles needed before ECS can run tasks.

  1. ecsTaskExecutionRole  -- allows ECS to pull images from ECR and write
                              logs to CloudWatch. AWS-managed policy.
  2. agent-platform-task-role -- allows the running container to call
                              Secrets Manager and other AWS services at runtime.

Run once before deploy.py:
    python deploy/aws/setup_iam.py

Why two roles?
  ecsTaskExecutionRole: used by the ECS *agent* (the control plane) to start
  the task. It needs ECR pull and CloudWatch write permissions.

  agent-platform-task-role: used by the *container itself* at runtime.
  It needs Secrets Manager read to fetch API keys. Keeping this separate
  from the execution role follows the principle of least privilege -- the
  running app can't modify ECS task definitions or create new resources.
"""

import json
import os

import boto3
from botocore.exceptions import ClientError

PROJECT = "agent-platform"


def log(msg: str) -> None:
    print(f"[iam] {msg}", flush=True)


def ensure_execution_role(iam, account_id: str) -> str:
    role_name = "ecsTaskExecutionRole"
    try:
        resp = iam.get_role(RoleName=role_name)
        log(f"Role exists: {role_name}")
        return resp["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }]
    }

    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="ECS task execution role (pull ECR images, write CloudWatch logs)",
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    )
    log(f"Created role: {role_name}")
    return resp["Role"]["Arn"]


def ensure_task_role(iam, account_id: str, region: str) -> str:
    role_name = f"{PROJECT}-task-role"
    try:
        resp = iam.get_role(RoleName=role_name)
        log(f"Role exists: {role_name}")
        return resp["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }]
    }

    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Runtime role for agent-platform ECS tasks",
    )

    # Allow reading secrets -- scoped to agent-platform/* only
    secrets_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{PROJECT}/*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/ecs/{PROJECT}*",
            },
        ]
    }

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{PROJECT}-runtime-policy",
        PolicyDocument=json.dumps(secrets_policy),
    )

    log(f"Created role: {role_name}")
    return resp["Role"]["Arn"]


if __name__ == "__main__":
    region     = os.environ.get("AWS_REGION", "us-east-1")
    iam        = boto3.client("iam", region_name=region)
    sts        = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    exec_role_arn = ensure_execution_role(iam, account_id)
    task_role_arn = ensure_task_role(iam, account_id, region)

    print(f"\nIAM roles ready:")
    print(f"  ecsTaskExecutionRole  : {exec_role_arn}")
    print(f"  {PROJECT}-task-role : {task_role_arn}")
    print("\nRun next:")
    print("  python deploy/aws/deploy.py --env production")