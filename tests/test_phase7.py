"""
Tests for phase 7: AWS deployment scripts.

These tests use boto3's Stubber to intercept AWS API calls so nothing
touches real AWS infrastructure. They verify the logic of each deploy
function -- the idempotent "exists already" paths and the "create from
scratch" paths -- without needing credentials or network access.

Structure mirrors the deploy script sections:
    TestECR            -- repository create/exists
    TestSecrets        -- Secrets Manager create/exists/skip
    TestRDS            -- instance exists / RDS endpoint extraction
    TestALB            -- load balancer exists / new create path
    TestDeployDryRun   -- full deploy() call in dry-run mode
    TestTaskDefinition -- task definition JSON structure
    TestDockerfile     -- Dockerfile.prod stage names
    TestGitHubActions  -- CI/CD workflow YAML structure
"""

import json
import sys
from pathlib import Path

import boto3
import pytest
import yaml
from botocore.stub import Stubber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.aws.deploy import (
    ensure_alb,
    ensure_ecr_repos,
    ensure_rds,
    ensure_secrets,
)

# Helpers

def make_client(service: str) -> boto3.client:
    return boto3.client(service, region_name="us-east-1")

# ECR

class TestECR:

    def test_returns_repo_uris_when_both_exist(self):
        ecr = make_client("ecr")
        with Stubber(ecr) as stub:
            for repo in ["agent-platform-api", "agent-platform-mcp"]:
                stub.add_response(
                    "describe_repositories",
                    {"repositories": [{"repositoryName": repo, "repositoryUri": f"123.dkr.ecr.us-east-1.amazonaws.com/{repo}"}]},
                    {"repositoryNames": [repo]},
                )
            repos = ensure_ecr_repos(ecr, "123456789012", "us-east-1", dry_run=False)
            stub.assert_no_pending_responses()

        assert "agent-platform-api" in repos
        assert "agent-platform-mcp" in repos
        assert "123456789012" in repos["agent-platform-api"]

    def test_creates_repos_when_not_found(self):
        ecr = make_client("ecr")
        with Stubber(ecr) as stub:
            for repo in ["agent-platform-api", "agent-platform-mcp"]:
                stub.add_client_error(
                    "describe_repositories",
                    service_error_code="RepositoryNotFoundException",
                    expected_params={"repositoryNames": [repo]},
                )
                stub.add_response(
                    "create_repository",
                    {"repository": {"repositoryUri": f"123.dkr.ecr.us-east-1.amazonaws.com/{repo}"}},
                    {
                        "repositoryName": repo,
                        "imageScanningConfiguration": {"scanOnPush": True},
                        "encryptionConfiguration": {"encryptionType": "AES256"},
                    },
                )
            repos = ensure_ecr_repos(ecr, "123456789012", "us-east-1", dry_run=False)
            stub.assert_no_pending_responses()

        assert len(repos) == 2

    def test_dry_run_skips_create(self):
        ecr = make_client("ecr")
        with Stubber(ecr) as stub:
            for repo in ["agent-platform-api", "agent-platform-mcp"]:
                stub.add_client_error(
                    "describe_repositories",
                    service_error_code="RepositoryNotFoundException",
                    expected_params={"repositoryNames": [repo]},
                )
            # No create_repository calls expected in dry-run
            repos = ensure_ecr_repos(ecr, "123456789012", "us-east-1", dry_run=True)
            stub.assert_no_pending_responses()

        assert len(repos) == 2

# Secrets Manager

class TestSecrets:

    def test_skips_existing_secrets(self):
        sm = make_client("secretsmanager")
        with Stubber(sm) as stub:
            for secret in ["agent-platform/google-api-key", "agent-platform/groq-api-key", "agent-platform/langchain-api-key"]:
                stub.add_response(
                    "describe_secret",
                    {"ARN": f"arn:aws:secretsmanager:us-east-1:123:secret:{secret}-abc123", "Name": secret},
                    {"SecretId": secret},
                )
            arns = ensure_secrets(sm, "us-east-1", "123456789012", "production", dry_run=False, api_keys={"google": "sk-google-test", "groq": "sk-groq-test", "langchain": "ls-test"})
            stub.assert_no_pending_responses()

        assert len(arns) == 3

    def test_creates_missing_secrets(self):
        sm = make_client("secretsmanager")
        with Stubber(sm) as stub:
            for secret in ["agent-platform/google-api-key", "agent-platform/groq-api-key", "agent-platform/langchain-api-key"]:
                stub.add_client_error(
                    "describe_secret",
                    service_error_code="ResourceNotFoundException",
                    expected_params={"SecretId": secret},
                )
                stub.add_response(
                    "create_secret",
                    {"ARN": f"arn:aws:secretsmanager:us-east-1:123:secret:{secret}"},
                )
            arns = ensure_secrets(sm, "us-east-1", "123456789012", "production", dry_run=False, api_keys={"google": "sk-google-test", "groq": "", "langchain": ""})
            stub.assert_no_pending_responses()

        assert len(arns) == 3

# RDS

class TestRDS:

    def test_returns_endpoint_when_instance_exists(self):
        rds = make_client("rds")
        with Stubber(rds) as stub:
            stub.add_response(
                "describe_db_instances",
                {"DBInstances": [{"DBInstanceStatus": "available", "Endpoint": {"Address": "agent-platform-production.abc.us-east-1.rds.amazonaws.com", "Port": 5432}}]},
                {"DBInstanceIdentifier": "agent-platform-production"},
            )
            endpoint = ensure_rds(rds, {"private_subnets": ["sn-1", "sn-2"]}, "sg-rds-123", "production", dry_run=False)
            stub.assert_no_pending_responses()

        assert endpoint == "agent-platform-production.abc.us-east-1.rds.amazonaws.com"

    def test_dry_run_returns_empty_endpoint(self):
        rds = make_client("rds")
        with Stubber(rds) as stub:
            stub.add_client_error(
                "describe_db_instances",
                service_error_code="DBInstanceNotFound",
                expected_params={"DBInstanceIdentifier": "agent-platform-production"},
            )
            endpoint = ensure_rds(rds, {"private_subnets": ["sn-1", "sn-2"]}, "sg-rds-123", "production", dry_run=True)
            stub.assert_no_pending_responses()

        assert endpoint == ""

# ALB

class TestALB:

    def test_returns_dns_when_alb_exists(self):
        elb = make_client("elbv2")
        with Stubber(elb) as stub:
            stub.add_response(
                "describe_load_balancers",
                {"LoadBalancers": [{"LoadBalancerName": "agent-platform-production", "DNSName": "agent-platform-production-123.us-east-1.elb.amazonaws.com", "LoadBalancerArn": "arn:aws:elasticloadbalancing:us-east-1:123:loadbalancer/app/x/y"}]},
                {"Names": ["agent-platform-production"]},
            )
            stub.add_response(
                "describe_target_groups",
                {"TargetGroups": [{"TargetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:123:targetgroup/x/y", "TargetGroupName": "agent-platform-api-production"}]},
                {"Names": ["agent-platform-api-production"]},
            )
            alb_dns, tg_arn = ensure_alb(elb, {"public_subnets": ["sn-1", "sn-2"], "vpc_id": "vpc-123"}, "sg-alb-123", "production", dry_run=False)
            stub.assert_no_pending_responses()

        assert "elb.amazonaws.com" in alb_dns
        assert "targetgroup" in tg_arn

    def test_dry_run_returns_placeholder(self):
        # In dry-run mode ensure_alb returns immediately before touching the
        # boto3 client -- pass a stubbed client to avoid credential errors.
        elb = make_client("elbv2")
        with Stubber(elb):
            alb_dns, tg_arn = ensure_alb(elb, {"public_subnets": [], "vpc_id": "vpc-dry"}, "sg-dry", "production", dry_run=True)
        assert "dryrun" in alb_dns

# Full dry-run deploy (no AWS calls)

class TestDeployDryRun:

    def test_full_dry_run_completes_all_seven_steps(self, capsys, monkeypatch):
        """
        Full deploy() in dry-run mode: stubs every AWS client so no network
        call is made, and confirms all 7 steps are printed.
        """
        import boto3
        from unittest.mock import MagicMock, patch

        # In dry-run mode the functions skip all AWS API calls for VPC and
        # security groups -- stub the ones that DO still run (ECR describe, secrets describe)
        def fake_ecr_describe(**kwargs):
            raise boto3.client("ecr", region_name="us-east-1").exceptions.RepositoryNotFoundException(
                {"Error": {"Code": "RepositoryNotFoundException", "Message": ""}}, "DescribeRepositories"
            )

        import deploy.aws.deploy as deploy_mod

        monkeypatch.setattr(deploy_mod, "ensure_ecr_repos",  lambda *a, **kw: {"api": "uri-api", "mcp": "uri-mcp"})
        monkeypatch.setattr(deploy_mod, "ensure_secrets",     lambda *a, **kw: {})
        monkeypatch.setattr(deploy_mod, "ensure_vpc",         lambda *a, **kw: {"vpc_id": "vpc-dry", "public_subnets": ["sn-pub"], "private_subnets": ["sn-priv"]})
        monkeypatch.setattr(deploy_mod, "ensure_security_groups", lambda *a, **kw: {"alb": "sg-alb", "ecs": "sg-ecs", "rds": "sg-rds"})
        monkeypatch.setattr(deploy_mod, "ensure_rds",         lambda *a, **kw: "")
        monkeypatch.setattr(deploy_mod, "ensure_alb",         lambda *a, **kw: ("dry-alb.elb.amazonaws.com", "arn:dry-tg"))
        monkeypatch.setattr(deploy_mod, "ensure_ecs",         lambda *a, **kw: None)

        deploy_mod.deploy(env="production", image_tag="abc1234", dry_run=True, api_keys={})

        captured = capsys.readouterr()
        for step in ["Step 1/7", "Step 2/7", "Step 3/7", "Step 4/7", "Step 5/7", "Step 6/7", "Step 7/7"]:
            assert step in captured.out, f"Missing: {step}"
        assert "dry-alb.elb.amazonaws.com" in captured.out

# Task definition JSON structure

class TestTaskDefinition:

    def test_task_def_has_required_fields(self):
        raw = (PROJECT_ROOT / "deploy" / "task-definitions" / "api.json").read_text()
        raw = raw.replace("${AWS_ACCOUNT_ID}", "123").replace("${AWS_REGION}", "us-east-1").replace("${IMAGE_TAG}", "latest")
        td = json.loads(raw)

        assert td["family"] == "agent-platform-api"
        assert td["networkMode"] == "awsvpc"
        assert "FARGATE" in td["requiresCompatibilities"]
        assert int(td["cpu"]) >= 256
        assert int(td["memory"]) >= 512

    def test_task_def_has_api_and_mcp_containers(self):
        raw = (PROJECT_ROOT / "deploy" / "task-definitions" / "api.json").read_text()
        raw = raw.replace("${AWS_ACCOUNT_ID}", "123").replace("${AWS_REGION}", "us-east-1").replace("${IMAGE_TAG}", "latest")
        td = json.loads(raw)

        names = [c["name"] for c in td["containerDefinitions"]]
        assert "api" in names
        assert "mcp-server" in names

    def test_api_container_has_secrets_from_secrets_manager(self):
        raw = (PROJECT_ROOT / "deploy" / "task-definitions" / "api.json").read_text()
        raw = raw.replace("${AWS_ACCOUNT_ID}", "123").replace("${AWS_REGION}", "us-east-1").replace("${IMAGE_TAG}", "latest")
        td = json.loads(raw)

        api = next(c for c in td["containerDefinitions"] if c["name"] == "api")
        secret_names = [s["name"] for s in api["secrets"]]
        assert "GOOGLE_API_KEY" in secret_names
        assert "DATABASE_URL" in secret_names
        # All secrets must reference Secrets Manager ARNs, not plain values
        for s in api["secrets"]:
            assert s["valueFrom"].startswith("arn:aws:secretsmanager")

    def test_api_container_has_health_check(self):
        raw = (PROJECT_ROOT / "deploy" / "task-definitions" / "api.json").read_text()
        raw = raw.replace("${AWS_ACCOUNT_ID}", "123").replace("${AWS_REGION}", "us-east-1").replace("${IMAGE_TAG}", "latest")
        td = json.loads(raw)

        api = next(c for c in td["containerDefinitions"] if c["name"] == "api")
        hc = api["healthCheck"]
        assert "/health" in " ".join(hc["command"])
        assert hc["interval"] >= 10

    def test_containers_have_cloudwatch_logging(self):
        raw = (PROJECT_ROOT / "deploy" / "task-definitions" / "api.json").read_text()
        raw = raw.replace("${AWS_ACCOUNT_ID}", "123").replace("${AWS_REGION}", "us-east-1").replace("${IMAGE_TAG}", "latest")
        td = json.loads(raw)

        for container in td["containerDefinitions"]:
            log_cfg = container["logConfiguration"]
            assert log_cfg["logDriver"] == "awslogs"
            assert "awslogs-group" in log_cfg["options"]

# Dockerfile.prod structure

class TestDockerfileProd:

    def _get_stages(self) -> list[str]:
        content = (PROJECT_ROOT / "Dockerfile.prod").read_text()
        stages = []
        for line in content.splitlines():
            parts = line.strip().split()
            if parts and parts[0] == "FROM" and "AS" in parts:
                stages.append(parts[-1])
        return stages

    def test_has_three_stages(self):
        stages = self._get_stages()
        assert len(stages) == 3

    def test_stage_names(self):
        stages = self._get_stages()
        assert stages == ["base", "api", "mcp-server"]

    def test_api_stage_exposes_correct_port(self):
        content = (PROJECT_ROOT / "Dockerfile.prod").read_text()
        api_section = content.split("AS api")[-1].split("AS mcp-server")[0]
        assert "EXPOSE 8000" in api_section

    def test_mcp_stage_exposes_correct_port(self):
        content = (PROJECT_ROOT / "Dockerfile.prod").read_text()
        mcp_section = content.split("AS mcp-server")[-1]
        assert "EXPOSE 8001" in mcp_section

    def test_health_checks_present(self):
        content = (PROJECT_ROOT / "Dockerfile.prod").read_text()
        assert content.count("HEALTHCHECK") == 2  # one per service stage

    def test_runs_as_non_root(self):
        content = (PROJECT_ROOT / "Dockerfile.prod").read_text()
        assert "USER appuser" in content

# GitHub Actions workflows

class TestGitHubActionsWorkflows:

    def _load(self, filename: str) -> dict:
        content = (PROJECT_ROOT / ".github" / "workflows" / filename).read_text()
        d = yaml.safe_load(content)
        # pyyaml parses the YAML 'on' key as Python bool True
        # Normalize it so tests can use 'on' uniformly.
        if True in d:
            d["on"] = d.pop(True)
        return d

    def test_ci_workflow_is_valid_yaml(self):
        d = self._load("ci.yml")
        assert "jobs" in d
        assert "test" in d["jobs"]

    def test_ci_runs_on_all_branches(self):
        d = self._load("ci.yml")
        assert "**" in d["on"]["push"]["branches"]

    def test_ci_runs_pytest(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "pytest" in content

    def test_ci_runs_security_evals(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "evals.runner" in content
        assert "security" in content

    def test_ci_uploads_eval_artifacts(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "upload-artifact" in content

    def test_cd_workflow_has_three_jobs(self):
        d = self._load("deploy.yml")
        assert len(d["jobs"]) == 3

    def test_cd_job_ordering(self):
        d = self._load("deploy.yml")
        assert "test" in d["jobs"]["build-and-push"]["needs"]
        assert "build-and-push" in d["jobs"]["deploy"]["needs"]

    def test_cd_uses_ecr_login_action(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        assert "amazon-ecr-login" in content

    def test_cd_deploys_both_images(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        assert "Dockerfile.prod" in content
        assert "--target api" in content
        assert "--target mcp-server" in content

    def test_cd_triggers_on_main_push(self):
        d = self._load("deploy.yml")
        assert "main" in d["on"]["push"]["branches"]

    def test_cd_supports_manual_trigger(self):
        d = self._load("deploy.yml")
        assert "workflow_dispatch" in d["on"]

    def test_cd_uses_aws_secrets_for_credentials(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        assert "AWS_ACCESS_KEY_ID" in content
        assert "AWS_SECRET_ACCESS_KEY" in content
        assert "secrets." in content
