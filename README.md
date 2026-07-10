# Agent Platform — Complete (Phases 1–7)

A production-grade multi-agent AI platform, built phase by phase so every
technology is understood before the next one is layered on top.

## What's in each phase

| Phase | What you built | Key technologies |
|-------|---------------|-----------------|
| 1 | FastAPI REST API + single LangChain agent | FastAPI, Pydantic, LangChain, Docker |
| 2 | RAG over uploaded documents | Chroma, FastEmbed, LangChain splitters |
| 3 | Multi-agent with LangGraph | LangGraph, Postgres checkpointer, supervisor pattern |
| 4 | MCP tool server + fallback | MCP (FastMCP), langchain-mcp-adapters, async agents |
| 5 | AI security gateway | Presidio PII detection, guardrails, rate limiting, model fallback |
| 6 | LangSmith tracing + eval | LangSmith, @traceable, 20-case eval dataset, local eval runner |
| 7 | AWS deployment + CI/CD | ECS Fargate, ECR, RDS, ALB, Secrets Manager, GitHub Actions |

## Architecture

```
Internet
    │
    ▼
ALB (port 80/443)
    │
    ▼
ECS Fargate Task  ─────────────────────────────────────┐
  ┌──────────────────┐   ┌──────────────────────┐       │
  │  api (port 8000) │   │  mcp-server (8001)   │       │
  │                  │◀─▶│  calculator          │       │
  │  FastAPI         │   │  web_search          │       │
  │  AI Gateway      │   │  search_documents    │       │
  │  guardrails      │   └──────────────────────┘       │
  │  LangGraph       │                                   │
  │  supervisor      │   ┌──────────────────────┐       │
  │  3 specialists   │◀─▶│  RDS PostgreSQL       │       │
  └──────────────────┘   │  (LangGraph sessions) │       │
                          └──────────────────────┘       │
                         Chroma (EFS volume) ────────────┘
```

## Deploy to AWS (first time, ~15 minutes)

```bash
# 0. Prerequisites: AWS CLI configured, Docker running, your API keys in hand.
pip install boto3

# 1. Create IAM roles (once per AWS account)
python deploy/aws/setup_iam.py

# 2. Dry-run to see the plan
python deploy/aws/deploy.py --env production --dry-run

# 3. Build + push Docker images to ECR
python deploy/aws/push_images.py --tag $(git rev-parse --short HEAD)

# 4. Create all infrastructure + deploy
python deploy/aws/deploy.py \
  --env production \
  --image-tag $(git rev-parse --short HEAD) \
  --anthropic-key $ANTHROPIC_API_KEY

# 5. Update the DATABASE_URL secret (printed by deploy.py after RDS is ready)
aws secretsmanager put-secret-value \
  --secret-id agent-platform/database-url \
  --secret-string "postgresql://agent_user:YOUR_PASSWORD@RDS_ENDPOINT:5432/agent_platform"
```

The script prints the ALB DNS name at the end:
```
http://agent-platform-production-123.us-east-1.elb.amazonaws.com/docs
```

## Update after a code change

```bash
git add . && git commit -m "fix: ..."
git push origin main
# -> GitHub Actions CI runs tests, builds images, pushes to ECR, deploys to ECS
```

Or manually:
```bash
TAG=$(git rev-parse --short HEAD)
python deploy/aws/push_images.py --tag $TAG
python deploy/aws/deploy.py --env production --image-tag $TAG
```

## CI/CD pipeline

```
push to any branch
    └─> ci.yml: pytest (85 tests) + security eval (7 cases, no API key needed)
                └─> upload eval report as artifact

push to main
    └─> deploy.yml:
         1. test (same as CI)
         2. build-and-push: docker build --target api && --target mcp-server
                            docker push to ECR (tagged + :latest)
         3. deploy: python deploy/aws/deploy.py --image-tag $SHA
```

**GitHub secrets required:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ACCOUNT_ID`

## Run tests

```bash
pytest
# 118 passed, 1 skipped
```

Phase 7 adds `tests/test_phase7.py` (33 tests):
- ECR create/exists paths via boto3 Stubber
- Secrets Manager create/exists paths
- RDS endpoint extraction
- ALB DNS and target group wiring
- Full dry-run of the 7-step deploy pipeline (monkeypatches all AWS calls)
- Task definition JSON: required fields, secrets in Secrets Manager, health checks, CloudWatch logging
- Dockerfile.prod: 3 stages, correct ports, HEALTHCHECK, non-root USER
- GitHub Actions YAML: job ordering, ECR login, both image targets, secrets usage

## Run locally (no AWS)

```bash
# Terminal 1: MCP server
python -m mcp_server.server

# Terminal 2: API (in-memory sessions, local tool fallback)
uvicorn app.main:app --reload

# Terminal 3: Eval suite
python -m evals.runner
```

## Tear down AWS resources

```bash
python deploy/aws/teardown.py --env production
```

RDS has deletion protection enabled by default — teardown will print the
command to disable it before deleting, preventing accidental data loss.

## What to say in an interview

**"Walk me through the architecture."**
Request hits the ALB → ECS Fargate task (API container + MCP container in
the same task) → gateway applies guardrails + rate limiting → LangGraph
supervisor routes to the right specialist → specialist calls its MCP tool
(calculator, web search, or document search) → output guardrails check the
response → LangSmith traces the whole thing.

**"Why ECS and not Lambda?"**
The agents maintain in-flight async connections to the MCP server and
LangGraph holds conversation state across multiple tool calls within a
single request. Lambda's stateless, cold-start model would make those
connection lifetimes unpredictable. ECS Fargate gives persistent, long-lived
containers that can hold the MCP client connection warm.

**"How do you manage secrets?"**
API keys are stored in AWS Secrets Manager, never in environment variables
or task definitions as plain text. ECS injects them as environment variables
at runtime via the `secrets` field in the task definition, sourced by ARN.
The task role is scoped to read only `agent-platform/*` secrets.

**"How does a new deployment reach production?"**
Push to main → GitHub Actions runs all 118 tests and the security eval →
builds two Docker images from a multi-stage Dockerfile.prod (shared base
layer, separate `api` and `mcp-server` targets) → pushes both to ECR tagged
with the git SHA → runs deploy.py which registers a new ECS task definition
and forces a new deployment → ECS does a rolling replace with
`minimumHealthyPercent: 100` so there's no downtime.

## Project roadmap — COMPLETE

1. ✅ Phase 1: FastAPI + Pydantic + single LangChain agent + Docker
2. ✅ Phase 2: RAG — document upload, chunking, FastEmbed, Chroma
3. ✅ Phase 3: Multi-agent with LangGraph — supervisor + specialists, Postgres memory
4. ✅ Phase 4: MCP — standalone tool server, graceful fallback
5. ✅ Phase 5: AI security — guardrails, PII detection, rate limiting, model fallback
6. ✅ Phase 6: LangSmith tracing + automated agent evaluation (20 cases, 3 evaluators)
7. ✅ Phase 7: AWS deployment — ECS Fargate, ECR, RDS, ALB, Secrets Manager, CI/CD
