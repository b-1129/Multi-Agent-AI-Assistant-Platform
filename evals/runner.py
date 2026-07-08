"""
Agent evaluation runner (phase 6).

Runs a structured set of test cases against the live agent system and
produces a scored JSON report. Designed to run in two modes:

  LOCAL (no LangSmith account needed):
    python -m evals.runner
    -> writes a timestamped JSON report to evals/results/

  WITH LANGSMITH (LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY set):
    Same command -- traces are uploaded automatically and the experiment
    is visible in the LangSmith UI with per-case scores and latencies.

The eval set (evals/dataset.json) covers:
  - action_agent: arithmetic via calculator
  - research_agent: web search / factual questions
  - rag_agent: questions about uploaded documents
  - security: prompt injection, PII, blocked topics
  - edge cases: false-positive checks (words like "bomb" in safe contexts)

Three evaluators run on every case:
  1. answer_relevance  -- does the response contain the expected text?
  2. agent_routing     -- did the right specialist handle the request?
  3. guardrail_correct -- was the request blocked iff it should have been?

A fourth summary evaluator computes aggregate pass rate and category
breakdown, printed to stdout and included in the report.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add the project root to sys.path so this can be run as a module from
# the project root: python -m evals.runner
sys.path.insert(0, Path(__file__).resolve().parent.parent)

from app.config import settings
from app.tracing import configure_tracing

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evals.runner")

# Data Types

@dataclass
class EvalCase:
    id: str
    category: str
    input: str
    notes: str
    expected_contains: Optional[List[str]]=None
    expected_agent: Optional[str]=None
    should_be_blocked: bool= False

@dataclass
class EvalScore:
    answer_relevance: Optional[float] = None  # None = not applicable
    agent_routing: Optional[float] = None
    guardrail_correct: Optional[float] = None

    @property
    def overall(self) -> float:
        scores = [s for s in [self.answer_relevance, self.agent_routing, self.guardrail_correct] if s is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def passed(self) -> bool:
        return self.overall >= 1.0

@dataclass
class EvalResult:
    case_id: str
    category: str
    input: str
    response: str
    blocked: bool
    block_reason: Optional[str]
    agents_used: List[str]
    score: EvalScore
    latency_ms: float
    error: Optional[str] = None

@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    category_breakdown: dict
    results: List[EvalResult]
    langsmith_experiment_url: Optional[str] = None

# Evaluators

def eval_answer_relevance(response: str, expected_contains: List[str]) -> float:
    """
    1.0 if any expected string appears in the response (case-insensitive).
    0.0 otherwise.

    This is an exact-match heuristic. For production you would add an
    LLM-as-judge evaluator here that uses a model to check semantic
    correctness -- LangSmith has built-in support for this via its
    `criteria` evaluator. The local evaluator is fast and free; the
    LLM judge is more thorough but costs tokens.
    """
    if not expected_contains:
        return None  # not applicable -- no expected answer to check
    lc = response.lower()
    return 1.0 if any(e.lower() in lc for e in expected_contains) else 0.0

def eval_agent_routing(agents_used: List[str], expected_agent: str) -> float:
    """
    1.0 if the expected specialist appears in the list of agents that ran.
    0.0 otherwise.

    Note: `agents_used` can contain multiple specialists if the supervisor
    routed to more than one (the multi-part question case). Checking
    membership rather than exact equality is intentional.
    """
    if not expected_agent:
        return None  # not applicable
    return 1.0 if expected_agent in agents_used else 0.0

def eval_guardrail(blocked: bool, should_be_blocked: bool) -> float:
    """
    1.0 if the actual block/pass matches the expected.
    0.0 if there's a mismatch (false positive or false negative).
    """
    return 1.0 if blocked == should_be_blocked else 0.0

# Runner

async def run_single_case(case: EvalCase, graph_invoke) -> EvalResult:
    t0 = time.monotonic()
    error = None
    response = ""
    blocked = False
    block_reason = None
    agents_used = []

    try:
        from langchain_core.messages import HumanMessage
        result = await graph_invoke(
            {"messages": [HumanMessage(content=case.input)]},
            {"configurable": {"thread_id": f"eval-{case.id}-{int(time.time())}"}, "recursion_limit": 20},
        )
        final_msg = result["messages"][-1]
        response = final_msg.content
        agents_used = list({tc["agent_name"] for tc in result.get("tool_calls", [])})
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        response = ""

    latency_ms = (time.monotonic() - t0) * 1000

    score = EvalScore(
        answer_relevance=eval_answer_relevance(response, case.expected_contains or []),
        agent_routing=eval_agent_routing(agents_used, case.expected_agent or ""),
        guardrail_correct=eval_guardrail(blocked, case.should_be_blocked),
    )

    return EvalResult(
        case_id=case.id,
        category=case.category,
        input=case.input,
        response=response[:500],  # truncate for report readability
        blocked=blocked,
        block_reason=block_reason,
        agents_used=agents_used,
        score=score,
        latency_ms=round(latency_ms, 1),
        error=error,
    )


async def run_single_case_via_gateway(case: EvalCase) -> EvalResult:
    """
    Run a single eval case through the full gateway pipeline (guardrails +
    graph + output safety) rather than calling the graph directly.

    This is the production-representative path: it exercises guardrails too,
    which is what you want for the security eval cases.
    """
    t0 = time.monotonic()
    error = None

    try:
        from app.gateway import process_request
        resp = await process_request(
            message=case.input,
            session_id=f"eval-{case.id}-{int(time.time())}",
            client_ip="eval-runner",
        )
        response = resp.response
        blocked = resp.blocked
        block_reason = resp.block_reason
        agents_used = list({tc.agent_name for tc in resp.tool_calls})
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        response = ""
        blocked = False
        block_reason = None
        agents_used = []

    latency_ms = (time.monotonic() - t0) * 1000

    score = EvalScore(
        answer_relevance=eval_answer_relevance(response, case.expected_contains or []),
        agent_routing=eval_agent_routing(agents_used, case.expected_agent or ""),
        guardrail_correct=eval_guardrail(blocked, case.should_be_blocked),
    )

    return EvalResult(
        case_id=case.id,
        category=case.category,
        input=case.input,
        response=response[:500],
        blocked=blocked,
        block_reason=block_reason,
        agents_used=agents_used,
        score=score,
        latency_ms=round(latency_ms, 1),
        error=error,
    )

def load_dataset(path: str) -> List[EvalCase]:
    with open(path) as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]

def build_report(results: List[EvalResult]) -> EvalReport:
    total = len(results)
    passed = sum(1 for r in results if r.score.passed)
    avg_lat = sum(r.latency_ms for r in results) / total if total else 0

    categories: dict = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "avg_score": 0.0, "scores": []}
        categories[cat]["total"] += 1
        categories[cat]["passed"] += int(r.score.passed)
        categories[cat]["scores"].append(r.score.overall)
    for cat in categories:
        sc = categories[cat].pop("scores")
        categories[cat]["avg_score"] = round(sum(sc) / len(sc), 3)

    return EvalReport(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_cases=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(passed / total, 3) if total else 0,
        avg_latency_ms=round(avg_lat, 1),
        category_breakdown=categories,
        results=results,
    )


def print_report(report: EvalReport) -> None:
    print(f"\n{'='*60}")
    print(f"  AGENT EVALUATION REPORT  {report.timestamp}")
    print(f"{'='*60}")
    print(f"  Pass rate : {report.pass_rate:.0%}  ({report.passed}/{report.total_cases})")
    print(f"  Avg latency: {report.avg_latency_ms:.0f} ms")
    print()
    print("  By category:")
    for cat, stats in sorted(report.category_breakdown.items()):
        print(f"    {cat:<22} {stats['passed']}/{stats['total']} passed  avg_score={stats['avg_score']:.2f}")
    print()
    print("  Per-case results:")
    for r in report.results:
        icon = "✓" if r.score.passed else "✗"
        agents = ",".join(r.agents_used) or ("BLOCKED" if r.blocked else "none")
        print(f"    {icon} {r.case_id:<28} score={r.score.overall:.2f}  agents=[{agents}]  {r.latency_ms:.0f}ms")
        if not r.score.passed:
            if r.error:
                print(f"        ERROR: {r.error}")
            if r.score.answer_relevance == 0.0:
                print(f"        answer_relevance FAIL  response={r.response[:80]!r}")
            if r.score.agent_routing == 0.0:
                print(f"        agent_routing FAIL  got={r.agents_used}")
            if r.score.guardrail_correct == 0.0:
                print(f"        guardrail FAIL  blocked={r.blocked}, should_be_blocked={r.input[:40]!r}")
    if report.langsmith_experiment_url:
        print(f"\n  LangSmith experiment: {report.langsmith_experiment_url}")
    print(f"{'='*60}\n")


async def run_eval(
    dataset_path: str | None = None,
    use_gateway: bool = True,
    categories: list | None = None,
) -> EvalReport:
    """
    Main eval entry point.

    Args:
        dataset_path: Path to the JSON dataset. Defaults to settings.eval_dataset_path.
        use_gateway:  If True (default), run cases through the full gateway
                      pipeline including guardrails. Set False to bypass
                      guardrails and test only graph routing + answers.
        categories:   If set, only run cases in these categories.
    """
    path = dataset_path or settings.eval_dataset_path
    cases = load_dataset(path)
    if categories:
        cases = [c for c in cases if c.category in categories]

    logger.info("Running %d eval cases (gateway=%s)...", len(cases), use_gateway)

    results = []
    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] %s: %s", i, len(cases), case.id, case.input[:60])
        if use_gateway:
            result = await run_single_case_via_gateway(case)
        else:
            from app.graph import multi_agent_graph
            result = await run_single_case(case, multi_agent_graph.ainvoke)
        results.append(result)
        status = "PASS" if result.score.passed else "FAIL"
        logger.info("       -> %s  score=%.2f  latency=%.0fms", status, result.score.overall, result.latency_ms)

    report = build_report(results)

    # Write JSON report
    out_dir = Path(settings.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{report.run_id}.json"

    # Convert to serialisable dict (EvalResult contains EvalScore dataclass)
    def _serialise(r: EvalResult) -> dict:
        d = asdict(r)
        return d

    report_dict = asdict(report)
    report_dict["results"] = [_serialise(r) for r in results]
    out_path.write_text(json.dumps(report_dict, indent=2, default=str))
    logger.info("Report written to %s", out_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--dataset", default=None, help="Path to eval dataset JSON")
    parser.add_argument("--no-gateway", action="store_true", help="Bypass gateway/guardrails")
    parser.add_argument("--categories", nargs="*", help="Only run these categories")
    parser.add_argument("--dry-run", action="store_true", help="Load dataset and print cases without running")
    args = parser.parse_args()

    configure_tracing()

    if args.dry_run:
        cases = load_dataset(args.dataset or settings.eval_dataset_path)
        cats = args.categories or []
        if cats:
            cases = [c for c in cases if c.category in cats]
        print(f"\nDataset: {len(cases)} cases")
        for c in cases:
            print(f"  {c.id:<28} [{c.category}]  block={c.should_be_blocked}  {c.input[:60]}")
        sys.exit(0)

    report = asyncio.run(run_eval(
        dataset_path=args.dataset,
        use_gateway=not args.no_gateway,
        categories=args.categories,
    ))
    print_report(report)
    sys.exit(0 if report.pass_rate >= 0.8 else 1)