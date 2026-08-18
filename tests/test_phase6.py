"""
Tests for phase 6: LangSmith tracing config and the eval runner.

Tracing tests verify the configure_tracing() function correctly sets or
avoids setting environment variables, without making real network calls.

Eval runner tests exercise every layer: individual evaluator functions,
the report builder, dataset loading, and the full async pipeline through
the gateway with stubbed graph calls -- same pattern as test_gateway.py.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.tracing import configure_tracing, get_run_metadata
from evals.runner import (
    EvalCase,
    EvalResult,
    EvalScore,
    build_report,
    eval_answer_relevance,
    eval_agent_routing,
    eval_guardrail,
    load_dataset,
    print_report,
    run_eval,
)


def run(coro):
    return asyncio.run(coro)

# Tracing configuration

class TestTracingConfig:

    def test_tracing_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        from app.config import settings
        monkeypatch.setattr(settings, "langchain_tracing_v2", False)
        monkeypatch.setattr(settings, "langchain_api_key", "")

        enabled = configure_tracing()
        assert not enabled
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"

    def test_tracing_enabled_when_both_vars_set(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "langchain_tracing_v2", True)
        monkeypatch.setattr(settings, "langchain_api_key", "ls_fake_key_12345")
        monkeypatch.setattr(settings, "langchain_project", "test-project")

        enabled = configure_tracing()
        assert enabled
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "ls_fake_key_12345"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"

    def test_tracing_disabled_when_api_key_missing(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "langchain_tracing_v2", True)
        monkeypatch.setattr(settings, "langchain_api_key", "")  # no key

        enabled = configure_tracing()
        assert not enabled

    def test_run_metadata_shape(self):
        meta = get_run_metadata("session-1", "gemini-3.5-flash", False)
        assert meta["session_id"] == "session-1"
        assert meta["model_used"] == "gemini-3.5-flash"
        assert meta["blocked"] is False
        assert meta["block_reason"] == "none"

    def test_run_metadata_with_block(self):
        meta = get_run_metadata("s2", "llama-3.3-70b-versatile", True, "input_guardrail")
        assert meta["blocked"] is True
        assert meta["block_reason"] == "input_guardrail"
        assert meta["model_used"] == "llama-3.3-70b-versatile"

# Individual evaluators

class TestEvaluators:

    def test_answer_relevance_hit(self):
        assert eval_answer_relevance("The answer is Paris.", ["Paris"]) == 1.0

    def test_answer_relevance_miss(self):
        assert eval_answer_relevance("I have no idea.", ["Paris"]) == 0.0

    def test_answer_relevance_case_insensitive(self):
        assert eval_answer_relevance("PARIS is the capital.", ["paris"]) == 1.0

    def test_answer_relevance_any_match(self):
        # Any one of the expected strings matching is sufficient
        assert eval_answer_relevance("42 is the answer.", ["41", "42", "43"]) == 1.0

    def test_answer_relevance_not_applicable(self):
        # Empty expected_contains -> None (not applicable)
        assert eval_answer_relevance("anything", []) is None

    def test_agent_routing_correct(self):
        assert eval_agent_routing(["action_agent"], "action_agent") == 1.0

    def test_agent_routing_wrong_agent(self):
        assert eval_agent_routing(["research_agent"], "action_agent") == 0.0

    def test_agent_routing_multi_agent_includes_expected(self):
        # Supervisor routed to two specialists -- expected one is present
        assert eval_agent_routing(["action_agent", "research_agent"], "action_agent") == 1.0

    def test_agent_routing_not_applicable(self):
        assert eval_agent_routing([], "") is None

    def test_guardrail_correct_blocked_and_should_be(self):
        assert eval_guardrail(True, True) == 1.0

    def test_guardrail_correct_not_blocked_and_should_not_be(self):
        assert eval_guardrail(False, False) == 1.0

    def test_guardrail_false_negative(self):
        assert eval_guardrail(False, True) == 0.0

    def test_guardrail_false_positive(self):
        assert eval_guardrail(True, False) == 0.0

# EvalScore

class TestEvalScore:

    def test_all_pass_gives_100_percent(self):
        score = EvalScore(answer_relevance=1.0, agent_routing=1.0, guardrail_correct=1.0)
        assert score.overall == 1.0
        assert score.passed

    def test_one_fail_reduces_overall(self):
        score = EvalScore(answer_relevance=0.0, agent_routing=1.0, guardrail_correct=1.0)
        assert score.overall < 1.0
        assert not score.passed

    def test_none_scores_are_excluded_from_average(self):
        # Only guardrail_correct is applicable (answer_relevance and agent_routing are None)
        score = EvalScore(answer_relevance=None, agent_routing=None, guardrail_correct=1.0)
        assert score.overall == 1.0
        assert score.passed

# Report builder

class TestBuildReport:

    def make_result(self, case_id, category, score_overall, blocked=False):
        s = EvalScore(guardrail_correct=score_overall)
        return EvalResult(case_id, category, "input", "response", blocked, None, [], s, 10.0)

    def test_pass_rate_calculation(self):
        results = [
            self.make_result("c1", "action_agent", 1.0),
            self.make_result("c2", "action_agent", 1.0),
            self.make_result("c3", "security", 0.0),
        ]
        report = build_report(results)
        assert report.total_cases == 3
        assert report.passed == 2
        assert report.failed == 1
        assert abs(report.pass_rate - 0.667) < 0.01

    def test_category_breakdown(self):
        results = [
            self.make_result("c1", "action_agent", 1.0),
            self.make_result("c2", "security", 1.0),
            self.make_result("c3", "security", 0.0),
        ]
        report = build_report(results)
        assert report.category_breakdown["action_agent"]["passed"] == 1
        assert report.category_breakdown["security"]["passed"] == 1
        assert report.category_breakdown["security"]["total"] == 2

    def test_report_is_printable(self, capsys):
        results = [self.make_result("c1", "action_agent", 1.0)]
        report = build_report(results)
        print_report(report)  # should not raise
        captured = capsys.readouterr()
        assert "Pass rate" in captured.out
        assert "100%" in captured.out

# Dataset loading

class TestDatasetLoading:

    def test_loads_real_dataset(self):
        cases = load_dataset("evals/dataset.json")
        assert len(cases) == 20
        assert all(isinstance(c, EvalCase) for c in cases)

    def test_all_cases_have_ids(self):
        cases = load_dataset("evals/dataset.json")
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_security_cases_should_be_blocked(self):
        cases = load_dataset("evals/dataset.json")
        security = [c for c in cases if c.category == "security"]
        assert len(security) >= 6
        assert all(c.should_be_blocked for c in security)

    def test_non_security_cases_should_not_be_blocked(self):
        cases = load_dataset("evals/dataset.json")
        non_security = [c for c in cases if c.category != "security"]
        assert all(not c.should_be_blocked for c in non_security)

    def test_loads_custom_dataset(self, tmp_path):
        custom = [
            {"id": "t1", "category": "test", "input": "hi", "notes": "test",
             "expected_contains": ["hello"], "expected_agent": None, "should_be_blocked": False}
        ]
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(custom))
        cases = load_dataset(str(p))
        assert len(cases) == 1
        assert cases[0].id == "t1"

# Full eval pipeline (with stubbed gateway)

class TestEvalPipeline:

    def test_security_cases_pass_without_llm(self, tmp_path, monkeypatch):
        """Security cases hit guardrails before the LLM -- they pass without
        a real API key, which is the whole point of this test."""
        from app.config import settings
        monkeypatch.setattr(settings, "eval_results_dir", str(tmp_path / "results"))

        import app.vectorstore as vs
        from langchain_core.embeddings import DeterministicFakeEmbedding
        monkeypatch.setattr(vs, "_embeddings", DeterministicFakeEmbedding(size=384))

        report = run(run_eval(categories=["security"]))
        assert report.pass_rate == 1.0, (
            f"Security eval failed: {[(r.case_id, r.score.overall) for r in report.results if not r.score.passed]}"
        )

    def test_report_written_to_disk(self, tmp_path, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "eval_results_dir", str(tmp_path / "results"))

        import app.vectorstore as vs
        from langchain_core.embeddings import DeterministicFakeEmbedding
        monkeypatch.setattr(vs, "_embeddings", DeterministicFakeEmbedding(size=384))

        run(run_eval(categories=["security"]))
        result_files = list(Path(settings.eval_results_dir).glob("eval_*.json"))
        assert len(result_files) == 1
        report_data = json.loads(result_files[0].read_text())
        assert report_data["total_cases"] == 7  # 7 security cases
        assert "results" in report_data
