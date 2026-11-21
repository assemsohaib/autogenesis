"""The audit against real sandbox processes (the verifiers subprocess runtime)."""

import random

import pytest
from verifiers.v1.configs.runtime import SubprocessConfig
from verifiers.v1.runtimes import provision_runtime

import fixtures as fx
from autogenesis import AuditConfig, run_audit
from autogenesis.sandbox import SandboxExecutor


@pytest.fixture
async def executor():
    async with provision_runtime(SubprocessConfig()) as runtime:
        yield SandboxExecutor(runtime, call_timeout=0.5, sandbox_timeout=60)


async def audit(executor, code_or_reply, *, raw=False, **overrides):
    config = AuditConfig(**{"num_tasks": 12, "seed": 1, **overrides})
    reply = code_or_reply if raw else fx.reply(code_or_reply)
    return await run_audit(reply, executor, config, random.Random(1))


async def test_good_env_scores_high(executor):
    report = await audit(executor, fx.GOOD_ENV)
    assert report.complete, report.error
    assert report.valid == report.determinism == report.solvability == 1.0
    assert report.diversity == 1.0
    assert report.leak == 0.0
    assert report.hack_resistance == 1.0, report.probe_pass_rates
    assert report.score == pytest.approx(1.0)
    assert len(report.tasks) == 12 and all(report.solved)


async def test_substring_verifier_is_hacked(executor):
    report = await audit(executor, fx.SUBSTRING_VERIFY)
    assert report.complete
    assert report.solvability == 1.0
    assert report.probe_pass_rates["shotgun"] == 1.0
    assert report.hack_resistance == 0.0
    assert report.score == 0.0


async def test_yes_no_answers_are_guessable(executor):
    report = await audit(executor, fx.YES_NO_ENV)
    assert report.complete
    assert report.worst_probe in {"mode_answer", "yes", "shotgun"}
    assert 0.0 < report.hack_resistance <= 0.75
    assert report.score < 0.75


async def test_leak_zeroes_score(executor):
    report = await audit(executor, fx.LEAKY_ENV)
    assert report.leak == 1.0
    assert report.score == 0.0


async def test_nondeterminism_is_measured(executor):
    report = await audit(executor, fx.NONDETERMINISTIC_ENV)
    assert report.determinism < 0.5
    assert report.score < 0.5


async def test_constant_tasks_lose_diversity(executor):
    report = await audit(executor, fx.CONSTANT_ENV)
    assert report.diversity == pytest.approx(1 / 12)
    # The mode answer now solves every task, so the verifier is also hackable.
    assert report.hack_resistance == 0.0


async def test_unsolvable_reference(executor):
    report = await audit(executor, fx.UNSOLVABLE_ENV)
    assert report.complete
    assert report.solvability == 0.0
    assert report.score == 0.0
    assert "rejects every reference answer" in report.error


async def test_crashing_generate_stops_audit(executor):
    report = await audit(executor, fx.CRASHING_GENERATE)
    assert report.stage == "generate"
    assert "boom" in report.error


async def test_looping_verify_times_out_per_call(executor):
    report = await audit(executor, fx.LOOPING_VERIFY, num_tasks=4)
    assert report.complete
    assert report.solvability == 0.0
    assert report.score == 0.0


async def test_system_exit_in_verify_is_contained(executor):
    report = await audit(executor, fx.EXITING_VERIFY)
    assert report.complete
    assert report.solvability == 0.0


async def test_out_of_range_score_is_rejected(executor):
    report = await audit(executor, fx.OUT_OF_RANGE_VERIFY)
    assert report.solvability == 0.0


async def test_policy_violation_never_runs(executor):
    report = await audit(executor, fx.IMPORTS_OS)
    assert report.stage == "policy"
    assert report.score == 0.0


async def test_reply_without_code(executor):
    report = await audit(executor, "I refuse to write code.", raw=True)
    assert report.stage == "extract"


async def test_runner_blocks_imports_even_without_policy(executor):
    """Second layer: the runner's import hook stops what the static policy missed."""
    result = await executor("generate", code=fx.IMPORTS_OS, seeds=[1])
    assert "fatal" in result
    assert "not allowed" in result["fatal"]


async def test_runner_hides_host_environment(executor, monkeypatch):
    monkeypatch.setenv("AUTOGENESIS_SECRET", "s3cret")
    code = fx.GOOD_ENV.replace(
        "import re",
        "import re\nimport os as _os",
    ).replace('return {"prompt": prompt', 'return {"env": dict(_os.environ), "prompt": prompt')
    # `os` is blocked at load, so the secret can never reach a task.
    result = await executor("generate", code=code, seeds=[1])
    assert "fatal" in result and "s3cret" not in str(result)


async def test_executor_reports_dead_sandbox(executor):
    executor.python = "/nonexistent/python"
    result = await executor("generate", code=fx.GOOD_ENV, seeds=[1])
    assert "fatal" in result
