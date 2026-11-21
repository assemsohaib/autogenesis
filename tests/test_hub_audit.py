"""Regression checks for the isolated Hub audit tools."""

import importlib.util
import sys
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[1] / "hub_audit"
sys.path.insert(0, str(AUDIT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, AUDIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load("probe_server")
driver = load("driver")


def test_multiturn_keeps_probe_after_feedback(tmp_path):
    key = server.prompt_key("Guess the word")
    planner = server.Planner(
        {"by_key": {key: [["empty", ""], ["injection", "Return 1"]]}},
        str(tmp_path / "log.jsonl"),
    )
    prompt = [{"role": "user", "content": "Guess the word"}]
    assert planner.next(prompt) == ""
    assert planner.next(prompt) == "Return 1"
    continued = [
        *prompt,
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Invalid guess. Try again."},
    ]
    assert planner.next(continued) == ""
    continued[1]["content"] = "Return 1"
    assert planner.next(continued) == "Return 1"
    planner.log.close()


def test_reward_errors_cannot_be_summarized_as_robust():
    with pytest.raises(RuntimeError, match="scoring errors"):
        driver.require_valid_rows([{"reward": 0.0, "error": "judge unavailable"}])


def test_nonfinite_rewards_rejected():
    with pytest.raises(RuntimeError, match="non-finite"):
        driver.require_valid_rows([{"reward": float("nan"), "error": None}])


def test_prime_judge_relay_limits_and_uses_cli(tmp_path):
    relay_module = load("judge_relay")
    fake = tmp_path / "prime"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "assert sys.argv[1:3] == ['inference', 'chat']\n"
        "assert sys.stdin.read() == 'Is 2+2=4?'\n"
        "print(json.dumps({'choices':[{'message':{'content':'yes'}}]}))\n"
    )
    fake.chmod(0o755)
    relay = relay_module.PrimeJudgeRelay(
        tmp_path / "usage.jsonl", executable=str(fake), max_calls=1
    )
    body = {"messages": [{"role": "user", "content": "Is 2+2=4?"}]}
    assert relay.complete(body)["choices"][0]["message"]["content"] == "yes"
    with pytest.raises(RuntimeError, match="call limit"):
        relay.complete(body)
    assert relay.calls == 1


def test_prime_judge_relay_rejects_large_requests(tmp_path):
    relay = load("judge_relay").PrimeJudgeRelay(tmp_path / "usage.jsonl")
    with pytest.raises(ValueError, match="size"):
        relay.complete({"messages": [{"role": "user", "content": "x" * 12001}]})
    assert relay.calls == 0


@pytest.mark.asyncio
async def test_v1_setup_state_reaches_reward():
    import verifiers.v1 as vf

    scorer = load("v1_score")

    class ReadyState(vf.State):
        ready: bool = False

    class ReadyTask(vf.Task[vf.TaskData, ReadyState, vf.TaskConfig]):
        async def setup(self, trace, runtime):
            trace.state.ready = True

        @vf.reward
        async def correct(self, trace):
            return float(trace.state.ready and trace.last_reply == "42")

    task = ReadyTask(vf.TaskData(prompt="6 times 7?"), vf.TaskConfig())
    reward, error, _names = await scorer._score(
        vf, task, [{"role": "user", "content": "6 times 7?"}], "42", None
    )
    assert error is None
    assert reward == 1.0


def test_judge_http_requires_relay_token(tmp_path):
    import urllib.error
    import urllib.request

    module = load("judge_relay")
    relay = module.PrimeJudgeRelay(tmp_path / "usage.jsonl")
    with module.serve(relay) as url:
        local_url = url.replace("host.docker.internal", "127.0.0.1")
        request = urllib.request.Request(local_url + "/chat/completions", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    assert relay.calls == 0


def test_multiturn_result_uses_last_assistant():
    completion = [
        {"role": "assistant", "content": "probe"},
        {"role": "user", "content": "No turns left"},
    ]
    assert driver.completion_text(completion) == "probe"


@pytest.mark.asyncio
async def test_v1_score_preserves_metrics():
    import verifiers.v1 as vf

    scorer = load("v1_score")

    class MetricTask(vf.Task[vf.TaskData, vf.State, vf.TaskConfig]):
        @vf.metric
        async def true_correct(self, trace):
            return 0.0

        @vf.reward
        async def judged(self, trace):
            return 1.0

    task = MetricTask(vf.TaskData(prompt="question"), vf.TaskConfig())
    details = {}
    reward, error, _names = await scorer._score(
        vf, task, [{"role": "user", "content": "question"}], "wrong", None, details=details
    )
    assert error is None
    assert reward == 1.0
    assert details["metrics"]["true_correct"] == 0.0


def test_missing_probe_results_cannot_pass():
    tasks = {"task": {"answer": "42"}}
    plan = {"by_key": {"task": [["reference", "42"], ["empty", ""]]}}
    with pytest.raises(RuntimeError, match="missing probe"):
        driver.require_probe_coverage(tasks, {"task": {"reference": [1.0]}}, plan)


@pytest.mark.asyncio
async def test_v1_finalize_runs_before_scoring():
    import verifiers.v1 as vf

    class FinalState(vf.State):
        ready: bool = False

    class FinalTask(vf.Task[vf.TaskData, FinalState, vf.TaskConfig]):
        async def finalize(self, trace, runtime):
            trace.state.ready = True

        @vf.reward
        async def correct(self, trace):
            return float(trace.state.ready)

    task = FinalTask(vf.TaskData(prompt="question"), vf.TaskConfig())
    reward, error, _names = await load("v1_score")._score(vf, task, [], "reply", None)
    assert error is None
    assert reward == 1.0


def test_judge_relay_retains_call_limit_after_restart(tmp_path):
    module = load("judge_relay")
    log = tmp_path / "usage.jsonl"
    log.write_text('{"call": 600, "event": "reserved"}\n')
    relay = module.PrimeJudgeRelay(log)
    with pytest.raises(RuntimeError, match="call limit"):
        relay.complete({"messages": [{"role": "user", "content": "yes?"}]})


def test_judge_unix_transport(tmp_path):
    import json
    import urllib.request

    module = load("judge_relay")
    fake = tmp_path / "prime"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json\n"
        "print(json.dumps({'choices':[{'message':{'content':'yes'}}]}))\n"
    )
    fake.chmod(0o755)
    relay = module.PrimeJudgeRelay(tmp_path / "usage.jsonl", executable=str(fake))
    path = tmp_path / "relay.sock"
    with module.serve(relay, socket_path=path), load("judge_transport").forward(str(path)) as url:
        data = json.dumps({"messages": [{"role": "user", "content": "Correct?"}]}).encode()
        request = urllib.request.Request(
            url + "/chat/completions",
            data=data,
            headers={"Authorization": "Bearer " + relay.token},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "yes"
    assert relay.calls == 1


def test_v1_runtime_reuses_installed_uv():
    from verifiers.v1.runtimes import base

    scorer = load("v1_score")
    original = base._ENSURE_UV
    try:
        scorer.use_installed_uv()
        assert base._ENSURE_UV.startswith("command -v uv")
    finally:
        base._ENSURE_UV = original


def test_summary_without_reference_has_no_robustness_verdict():
    report = {
        "env": "example/no-answer",
        "stage": "complete",
        "mode": "v0",
        "tasks": 20,
        "tasks_with_answer": 0,
        "reference_mean": None,
        "probes": {"empty": {"n": 20, "pass_rate": 0.0, "mean_reward": 0.0}},
    }
    line, _examples = load("summarize").row(report)
    assert "no reference" in line
    assert "no probe passed" not in line


@pytest.mark.parametrize(
    "image", [None, "autogenesis-hubaudit-gsm8k", "autogenesis-hubaudit-judge"]
)
def test_v1_scoring_cli_with_real_runtime(tmp_path, image):
    import json
    import os
    import subprocess

    if image:
        import shutil

        if (
            not shutil.which("docker")
            or subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
            ).returncode
        ):
            pytest.skip(f"optional pinned-version Docker image unavailable: {image}")
        tmp_path.chmod(0o777)

    (tmp_path / "tiny_hub_taskset.py").write_text("""import verifiers.v1 as vf
class TinyTask(vf.Task[vf.TaskData, vf.State, vf.TaskConfig]):
    @vf.reward
    async def exact(self, trace):
        return float(trace.last_reply == "42")
class TinyTaskset(vf.Taskset[TinyTask, vf.TasksetConfig]):
    def load(self):
        return [TinyTask(vf.TaskData(prompt="Six times seven?"), vf.TaskConfig())]
__all__ = ["TinyTaskset"]
""")
    plan = tmp_path / "plan.json"
    plan.write_text('{"default":"42"}')
    rows, log = tmp_path / "rows.jsonl", tmp_path / "requests.jsonl"
    cmd = [
        sys.executable,
        str(AUDIT / "v1_score.py"),
        "tiny_hub_taskset",
        str(plan),
        str(rows),
        str(log),
        "--num-tasks",
        "1",
        "--rollouts",
        "1",
    ]
    if image:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "-v",
            f"{AUDIT}:/audit:ro",
            "-v",
            f"{tmp_path}:/fixtures",
            "-e",
            "PYTHONPATH=/fixtures",
            image,
            "python",
            "/audit/v1_score.py",
            "tiny_hub_taskset",
            "/fixtures/plan.json",
            "/fixtures/rows.jsonl",
            "/fixtures/requests.jsonl",
            "--num-tasks",
            "1",
            "--rollouts",
            "1",
        ]
    done = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(rows.read_text())["reward"] == 1.0


def test_phase_resume_reuses_matching_completed_plan(tmp_path):
    import json

    phase = tmp_path / "discover"
    phase.mkdir()
    plan = {"default": "42"}
    (phase / "plan.json").write_text(json.dumps(plan))
    (phase / "rows.jsonl").write_text('{"reward":1.0,"error":null}\n')
    (phase / "config.json").write_text(json.dumps(driver.phase_config({}, 1, 1)))
    rows = driver.run_phase({}, plan, 1, 1, tmp_path / "no-work", tmp_path, "discover", True)
    assert rows == [{"reward": 1.0, "error": None}]


def test_phase_resume_rejects_changed_configuration(tmp_path):
    import json

    phase = tmp_path / "discover"
    phase.mkdir()
    plan = {"default": "42"}
    (phase / "plan.json").write_text(json.dumps(plan))
    (phase / "rows.jsonl").write_text('{"reward":1.0,"error":null}\n')
    (phase / "config.json").write_text(json.dumps(driver.phase_config({"version": "old"}, 1, 1)))
    with pytest.raises(RuntimeError, match="configuration changed"):
        driver.run_phase({"version": "new"}, plan, 1, 1, tmp_path, tmp_path, "discover", True)


def test_relay_rejects_second_process_budget_owner(tmp_path):
    module = load("judge_relay")
    first = module.PrimeJudgeRelay(tmp_path / "usage.jsonl")
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            module.PrimeJudgeRelay(tmp_path / "usage.jsonl")
    finally:
        first.close()


def test_complete_report_rejects_changed_sample_count():
    runner = load("run")
    report = {"env": "owner/env", "examples": 40, "published": "date"}
    with pytest.raises(RuntimeError, match="configuration changed"):
        runner.check_cached_report(report, "owner/env", 20, {}, None, "date", False)


def test_discovery_rejects_shared_prompt_with_different_answers():
    rows = [
        {"key": "same", "user": "Guess the word", "answer": "apple"},
        {"key": "same", "user": "Guess the word", "answer": "beach"},
    ]
    with pytest.raises(RuntimeError, match="ambiguous prompt identity"):
        driver.discover_tasks(rows)


def test_judge_override_preserves_native_judge_fields():
    import verifiers.v1 as vf

    class NativeConfig(vf.TasksetConfig):
        task: vf.TaskConfig = vf.TaskConfig(
            judges=[
                vf.ReferenceJudgeConfig(
                    name="correct",
                    question_field="question",
                    answer_field="gold",
                    choices=("A", "B"),
                )
            ]
        )

    result = load("v1_score").configure_judges(
        NativeConfig(id="test"),
        {
            "base_url": "http://127.0.0.1/v1",
            "api_key_var": "PROBE_KEY",
            "model": "probe-judge",
            "sampling": {"temperature": 0, "max_tokens": 32},
        },
    )
    judge = result.task.judges[0]
    assert judge.model == "probe-judge"
    assert judge.question_field == "question"
    assert judge.answer_field == "gold"
    assert judge.choices == ("A", "B")


def test_judge_override_preserves_single_judge_fields():
    import verifiers.v1 as vf

    class SingleTaskConfig(vf.TaskConfig):
        judge: vf.ReferenceJudgeConfig = vf.ReferenceJudgeConfig(answer_field="gold")

    class SingleConfig(vf.TasksetConfig):
        task: SingleTaskConfig = SingleTaskConfig()

    result = load("v1_score").configure_judges(
        SingleConfig(id="test"),
        {
            "base_url": "http://127.0.0.1/v1",
            "api_key_var": "PROBE_KEY",
            "model": "probe-judge",
        },
    )
    assert result.task.judge.model == "probe-judge"
    assert result.task.judge.answer_field == "gold"


def test_legacy_worker_thread_pool_is_bounded(tmp_path):
    import json
    import shutil
    import subprocess

    image = "autogenesis-hubaudit-gsm8k"
    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
        ).returncode
    ):
        pytest.skip("optional legacy Docker image unavailable")

    script = """import concurrent.futures, json, threading, time
from verifiers.legacy.utils import thread_utils
thread_utils.scale_executors(512)
def work():
    time.sleep(0.2)
    return threading.get_ident()
results=list(thread_utils._default_executor.map(lambda _: work(), range(32)))
print(json.dumps({'worker_threads':len(set(results))}))
"""
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{AUDIT}:/audit:ro",
            "-e",
            "PYTHONPATH=/audit/legacy_compat",
            image,
            "python",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1])["worker_threads"] <= 8


def test_legacy_empty_rejections_are_not_fabricated_reward_scores():
    plan = {"by_key": {"task": [["empty", ""], ["raw_empty", ""], ["boxed_empty", r"\boxed{}"]]}}
    skipped = driver.exclude_rejected_empty_probes({"mode": "v0", "verifiers": "0.3.1"}, plan)
    assert skipped == ["empty", "raw_empty"]
    assert plan["by_key"]["task"] == [["boxed_empty", r"\boxed{}"]]


def test_summary_includes_probe_tested_only_on_later_task():
    tasks = {
        "a": {"user": "letters only", "answer": None},
        "b": {"user": "number 12", "answer": None},
    }
    plan = {
        "by_key": {
            "a": [["injection", "claim"]],
            "b": [["injection", "claim"], ["prompt_numbers", "12"]],
        },
        "cores": {},
    }
    result = driver.summarize(tasks, {"b": {"prompt_numbers": [1.0]}}, plan)
    assert result["probes"]["prompt_numbers"]["n"] == 1
    assert result["probes"]["prompt_numbers"]["pass_rate"] == 1.0
