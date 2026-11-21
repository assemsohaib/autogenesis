"""One full episode through `vf-run`, against the local fake model server.

No network and no credentials: the builder, weak and strong models are all served
by tests/fake_server.py on localhost.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
HERE = Path(__file__).parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "fake_server.py"), str(port)])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("fake server did not start")
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()
    proc.wait(timeout=10)


def run_episode(server: str, out: Path, weak: str) -> tuple[str, dict]:
    vf_run = Path(sys.executable).parent / "vf-run"
    cmd = [
        str(vf_run), "autogenesis", "-m", "fake-builder", "-n", "1", "-r", "1",
        "--client.base-url", server, "--client.api-key-var", "FAKE_API_KEY",
        "--env.weak.model", weak, "--env.strong.model", "fake-strong",
        "--env.audit.seed", "7", "--env.panel.instances", "3", "--env.sandbox.type", "subprocess",
        "-o", str(out),
    ]  # fmt: skip
    env = {**os.environ, "FAKE_API_KEY": "unused"}
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    # vf-run exits 1 when an episode is not `ok`, e.g. after a failed solver call.
    assert done.returncode in (0, 1), done.stdout[-2000:] + done.stderr[-2000:]
    (traces_file,) = out.glob("*/traces.jsonl")
    return done.stdout, json.loads(traces_file.read_text().splitlines()[0])


def test_full_episode(server, tmp_path):
    stdout, episode = run_episode(server, tmp_path, "fake-weak")
    assert "1/1 episodes ok" in stdout, stdout
    builder, *solvers = episode["traces"]
    assert builder["rewards"]["autogenesis"]["score"] == pytest.approx(1.0)
    metrics = builder["metrics"]
    assert metrics["hack_resistance"] == 1.0
    assert (metrics["weak_accuracy"], metrics["strong_accuracy"]) == (0.0, 1.0)
    assert len(solvers) == 6
    assert all(not s["rewards"] for s in solvers)
    assert sorted(s["metrics"]["solved"] for s in solvers) == [0.0] * 3 + [1.0] * 3


def test_flaky_solver_loses_tasks_not_the_episode(server, tmp_path):
    _stdout, episode = run_episode(server, tmp_path, "fake-flaky")
    builder = next(t for t in episode["traces"] if t["agent"]["name"] == "builder")
    weak = [t for t in episode["traces"] if t["agent"]["name"] == "weak"]
    assert builder.get("dropped") is None
    assert sum(t.get("dropped") is not None for t in weak) == 1  # requests 2 of 1..3 fail
    info = builder["info"]["autogenesis"]
    assert info["panel_pairs"] == 2
    assert "lost to solver errors" in info["panel_error"]
    assert builder["metrics"]["strong_accuracy"] == 1.0
    assert builder["rewards"]["autogenesis"]["score"] == pytest.approx(1.0)


def test_dead_solver_drops_the_builder_trace(server, tmp_path):
    _stdout, episode = run_episode(server, tmp_path, "fake-broken")
    builder = next(t for t in episode["traces"] if t["agent"]["name"] == "builder")
    assert "solver panel failed" in (builder.get("dropped") or "")
    assert builder["metrics"]["audit_score"] == pytest.approx(1.0)
