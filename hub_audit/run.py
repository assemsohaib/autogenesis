"""Run the Hub audit on the host: one Docker container per environment.

Usage:
  python hub_audit/run.py --out runs/hub_audit owner/name [owner/name ...]
  python hub_audit/run.py --out runs/hub_audit --list targets.txt

Hub environment code is third-party code. It runs only inside the container
(`autogenesis-hubaudit`, built from hub_audit/Dockerfile) as a non-root user, with
memory and CPU limits and no host credentials. The container needs the network to
install the package and download its dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import re
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "autogenesis-hubaudit"
_INSTALL = re.compile(r"uv pip install (\S+) --extra-index-url (\S+)")


def check_cached_report(report, env_id, examples, config, env_args, published, local_judge):
    def settings(value):
        value = copy.deepcopy(value or {})
        if local_judge:
            value.get("task", {}).get("judge", {}).pop("base_url", None)
            value.get("_audit_judge", {}).pop("base_url", None)
        return value

    if (
        report.get("env") != env_id
        or report.get("examples") != examples
        or report.get("published") != published
        or settings(report.get("taskset_config")) != settings(config)
        or (report.get("env_args") or {}) != (env_args or {})
    ):
        raise RuntimeError("cached report configuration changed; use a new output directory")


def resolve(env_id: str) -> tuple[str, str, str]:
    """Package name, index URL, and the publish time of the latest version."""
    done = subprocess.run(
        ["prime", "env", "info", env_id, "--plain"], capture_output=True, text=True, timeout=120
    )
    match = _INSTALL.search(done.stdout)
    if not match:
        raise RuntimeError(f"no install command in `prime env info {env_id}`")
    status = subprocess.run(
        ["prime", "env", "status", env_id, "--output", "json", "--plain"],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    created = json.loads(status.stdout)["latest_version"]["created_at"]
    return match.group(1), match.group(2), created


def audit(
    env_id: str,
    out_root: Path,
    examples: int,
    timeout: int,
    taskset_config: dict | None = None,
    relay_token: str | None = None,
    relay_dir: str | None = None,
    image: str = IMAGE,
    env_args: dict | None = None,
    resume: bool = False,
) -> dict:
    out = out_root / env_id.replace("/", "__")
    report_path = out / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("stage") == "complete":
            _package, _index, published = resolve(env_id)
            check_cached_report(
                previous,
                env_id,
                examples,
                taskset_config,
                env_args,
                published,
                bool(relay_token),
            )
        if not resume or previous.get("stage") == "complete":
            return previous
        report_path.rename(out / f"report.previous-{uuid.uuid4().hex[:8]}.json")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o777)
    container = "hubaudit-" + uuid.uuid4().hex[:12]
    try:
        package, index, created = resolve(env_id)
        cmd = [
            "docker", "run", "--rm", "--name", container,
            "--memory", "6g", "--cpus", "2", "--pids-limit", "512",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "-v", f"{HERE}:/audit:ro", "-v", f"{out}:/out",
            image, "python", "/audit/driver.py", "--package", package, "--index", index,
            "--out", "/out", "--examples", str(examples), "--published", created,
        ]  # fmt: skip
        if taskset_config:
            (out / "taskset-config.json").write_text(json.dumps(taskset_config))
            cmd += ["--taskset-config", "/out/taskset-config.json"]
        if env_args:
            cmd += ["--env-args", json.dumps(env_args)]
        if resume:
            cmd += ["--resume"]
        if relay_token:
            cmd += ["--relay-token", relay_token]
        if relay_dir:
            cmd[3:3] = ["-v", f"{relay_dir}:/judge:ro"]
            cmd += ["--judge-socket", "/judge/relay.sock"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if not report_path.exists():
            raise RuntimeError(f"no report (exit {done.returncode}): {done.stderr[-1500:]}")
        report = json.loads(report_path.read_text())
    except Exception as exc:
        report = {"stage": "host", "error": f"{type(exc).__name__}: {exc}"[:3000]}
        report_path.write_text(json.dumps(report, indent=2))
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)
    report["env"] = env_id
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("envs", nargs="*")
    parser.add_argument("--list")
    parser.add_argument("--out", default="runs/hub_audit")
    parser.add_argument("--examples", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--taskset-config", help="JSON file with v1 taskset configuration")
    parser.add_argument(
        "--env-args", type=json.loads, help="JSON object with v0 environment arguments"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse completed phases of a failed run"
    )
    parser.add_argument(
        "--image", default=IMAGE, help="isolated audit image (may contain cached deps)"
    )
    parser.add_argument(
        "--judge", action="store_true", help="use bounded host Prime CLI judge relay"
    )
    args = parser.parse_args()
    if args.env_args is not None and not isinstance(args.env_args, dict):
        parser.error("--env-args must be a JSON object")
    args.examples = args.examples if args.examples is not None else (10 if args.judge else 40)
    if args.judge and args.examples > 10:
        parser.error("--judge supports at most 10 examples within the fixed call budget")
    envs = list(args.envs)
    if args.list:
        envs += [
            line.split("#")[0].strip()
            for line in Path(args.list).read_text().splitlines()
            if line.split("#")[0].strip()
        ]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.taskset_config).read_text()) if args.taskset_config else {}
    with contextlib.ExitStack() as stack:
        token = None
        relay_dir = None
        if args.judge:
            from judge_relay import MODEL, PrimeJudgeRelay, serve

            relay = stack.enter_context(PrimeJudgeRelay(out_root / "judge-usage.jsonl"))
            relay_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="hubaudit-judge-"))
            Path(relay_dir).chmod(0o755)
            url = stack.enter_context(serve(relay, socket_path=Path(relay_dir) / "relay.sock"))
            token = relay.token
            config["_audit_judge"] = {
                "base_url": url,
                "api_key_var": "PROBE_KEY",
                "model": MODEL,
                "sampling": {"temperature": 0, "max_tokens": 32},
            }
        pool = stack.enter_context(ThreadPoolExecutor(args.jobs))
        for report in pool.map(
            lambda e: audit(
                e,
                out_root,
                args.examples,
                args.timeout,
                config,
                token,
                relay_dir,
                args.image,
                args.env_args,
                args.resume,
            ),
            envs,
        ):
            status = report.get("stage")
            print(f"{report['env']:55s} {status:9s} {(report.get('error') or '')[:100]}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
