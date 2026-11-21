"""Host-only logic: code extraction, policy, probes, reward combination."""

import pytest
import verifiers.v1 as vf

from autogenesis import AuditReport, combine, extract_code
from autogenesis.env import PanelResult
from autogenesis.policy import check_code
from autogenesis.probes import PROBE_NAMES, build_probes
from fixtures import GOOD_ENV, IMPORTS_OS


class TestExtractCode:
    def test_takes_last_python_block(self):
        reply = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
        assert extract_code(reply) == "second = 2"

    def test_prefers_python_over_other_languages(self):
        reply = "```python\nx = 1\n```\n```text\nnot code\n```"
        assert extract_code(reply) == "x = 1"

    def test_falls_back_to_any_fence(self):
        assert extract_code("```\ny = 2\n```") == "y = 2"

    @pytest.mark.parametrize("reply", ["", "no code here", "```python\n   \n```"])
    def test_none_without_code(self, reply):
        assert extract_code(reply) is None


class TestPolicy:
    def test_good_env_passes(self):
        assert check_code(GOOD_ENV).ok

    @pytest.mark.parametrize(
        ("code", "fragment"),
        [
            (IMPORTS_OS, "'os'"),
            ("from subprocess import run\n" + GOOD_ENV, "subprocess"),
            ("from . import x\n" + GOOD_ENV, "import from"),
            (GOOD_ENV + "\nx = eval('1')", "'eval'"),
            (GOOD_ENV + "\nx = open", "'open'"),
            (GOOD_ENV + "\nx = getattr(int, 'real')", "'getattr'"),
            (GOOD_ENV + "\nx = ().__class__", "__class__"),
            (GOOD_ENV + "\nx = __builtins__", "__builtins__"),
            (GOOD_ENV + "\ndef f():\n    global y\n", "global"),
            (GOOD_ENV.replace("def solve(task):", "def solve_it(task):"), "solve()"),
            (GOOD_ENV.replace("def verify(task, response):", "def verify(task):"), "verify()"),
            ("def generate(seed:\n", "syntax error"),
        ],
    )
    def test_violations(self, code, fragment):
        result = check_code(code)
        assert not result.ok
        assert any(fragment in v for v in result.violations), result.violations

    def test_size_limit(self):
        assert not check_code(GOOD_ENV + "#" * 30_000).ok

    def test_optional_extra_parameters_are_allowed(self):
        code = GOOD_ENV.replace("def solve(task):", "def solve(task, unused=None):")
        assert check_code(code).ok


class TestProbes:
    def test_every_probe_for_every_task(self):
        probes = build_probes(["p1 has 3", "p2 has 4"], ["a", "b"])
        assert len(probes) == 2 * len(PROBE_NAMES)
        by_name = {(name, i): response for name, i, response in probes}
        assert by_name[("other_answer", 0)] == "b"
        assert by_name[("shotgun", 1)] == "a\nb"
        assert by_name[("echo_prompt", 1)] == "p2 has 4"
        assert by_name[("prompt_numbers", 0)] == "1 3"

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError):
            build_probes(["p"], [])


class TestCombine:
    def test_no_panel_is_audit_score(self):
        assert combine(AuditReport(stage="complete", score=0.8), None, 0.25) == 0.8

    def test_floor_and_discrimination(self):
        report = AuditReport(stage="complete", score=0.8)
        assert combine(report, PanelResult(0.5, 0.5), 0.25) == pytest.approx(0.2)
        assert combine(report, PanelResult(0.0, 1.0), 0.25) == pytest.approx(0.8)
        # A weak model that beats the strong model is not negative discrimination.
        assert combine(report, PanelResult(1.0, 0.0), 0.25) == pytest.approx(0.2)


class TestEnvConfig:
    def _env(self, **extra):
        config = vf.resolve_env_config({"taskset": {"id": "autogenesis"}, **extra})
        return vf.load_environment(config)

    def test_panel_needs_two_pinned_models(self):
        with pytest.raises(ValueError, match="two different pinned models"):
            self._env()
        with pytest.raises(ValueError, match="two different pinned models"):
            self._env(weak={"model": "m"}, strong={"model": "m"})

    def test_panel_off_or_pinned_loads(self):
        assert self._env(panel={"instances": 0})
        assert self._env(weak={"model": "small"}, strong={"model": "large"})

    def test_splits_hold_out_families(self):
        train = list(self._env(panel={"instances": 0}).taskset)
        held_out = list(
            self._env(
                panel={"instances": 0}, taskset={"id": "autogenesis", "split": "eval"}
            ).taskset
        )
        assert {t.data.family for t in train}.isdisjoint({t.data.family for t in held_out})
        assert len(train) + len(held_out) == 35
        assert len({t.key for t in train}) == len(train)


def test_mode_probe_leaves_own_answer_out():
    probes = build_probes(["a", "b", "c"], ["x", "x", "y"])
    modes = {i: r for name, i, r in probes if name == "mode_answer"}
    assert modes == {0: "x", 1: "x", 2: "x"}
    unique = build_probes(["a", "b"], ["x", "y"])
    assert {i: r for name, i, r in unique if name == "mode_answer"} == {0: "y", 1: "x"}


def test_leak_ignores_template_words():
    from autogenesis.audit import leak_rate

    template = [f"Case {i}: is it valid or invalid?" for i in range(4)]
    assert leak_rate(template, ["valid", "invalid", "valid", "valid"], 4) == 0.0
    leaky = [f"Repeat the word {w}" for w in ("apple", "pear", "plum", "kiwi")]
    assert leak_rate(leaky, ["apple", "pear", "plum", "kiwi"], 4) == 1.0
    assert leak_rate(leaky, ["apple", "x", "y", "z"], 4) == 0.25
