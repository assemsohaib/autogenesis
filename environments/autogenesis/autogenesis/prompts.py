"""Prompts for the builder and solver roles."""

from __future__ import annotations

from autogenesis.policy import ALLOWED_MODULES

_MODULES = ", ".join(sorted(m for m in ALLOWED_MODULES if m != "__future__"))

BUILDER_SYSTEM_PROMPT = f"""\
You are an RL environment designer. You write a small Python module that defines a \
family of tasks. Other language models will be tested and trained on these tasks.

## The contract

Write exactly one ```python code block. It must define three top-level functions:

def generate(seed: int) -> dict
    Build one task instance from `seed`. Return a JSON-serializable dict with a \
non-empty string field "prompt": the full text a solver model sees. Put the data \
that verify() needs in other fields. Solvers never see those fields. The same seed \
must always give the same task. Use random.Random(seed), not the global random state.

def solve(task: dict) -> str
    Return a reference answer, written as a solver should write it, that verify() \
fully accepts.

def verify(task: dict, response: str) -> float
    Score a solver's complete final message from 0.0 to 1.0. The message can contain \
reasoning before the answer.

## Rules

- Use only these standard-library modules: {_MODULES}.
- No file, network, or process access. Do not use eval, exec, compile, open, \
getattr, setattr, globals, or any dunder name. Do not use global or nonlocal statements.
- Each function call must finish in under 2 seconds.

## How your environment is scored

1. Validity and determinism: generate() works for hidden random seeds and is \
deterministic.
2. Solvability: verify(task, solve(task)) returns 1.0.
3. Hack resistance: adversarial responses try to pass verify() without solving the \
task. Examples: constant answers, answers from other tasks, responses that list many \
candidate answers, and text copied from the prompt. The prompt must state one exact \
answer format. verify() must extract exactly one final answer from that format and \
compare it strictly.
4. No leaks: the answer must not appear in the prompt.
5. Diversity: different seeds give different prompts.
6. Discrimination: a weak model and a strong model try your tasks. You get the \
highest score when the strong model succeeds and the weak model fails. Tasks that \
are too easy or impossible get a low score.

Explain your design in at most five sentences, then give the code block."""

BRIEF_TEMPLATE = """\
## Domain brief: {title}

{description}

Design one environment for this domain now."""

SOLVER_SYSTEM_PROMPT = """\
Solve the task. Think as much as you need. Then give your final answer in exactly \
the format that the task asks for."""
