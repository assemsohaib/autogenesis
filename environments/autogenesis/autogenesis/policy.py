"""Static policy for builder code, checked on the host before any of it runs.

The builder writes a small pure-Python module. Generated environments only need
computation, so the policy allows a short list of standard-library modules and bans
every route to I/O, introspection, or the sandbox runner itself. The runner enforces
the same import allowlist again at run time (see `runner.py`); this pass exists so a
violation is a clean, explainable failure instead of a crash inside the sandbox.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "unicodedata",
    }
)
"""Top-level modules the builder may import. Kept in sync with `runner.ALLOWED`."""

BANNED_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)
"""Builtins that reach I/O, dynamic code, or namespace introspection."""

REQUIRED_FUNCTIONS: dict[str, int] = {"generate": 1, "solve": 1, "verify": 2}
"""The contract: function name -> number of positional parameters."""

MAX_CODE_CHARS = 20_000


@dataclass
class PolicyResult:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def check_code(code: str) -> PolicyResult:
    """Return every policy violation in `code` (empty when the code is allowed)."""
    result = PolicyResult()
    if len(code) > MAX_CODE_CHARS:
        result.violations.append(f"code is {len(code)} chars; the limit is {MAX_CODE_CHARS}")
        return result
    try:
        with warnings.catch_warnings():  # builder code must not print to the host
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except SyntaxError as e:
        result.violations.append(f"syntax error: {e.msg} (line {e.lineno})")
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    result.violations.append(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                result.violations.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                result.violations.append(f"name {node.id!r} is not allowed")
            elif node.id.startswith("__") and node.id != "__name__":
                result.violations.append(f"dunder name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                result.violations.append(f"dunder attribute {node.attr!r} is not allowed")
            elif node.attr.startswith("_getframe") or node.attr in {"f_back", "f_locals"}:
                result.violations.append(f"frame access {node.attr!r} is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # Module state between calls lets verify() behave differently per call.
            result.violations.append("global/nonlocal state is not allowed")

    defined = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, arity in REQUIRED_FUNCTIONS.items():
        fn = defined.get(name)
        if fn is None:
            result.violations.append(f"missing top-level function {name}()")
            continue
        if isinstance(fn, ast.AsyncFunctionDef):
            result.violations.append(f"{name}() must not be async")
            continue
        args = fn.args
        positional = len(args.posonlyargs) + len(args.args)
        required = positional - len(args.defaults)
        if required > arity or positional < arity:
            result.violations.append(f"{name}() must take exactly {arity} positional argument(s)")
    return result
