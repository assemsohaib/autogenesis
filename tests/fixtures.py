"""Builder replies used across the tests: one good environment and many broken ones.

Each broken environment has exactly one known defect, so a test can check that the
audit finds that defect and only that defect.
"""

GOOD_ENV = """
import random
import re

_ANSWER = re.compile(r"^ANSWER:\\s*(-?\\d+)\\s*$")


def generate(seed):
    rng = random.Random(seed)
    numbers = [rng.randint(-999, 999) for _ in range(rng.randint(6, 12))]
    weights = [rng.randint(1, 9) for _ in numbers]
    terms = ", ".join(f"{w}*({n})" for w, n in zip(weights, numbers))
    prompt = (
        "Compute the weighted sum " + terms + ". "
        "End your reply with exactly one line of the form `ANSWER: <integer>`."
    )
    return {"prompt": prompt, "numbers": numbers, "weights": weights}


def solve(task):
    total = sum(w * n for w, n in zip(task["weights"], task["numbers"]))
    return f"ANSWER: {total}"


def verify(task, response):
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    answers = [line for line in lines if line.upper().startswith("ANSWER")]
    if len(answers) != 1 or not lines or lines[-1] != answers[0]:
        return 0.0
    match = _ANSWER.match(answers[0])
    if not match:
        return 0.0
    expected = sum(w * n for w, n in zip(task["weights"], task["numbers"]))
    return 1.0 if int(match.group(1)) == expected else 0.0
"""


def reply(code: str, prose: str = "A weighted-sum environment.") -> str:
    return f"{prose}\n\n```python\n{code.strip()}\n```\n"


GOOD_REPLY = reply(GOOD_ENV)

# verify() accepts any response that contains the answer: the shotgun probe wins.
SUBSTRING_VERIFY = GOOD_ENV.replace(
    "def verify(task, response):",
    "def verify(task, response):\n"
    "    expected = sum(w * n for w, n in zip(task['weights'], task['numbers']))\n"
    "    return 1.0 if str(expected) in response else 0.0\n\n\n"
    "def _unused(task, response):",
)

# A yes/no task: the constant "yes" or the most common answer passes about half the time.
YES_NO_ENV = """
import random


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 10_000)
    return {"prompt": f"Is {n} even? Answer yes or no.", "n": n}


def solve(task):
    return "yes" if task["n"] % 2 == 0 else "no"


def verify(task, response):
    return 1.0 if response.strip().lower() == solve(task) else 0.0
"""

# The answer is printed in the prompt.
LEAKY_ENV = """
import random


def generate(seed):
    rng = random.Random(seed)
    word = "".join(rng.choice("abcdefghij") for _ in range(8))
    return {"prompt": f"Repeat this word exactly: {word}", "word": word}


def solve(task):
    return task["word"]


def verify(task, response):
    return 1.0 if response.strip() == task["word"] else 0.0
"""

# Uses the module-level random state, so the same seed gives different tasks.
NONDETERMINISTIC_ENV = GOOD_ENV.replace(
    "rng = random.Random(seed)", "rng = random.Random(seed + random.randint(1, 10**9))"
)

# Every task is the same prompt.
CONSTANT_ENV = GOOD_ENV.replace("rng = random.Random(seed)", "rng = random.Random(0)")

# verify() rejects even the reference answer.
UNSOLVABLE_ENV = GOOD_ENV.replace(
    "return 1.0 if int(match.group(1)) == expected else 0.0",
    "return 1.0 if int(match.group(1)) == expected + 1 else 0.0",
)

CRASHING_GENERATE = GOOD_ENV.replace(
    "rng = random.Random(seed)", "rng = random.Random(seed)\n    raise RuntimeError('boom')"
)

LOOPING_VERIFY = GOOD_ENV.replace(
    "def verify(task, response):", "def verify(task, response):\n    while True:\n        pass\n"
)

EXITING_VERIFY = GOOD_ENV.replace(
    "def verify(task, response):", "def verify(task, response):\n    raise SystemExit(0)\n"
)

OUT_OF_RANGE_VERIFY = GOOD_ENV.replace(
    "return 1.0 if int(match.group(1)) == expected else 0.0",
    "return 2.0 if int(match.group(1)) == expected else 0.0",
)

IMPORTS_OS = "import os\n" + GOOD_ENV
