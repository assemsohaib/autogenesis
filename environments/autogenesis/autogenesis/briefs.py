"""Domain briefs: the seed prompts for the builder.

Each brief names a domain, not a task, so the builder must choose the task family,
the difficulty dial, and the answer format itself. Whole families are held out for
the eval split, so eval measures transfer to domains the builder never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Brief:
    id: str
    family: str
    title: str
    description: str


EVAL_FAMILIES = frozenset({"grids", "code_tracing"})

BRIEFS: tuple[Brief, ...] = (
    # --- arithmetic and number theory
    Brief(
        "modular-chains",
        "numbers",
        "Modular arithmetic chains",
        "Tasks that need a sequence of modular operations, such as repeated "
        "exponentiation or linear congruences, with a single integer answer.",
    ),
    Brief(
        "base-conversion",
        "numbers",
        "Number bases",
        "Conversions and arithmetic across number bases (2 to 36), including "
        "mixed-base expressions.",
    ),
    Brief(
        "roman-arithmetic",
        "numbers",
        "Roman numerals",
        "Arithmetic expressed in Roman numerals, with the answer in Roman numerals.",
    ),
    Brief(
        "fraction-simplify",
        "numbers",
        "Exact fractions",
        "Expressions with fractions that must be reduced to lowest terms exactly.",
    ),
    Brief(
        "divisor-puzzles",
        "numbers",
        "Divisors and primes",
        "Questions about divisors, prime factors, totients, or gcd/lcm of generated numbers.",
    ),
    # --- strings and text transforms
    Brief(
        "cipher-decode",
        "strings",
        "Classical ciphers",
        "Decode or encode text with a generated classical cipher (Caesar, Vigenere, "
        "affine, rail fence) whose key is given or must be inferred.",
    ),
    Brief(
        "string-rewrite",
        "strings",
        "String rewriting systems",
        "Apply a generated list of rewrite rules to a string for a number of steps.",
    ),
    Brief(
        "anagram-sets",
        "strings",
        "Anagram and multiset reasoning",
        "Questions about letter multisets across several generated words.",
    ),
    Brief(
        "edit-distance",
        "strings",
        "Edit distance",
        "Minimum edit operations between generated strings under a given cost model.",
    ),
    # --- logic
    Brief(
        "truth-tellers",
        "logic",
        "Knights and knaves",
        "Generated islanders who always lie or always tell the truth; identify who "
        "is who from their statements.",
    ),
    Brief(
        "constraint-seating",
        "logic",
        "Seating constraints",
        "Place people in seats from a generated set of constraints with one unique solution.",
    ),
    Brief(
        "boolean-sat",
        "logic",
        "Boolean formulas",
        "Satisfiability or model counting for small generated boolean formulas.",
    ),
    Brief(
        "syllogism-chains",
        "logic",
        "Relation chains",
        "Long chains of order relations (taller than, before, inside) and a question "
        "about two of the entities.",
    ),
    # --- algorithms and graphs
    Brief(
        "shortest-path",
        "graphs",
        "Weighted shortest paths",
        "Shortest path cost or route in a generated weighted graph given as text.",
    ),
    Brief(
        "topo-order",
        "graphs",
        "Dependency ordering",
        "Task dependencies; give the lexicographically smallest valid order, or detect a cycle.",
    ),
    Brief(
        "max-flow",
        "graphs",
        "Network flow",
        "Maximum flow or minimum cut in a small generated network.",
    ),
    Brief(
        "interval-scheduling",
        "graphs",
        "Interval scheduling",
        "Choose or count meetings under room and time constraints.",
    ),
    # --- dates and time
    Brief(
        "calendar-math",
        "time",
        "Calendar arithmetic",
        "Weekdays, date differences, and business-day offsets with generated holidays.",
    ),
    Brief(
        "timezone-meetings",
        "time",
        "Time zones",
        "Meeting times across generated fixed-offset time zones.",
    ),
    # --- data wrangling
    Brief(
        "table-query",
        "data",
        "Table queries",
        "A generated text table and a question that needs filtering, grouping, and aggregation.",
    ),
    Brief(
        "log-forensics",
        "data",
        "Log analysis",
        "Generated server logs; answer a question about sessions, errors, or latencies.",
    ),
    Brief(
        "json-paths",
        "data",
        "Nested records",
        "A generated nested JSON document and a question that needs several joins "
        "across its parts.",
    ),
    # --- simulation and state machines
    Brief(
        "register-machine",
        "simulation",
        "Register machines",
        "A generated program for a tiny register machine; give the final register state.",
    ),
    Brief(
        "cellular-automata",
        "simulation",
        "Cellular automata",
        "Run a generated 1D cellular automaton rule for some steps from a given row.",
    ),
    Brief(
        "inventory-sim",
        "simulation",
        "Inventory simulation",
        "A sequence of generated stock events; give the final inventory value under "
        "an accounting rule (FIFO or LIFO).",
    ),
    Brief(
        "queue-sim",
        "simulation",
        "Queues",
        "Customers arrive and get served under a generated policy; answer a question "
        "about waiting times.",
    ),
    # --- combinatorics and probability
    Brief(
        "exact-probability",
        "combinatorics",
        "Exact probability",
        "Dice, cards, or urns with an exact rational probability as the answer.",
    ),
    Brief(
        "counting-paths",
        "combinatorics",
        "Lattice path counting",
        "Count lattice paths with generated blocked points or step rules.",
    ),
    Brief(
        "partition-count",
        "combinatorics",
        "Partitions",
        "Count integer partitions or compositions under generated constraints.",
    ),
    # --- held out: grids
    Brief(
        "grid-maze",
        "grids",
        "Grid mazes",
        "A generated text maze; give the length of the shortest path, or the path.",
    ),
    Brief(
        "sokoban-lite",
        "grids",
        "Box pushing",
        "A tiny box-pushing puzzle; give a valid move sequence or the minimum moves.",
    ),
    Brief(
        "flood-regions",
        "grids",
        "Grid regions",
        "Count or measure connected regions in a generated grid after some edits.",
    ),
    # --- held out: code tracing
    Brief(
        "trace-python",
        "code_tracing",
        "Program tracing",
        "A generated short program in a simple Python subset; give its exact output.",
    ),
    Brief(
        "stack-language",
        "code_tracing",
        "Stack languages",
        "Programs in a generated stack-based language with a given instruction set; "
        "give the final stack.",
    ),
    Brief(
        "regex-match",
        "code_tracing",
        "Regular expressions",
        "Generated regular expressions and strings; give which strings match, or the "
        "first match span.",
    ),
)


def briefs_for(split: str) -> list[Brief]:
    if split == "all":
        return list(BRIEFS)
    if split == "train":
        return [b for b in BRIEFS if b.family not in EVAL_FAMILIES]
    if split == "eval":
        return [b for b in BRIEFS if b.family in EVAL_FAMILIES]
    raise ValueError(f"unknown split {split!r}; use train, eval, or all")
