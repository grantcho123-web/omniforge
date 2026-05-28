"""Command-line interface for ebit-gym.

The CLI is the canonical way to run an eval end-to-end without writing
Python::

    ebit-gym list-models
    ebit-gym list-graders
    ebit-gym inspect-taskset corpora/reference-v0/manifest.json
    ebit-gym eval --task-set corpora/reference-v0/manifest.json \\
                  --model anthropic:claude-4.6-sonnet \\
                  --output runs/2026-05-22.json

Limitations of v0:
- ``llm_judge`` graders are skipped with a warning — they need a runtime
  judge callable. Build those evals programmatically until v0.3.
- One-shot Q&A only. Multi-turn and tool-use protocols come later.
- Synchronous; large task sets will take a while.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import IO

# Importing these registers all built-in graders and model adapters.
import ebit_gym.graders  # noqa: F401
import ebit_gym.models  # noqa: F401
from ebit_gym.core import Attempt, GradingResult, Task, TaskSet
from ebit_gym.core.grader import make_grader
from ebit_gym.core.grader import registered_graders as list_graders
from ebit_gym.core.model import make_model
from ebit_gym.core.model import registered_models as list_models
from ebit_gym.core.runner import AttemptRunner, RunnerConfig

EXIT_OK = 0
EXIT_ERROR = 1

# Graders we can build from spec alone (no runtime callables).
_GRADERS_REQUIRING_RUNTIME = {"llm_judge"}


def _load_taskset(path: Path) -> TaskSet:
    raw = path.read_text(encoding="utf-8")
    return TaskSet.model_validate_json(raw)


def _grade_attempt(
    task: Task,
    attempt: Attempt,
    *,
    out: IO[str],
) -> GradingResult | None:
    """Grade one attempt; returns None and prints a warning for unsupported graders."""
    if task.grader_spec.type in _GRADERS_REQUIRING_RUNTIME:
        print(
            f"  [skipped] task {task.metadata.task_id} uses '{task.grader_spec.type}' grader "
            f"which needs runtime resources — build programmatically",
            file=out,
        )
        return None
    try:
        grader = make_grader(task.grader_spec)
        return grader.grade(task, attempt)
    except Exception as e:  # noqa: BLE001 — surface any grader bug as a skip, not a crash
        print(
            f"  [grader error] task {task.metadata.task_id}: {type(e).__name__}: {e}",
            file=out,
        )
        return None


# ----------------------------------------------------------------- commands


def cmd_eval(args: argparse.Namespace) -> int:
    out = sys.stdout
    taskset = _load_taskset(Path(args.task_set))
    print(
        f"Loaded task set '{taskset.name}' v{taskset.version} "
        f"({len(taskset.tasks)} tasks)",
        file=out,
    )

    try:
        model = make_model(args.model)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Model: {model.name}", file=out)

    runner = AttemptRunner(
        model,
        RunnerConfig(
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    print("\nRunning attempts...", file=out)
    attempts = runner.run_set(taskset, split=args.split, ids=args.ids)

    print(f"Grading {len(attempts)} attempts...\n", file=out)
    results: list[tuple[Task, Attempt, GradingResult | None]] = []
    for attempt in attempts:
        task = taskset.task_by_id(attempt.task_id)
        result = _grade_attempt(task, attempt, out=out)
        results.append((task, attempt, result))

    _print_summary(results, out=out)

    if args.output:
        _dump_results(taskset, results, Path(args.output))
        print(f"\nWrote results JSON to {args.output}", file=out)

    return EXIT_OK


def cmd_list_models(_args: argparse.Namespace) -> int:
    for name in list_models():
        print(name)
    return EXIT_OK


def cmd_list_graders(_args: argparse.Namespace) -> int:
    for name in list_graders():
        print(name)
    print("llm_judge  (programmatic only — needs runtime judge callable)")
    return EXIT_OK


def cmd_inspect_taskset(args: argparse.Namespace) -> int:
    ts = _load_taskset(Path(args.task_set))
    print(f"name:        {ts.name}")
    print(f"version:     {ts.version}")
    print(f"description: {ts.description or '(none)'}")
    print(f"tasks:       {len(ts.tasks)}")
    if ts.splits:
        print("splits:")
        for s, ids in ts.splits.items():
            print(f"  {s}: {len(ids)} tasks")
    domains = Counter(t.metadata.domain for t in ts.tasks)
    print("domains:")
    for d, n in sorted(domains.items()):
        print(f"  {d}: {n}")
    grader_types = Counter(t.grader_spec.type for t in ts.tasks)
    print("graders:")
    for g, n in sorted(grader_types.items()):
        print(f"  {g}: {n}")
    return EXIT_OK


# ---------------------------------------------------------------- output


def _print_summary(
    results: list[tuple[Task, Attempt, GradingResult | None]],
    *,
    out: IO[str],
) -> None:
    scored = [r for _, _, r in results if r is not None]
    print(f"{'TASK':<24} {'SCORE':>6}  {'PASS':>5}  GRADED_BY", file=out)
    print("-" * 60, file=out)
    for task, _attempt, result in results:
        if result is None:
            print(f"{task.metadata.task_id:<24} {'--':>6}  {'--':>5}  (skipped)", file=out)
            continue
        print(
            f"{task.metadata.task_id:<24} {result.score:>6.3f}  "
            f"{'✓' if result.passed else '✗':>5}  {result.graded_by}",
            file=out,
        )
    print("-" * 60, file=out)
    if scored:
        avg = sum(r.score for r in scored) / len(scored)
        passes = sum(1 for r in scored if r.passed)
        print(
            f"Aggregate: avg score {avg:.3f}, passed {passes}/{len(scored)} "
            f"({passes / len(scored):.1%})",
            file=out,
        )
    else:
        print("Aggregate: no graded attempts.", file=out)

    total_cost = sum(
        (a.cost.usd or 0.0) for _, a, _ in results if a.cost and a.cost.usd is not None
    )
    if total_cost > 0:
        print(f"Estimated cost: ${total_cost:.4f}", file=out)


def _dump_results(
    taskset: TaskSet,
    results: list[tuple[Task, Attempt, GradingResult | None]],
    path: Path,
) -> None:
    payload = {
        "taskset": {"name": taskset.name, "version": taskset.version},
        "results": [
            {
                "task_id": task.metadata.task_id,
                "attempt": json.loads(attempt.model_dump_json()),
                "grading": json.loads(result.model_dump_json()) if result else None,
            }
            for task, attempt, result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ebit-gym")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run a model over a task set and grade.")
    p_eval.add_argument("--task-set", required=True, help="Path to a TaskSet JSON file.")
    p_eval.add_argument("--model", required=True, help="Provider-qualified model id.")
    p_eval.add_argument("--split", default=None, help="Split name to filter on.")
    p_eval.add_argument("--ids", nargs="*", default=None, help="Specific task ids to run.")
    p_eval.add_argument("--system", default=None, help="System prompt.")
    p_eval.add_argument("--max-tokens", type=int, default=1024)
    p_eval.add_argument("--temperature", type=float, default=0.0)
    p_eval.add_argument("--output", default=None, help="Path to dump full results JSON.")
    p_eval.set_defaults(func=cmd_eval)

    p_models = sub.add_parser("list-models", help="List registered model adapters.")
    p_models.set_defaults(func=cmd_list_models)

    p_graders = sub.add_parser("list-graders", help="List registered graders.")
    p_graders.set_defaults(func=cmd_list_graders)

    p_inspect = sub.add_parser("inspect-taskset", help="Summarize a TaskSet JSON file.")
    p_inspect.add_argument("task_set", help="Path to a TaskSet JSON file.")
    p_inspect.set_defaults(func=cmd_inspect_taskset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
