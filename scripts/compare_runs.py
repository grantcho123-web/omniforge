"""Compare two or more omniforge eval result JSONs side by side.

Usage:

    python scripts/compare_runs.py runs/haiku.json runs/sonnet.json
    python scripts/compare_runs.py runs/*.json   # arbitrary number of runs

Prints a per-task table with each model's score, a tally of where models
agree vs. disagree, and aggregate stats (pass rate, cost, latency) per
model. Marks disagreement rows with a `*` so you can scan for them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _model_id(run: dict[str, Any]) -> str:
    for r in run.get("results", []):
        m = r.get("attempt", {}).get("model")
        if m:
            return str(m)
    return "(unknown)"


def _short(name: str, width: int) -> str:
    """Trim provider:model id to fit a column."""
    s = name.split(":", 1)[-1]
    return s if len(s) <= width else s[: width - 1] + "…"


def _cell(result: dict[str, Any] | None, width: int) -> tuple[str, bool | None]:
    """Render one model's outcome for one task into a fixed-width cell."""
    if result is None:
        return f"{'(absent)':>{width}}", None
    grading = result.get("grading")
    if grading is None:
        return f"{'(skipped)':>{width}}", None
    score = grading["score"]
    mark = "✓" if grading["passed"] else "✗"
    return f"{score:.3f} {mark}".rjust(width), bool(grading["passed"])


def _row_disagrees(pass_states: list[bool | None]) -> bool:
    valid = [s for s in pass_states if s is not None]
    return bool(valid) and len(set(valid)) > 1


def _aggregate_for(run: dict[str, Any]) -> dict[str, Any]:
    scored = [r for r in run["results"] if r.get("grading")]
    passes = sum(1 for r in scored if r["grading"]["passed"])
    avg = sum(r["grading"]["score"] for r in scored) / len(scored) if scored else 0.0
    cost = sum(
        (r["attempt"]["cost"]["usd"] or 0.0)
        for r in run["results"]
        if r["attempt"].get("cost") and r["attempt"]["cost"].get("usd") is not None
    )
    latency = sum(
        (r["attempt"]["latency_ms"] or 0)
        for r in run["results"]
        if r["attempt"].get("latency_ms") is not None
    )
    return {
        "passes": passes,
        "of": len(scored),
        "avg": avg,
        "cost": cost,
        "latency_ms": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Result JSON files")
    parser.add_argument(
        "--col-width", type=int, default=14, help="Width of each model column."
    )
    args = parser.parse_args()

    if len(args.runs) < 1:
        print("error: need at least one run file", file=sys.stderr)
        return 1

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in args.runs]
    model_ids = [_model_id(r) for r in runs]
    col_short = [_short(m, args.col_width) for m in model_ids]

    # Collect task ids in order, taking the union (first-seen wins).
    seen: set[str] = set()
    all_ids: list[str] = []
    for run in runs:
        for r in run["results"]:
            tid = r["task_id"]
            if tid not in seen:
                seen.add(tid)
                all_ids.append(tid)

    # Index results per (task, run) for O(1) lookup.
    by_task: dict[str, list[dict[str, Any] | None]] = {tid: [] for tid in all_ids}
    for run in runs:
        index = {r["task_id"]: r for r in run["results"]}
        for tid in all_ids:
            by_task[tid].append(index.get(tid))

    # --- Per-task comparison ---
    header = f"{'TASK':<24}" + "".join(s.rjust(args.col_width + 2) for s in col_short) + "  DIFF"
    print(header)
    print("-" * len(header))

    disagreement_count = 0
    for tid in all_ids:
        row = f"{tid:<24}"
        states: list[bool | None] = []
        for result in by_task[tid]:
            cell, passed = _cell(result, args.col_width)
            row += "  " + cell
            states.append(passed)
        if _row_disagrees(states):
            row += "  *"
            disagreement_count += 1
        print(row)

    print("-" * len(header))

    # --- Per-model aggregates ---
    print()
    print(f"{'MODEL':<36}{'PASS':>10}{'AVG':>8}{'COST':>12}{'LATENCY':>12}")
    print("-" * 78)
    for model_id, run in zip(model_ids, runs, strict=True):
        agg = _aggregate_for(run)
        pass_cell = f"{agg['passes']}/{agg['of']}" if agg["of"] else "0/0"
        cost_cell = f"${agg['cost']:.4f}"
        lat_cell = f"{agg['latency_ms']:,} ms"
        print(
            f"{model_id:<36}{pass_cell:>10}{agg['avg']:>8.3f}"
            f"{cost_cell:>12}{lat_cell:>12}"
        )

    print()
    print(f"Disagreements (marked *): {disagreement_count} task(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
