"""Export adapters — turn graded corpora into lab-consumable training data.

Frontier labs each have their own ingestion format. The platform's job
is to keep the canonical Task/Attempt/GradingResult representation
clean, and emit the lab-specific shape only at the boundary.

Three exporters in v0.2:

* :func:`export_openai_finetune_jsonl` — OpenAI fine-tune format,
  one JSON object per line with a ``messages`` array.
* :func:`export_anthropic_jsonl` — Anthropic message format, one
  object per line.
* :func:`export_generic_jsonl` — flat ``{prompt, completion, score}``
  records for custom pipelines.

All three accept a list of ``(Task, Attempt, GradingResult | None)``
triples — exactly what the CLI's eval loop already produces.

Filtering:

* By default we export only attempts that ``passed`` (high-quality
  training data only). Set ``include_failed=True`` to export everything,
  useful for negative examples or preference-data construction.
* Skipped attempts (no GradingResult) are never exported.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from omniforge.core.task import Attempt, GradingResult, Task

ResultTriple = tuple[Task, Attempt, GradingResult | None]


def _filter(
    triples: Iterable[ResultTriple], *, include_failed: bool
) -> list[ResultTriple]:
    out: list[ResultTriple] = []
    for task, attempt, grade in triples:
        if grade is None:
            continue
        if attempt.error:
            continue
        if not include_failed and not grade.passed:
            continue
        out.append((task, attempt, grade))
    return out


# ------------------------------------------------------------------- writer


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------- openai fine-tune


def export_openai_finetune_jsonl(
    triples: Iterable[ResultTriple],
    path: Path,
    *,
    include_failed: bool = False,
    system_prompt: str | None = None,
) -> int:
    """Write OpenAI fine-tune format.

    Each line is ``{"messages": [...]}``. Optionally prepends a system
    message to every example. Returns the number of records written.
    """
    filtered = _filter(triples, include_failed=include_failed)

    def gen():
        for task, attempt, _grade in filtered:
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": _render_user_message(task)})
            messages.append({"role": "assistant", "content": attempt.raw_response})
            yield {"messages": messages}

    return _write_jsonl(path, gen())


# -------------------------------------------------------------- anthropic


def export_anthropic_jsonl(
    triples: Iterable[ResultTriple],
    path: Path,
    *,
    include_failed: bool = False,
    system_prompt: str | None = None,
) -> int:
    """Write Anthropic message format (one example per line).

    Schema: ``{"system": <optional str>, "messages": [{role, content}, ...]}``.
    """
    filtered = _filter(triples, include_failed=include_failed)

    def gen():
        for task, attempt, _grade in filtered:
            record: dict = {
                "messages": [
                    {"role": "user", "content": _render_user_message(task)},
                    {"role": "assistant", "content": attempt.raw_response},
                ],
            }
            if system_prompt:
                record["system"] = system_prompt
            yield record

    return _write_jsonl(path, gen())


# --------------------------------------------------------------- generic


def export_generic_jsonl(
    triples: Iterable[ResultTriple],
    path: Path,
    *,
    include_failed: bool = True,
) -> int:
    """Write a flat ``{task_id, prompt, completion, score, passed, ...}``
    JSONL — convenient for custom pipelines and for negative-example
    construction (preference data, DPO, etc).

    Defaults to ``include_failed=True`` since the explicit ``score``
    field makes labeling examples by quality the customer's call.
    """
    filtered = _filter(triples, include_failed=include_failed)

    def gen():
        for task, attempt, grade in filtered:
            yield {
                "task_id": task.metadata.task_id,
                "task_version": task.metadata.version,
                "domain": task.metadata.domain,
                "prompt": _render_user_message(task),
                "completion": attempt.raw_response,
                "model": attempt.model,
                "score": grade.score if grade else None,
                "passed": grade.passed if grade else None,
                "graded_by": grade.graded_by if grade else None,
            }

    return _write_jsonl(path, gen())


# ---------------------------------------------------------- shared rendering


def _render_user_message(task: Task) -> str:
    """Same shape the AttemptRunner used to produce the original prompt.

    Centralized so exports stay symmetric with the runtime: the model
    that produced the assistant message saw exactly this user message.
    """
    if not task.materials:
        return task.prompt
    parts: list[str] = [task.prompt, ""]
    for i, m in enumerate(task.materials, start=1):
        header = f"--- material {i}"
        if m.name:
            header += f" ({m.name})"
        header += " ---"
        parts.append(header)
        parts.append(m.content)
    return "\n".join(parts)
