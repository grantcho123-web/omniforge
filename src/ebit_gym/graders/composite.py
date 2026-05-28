"""Composite grader.

Combines multiple sub-graders into a single weighted score. Config:

* ``components``: list of ``{"weight": <float>, "spec": <GraderSpec dict>}``.
  Weights are normalized to sum to 1.
* ``pass_threshold``: default ``0.7``.

Sub-graders are built via the global registry, so this only works for
sub-graders that don't need runtime resources. For composites involving
an LLM judge, build the components manually and pass the list directly
via the alt constructor ``CompositeGrader.from_components([(w, g), ...])``.
"""
from __future__ import annotations

from ebit_gym.core.grader import Grader, make_grader, register_grader
from ebit_gym.core.task import Attempt, GraderSpec, GradingResult, Task


class CompositeGrader(Grader):
    name = "composite"

    def __init__(self, spec: GraderSpec) -> None:
        cfg = spec.config
        components_raw = cfg.get("components")
        if not components_raw:
            raise ValueError("composite grader requires 'components' in config")
        self.pass_threshold: float = float(cfg.get("pass_threshold", 0.7))

        pairs: list[tuple[float, Grader]] = []
        for c in components_raw:
            weight = float(c.get("weight", 1.0))
            sub_spec = GraderSpec.model_validate(c["spec"])
            pairs.append((weight, make_grader(sub_spec)))
        self._set_components(pairs)

    @classmethod
    def from_components(
        cls,
        components: list[tuple[float, Grader]],
        pass_threshold: float = 0.7,
    ) -> CompositeGrader:
        """Bypass the registry — useful when a sub-grader needs runtime resources."""
        obj = cls.__new__(cls)
        obj.pass_threshold = pass_threshold
        obj._set_components(components)
        return obj

    def _set_components(self, pairs: list[tuple[float, Grader]]) -> None:
        total = sum(w for w, _ in pairs)
        if total <= 0:
            raise ValueError("composite grader weights must sum to a positive number")
        self.components: list[tuple[float, Grader]] = [(w / total, g) for w, g in pairs]

    def grade(self, task: Task, attempt: Attempt) -> GradingResult:
        breakdown: dict[str, float] = {}
        weighted = 0.0
        rationale_parts: list[str] = []
        for i, (weight, sub) in enumerate(self.components):
            sub_result = sub.grade(task, attempt)
            key = f"{sub.name}#{i}"
            breakdown[key] = sub_result.score
            weighted += weight * sub_result.score
            if sub_result.rationale:
                rationale_parts.append(f"[{key} w={weight:.2f}] {sub_result.rationale}")

        return GradingResult(
            task_id=task.metadata.task_id,
            attempt_id=attempt.attempt_id,
            score=weighted,
            passed=weighted >= self.pass_threshold,
            rationale="; ".join(rationale_parts) or None,
            breakdown=breakdown,
            graded_by=self.name,
        )


@register_grader("composite")
def _factory(spec: GraderSpec) -> CompositeGrader:
    return CompositeGrader(spec)
