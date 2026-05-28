"""Build the v0.2 reference task corpus.

Produces ``corpora/reference-v0/manifest.json`` — the canonical demo
corpus shipped with the repo. Run from the project root:

    python scripts/build_reference_corpus.py

The corpus is intentionally small (10 tasks) and diverse:
- Mix of English, Korean, and Japanese
- Every reference grader type exercised (exact_match, regex,
  composite, human)
- One task generated from SimulatedTradingTaskBuilder so the trading
  task type is represented
- Splits for 'quick' iteration, 'auto_gradable' for CLI demos, and
  'korean' for the Korean-language subset

This script is the source of truth. The committed manifest.json is
its output — regenerate after any edit.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from omniforge.core.task import (
    GraderSpec,
    Task,
    TaskMetadata,
    TaskSet,
)

OUT_DIR = Path("corpora/reference-v0")
OUT_PATH = OUT_DIR / "manifest.json"

NOW = datetime(2026, 5, 22, tzinfo=timezone.utc)


def _meta(
    task_id: str,
    domain: str,
    difficulty: str = "medium",
    author: str = "omniforge-reference-v0",
    tags: list[str] | None = None,
    language: str = "en",
) -> TaskMetadata:
    return TaskMetadata(
        task_id=task_id,
        version="0.1.0",
        domain=domain,
        difficulty=difficulty,
        author=author,
        created_at=NOW,
        tags=tags or [],
        language=language,
    )


def hand_written_tasks() -> list[Task]:
    return [
        # ---------------------------- ENGLISH / FINANCE / NUMERIC -----------------
        Task(
            metadata=_meta(
                "q-bond-pv-001",
                "finance.quant.bonds",
                difficulty="medium",
                tags=["bond-pricing", "present-value"],
            ),
            prompt=(
                "A 5-year bond pays a $50 annual coupon and matures at $1000 face value. "
                "At a 6% annual discount rate, what is its present value? "
                "Round to the nearest whole dollar and answer with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 2.0}
            ),
            reference_answer="958",
        ),
        Task(
            metadata=_meta(
                "q-bs-call-001",
                "finance.quant.options",
                difficulty="hard",
                tags=["black-scholes", "options"],
            ),
            prompt=(
                "Compute the Black-Scholes call price with the following inputs:\n"
                "  Spot S = 100\n"
                "  Strike K = 100\n"
                "  Risk-free rate r = 0.05 (continuous)\n"
                "  Time to maturity T = 1 year\n"
                "  Volatility sigma = 0.20\n"
                "  No dividends.\n"
                "Round to two decimal places and answer with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.5}
            ),
            reference_answer="10.45",
        ),
        Task(
            metadata=_meta(
                "q-acct-ratio-001",
                "finance.accounting",
                difficulty="easy",
                tags=["financial-ratios", "current-ratio"],
            ),
            prompt=(
                "A company has $300,000 in current assets and $150,000 in current "
                "liabilities. What is its current ratio? Answer with just the number "
                "to one decimal place."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.05}
            ),
            reference_answer="2.0",
        ),
        # ---------------------------- KOREAN / NUMERIC ----------------------------
        Task(
            metadata=_meta(
                "q-ko-bond-001",
                "finance.korean.bonds",
                difficulty="medium",
                tags=["korean", "bond-pricing"],
                language="ko",
            ),
            prompt=(
                "다음 조건의 채권 현재가치를 계산하세요.\n"
                "  만기: 3년\n"
                "  액면가: 1,000원\n"
                "  연 표면이자: 60원 (매년 말 지급)\n"
                "  할인율: 5% (연복리)\n"
                "정수로 반올림하여 숫자만 답하세요."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 2.0}
            ),
            reference_answer="1027",
        ),
        # ---------------------------- KOREAN / CULTURE / HUMAN --------------------
        Task(
            metadata=_meta(
                "q-ko-culture-001",
                "language.korean.culture",
                difficulty="hard",
                tags=["korean", "open-ended", "cultural"],
                language="ko",
            ),
            prompt=(
                "한국의 명절 음식 문화에 대해 한 단락(4-6 문장)으로 설명해 주세요. "
                "최소 두 가지 명절을 언급하고, 각 명절의 대표 음식과 그 의미를 포함하세요."
            ),
            grader_spec=GraderSpec(
                type="human",
                config={
                    "queue": "ko-language-reviewers",
                    "instructions": (
                        "Score: 0.0 incoherent or factually wrong; 0.5 basic but "
                        "accurate; 0.8 fluent with cultural specificity; 1.0 native "
                        "quality with depth of insight."
                    ),
                },
            ),
            rubric=(
                "Native Korean fluency + accurate cultural detail + at least two "
                "distinct holiday traditions + meaning behind the foods."
            ),
        ),
        # ---------------------------- KOREAN / IDIOM / HUMAN ----------------------
        Task(
            metadata=_meta(
                "q-ko-idiom-001",
                "language.korean.reasoning",
                difficulty="hard",
                tags=["korean", "idioms", "open-ended"],
                language="ko",
            ),
            prompt=(
                "다음 사자성어의 의미를 한 문장으로 설명하고, 일상생활에서 사용할 수 있는 "
                "구체적인 예시 하나를 들어 주세요: 우공이산(愚公移山)."
            ),
            grader_spec=GraderSpec(
                type="human",
                config={
                    "queue": "ko-language-reviewers",
                    "instructions": (
                        "Score: correct literal meaning, correct figurative meaning, "
                        "and a plausible everyday example each contribute ~1/3."
                    ),
                },
            ),
            rubric=(
                "Literal: an old man moving a mountain. Figurative: persistence "
                "achieves the seemingly impossible. Example must illustrate this."
            ),
        ),
        # ---------------------------- JAPANESE / NUMERIC --------------------------
        Task(
            metadata=_meta(
                "q-ja-math-001",
                "math.basic",
                difficulty="medium",
                tags=["japanese", "arithmetic"],
                language="ja",
            ),
            prompt=(
                "次の問題に答えてください。ある商品の定価は 8,000円 で、20% の割引が "
                "適用されます。さらに消費税 10% が割引後の価格に加算されます。"
                "最終的な支払い額はいくらですか。数字のみで答えてください。"
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 1.0}
            ),
            reference_answer="7040",
        ),
        # ---------------------------- ENGLISH / REGEX -----------------------------
        Task(
            metadata=_meta(
                "q-currency-001",
                "language.keyword",
                difficulty="easy",
                tags=["regex", "keyword"],
            ),
            prompt=(
                "Briefly state the world's most-traded reserve currency. Use the "
                "three-letter ISO code somewhere in your answer."
            ),
            grader_spec=GraderSpec(
                type="regex", config={"pattern": r"\bUSD\b", "must_match": True}
            ),
        ),
        # ---------------------------- COMPOSITE / DEAL MEMO -----------------------
        Task(
            metadata=_meta(
                "q-deal-memo-001",
                "finance.banking.deal",
                difficulty="hard",
                tags=["composite", "investment-banking"],
            ),
            prompt=(
                "Compute the implied EV/EBITDA multiple for this deal:\n"
                "  Enterprise value: $1.2 billion\n"
                "  LTM EBITDA: $150 million\n"
                "State the multiple to one decimal place. In the SAME response, "
                "explicitly include the abbreviation EV/EBITDA somewhere in your text."
            ),
            grader_spec=GraderSpec(
                type="composite",
                config={
                    "components": [
                        {
                            "weight": 2,
                            "spec": {
                                "type": "regex",
                                "config": {
                                    "pattern": r"\b8\.0\b|\bx8\.0\b|\b8\.0x\b",
                                    "flags": "i",
                                },
                            },
                        },
                        {
                            "weight": 1,
                            "spec": {
                                "type": "regex",
                                "config": {"pattern": r"EV/EBITDA"},
                            },
                        },
                    ],
                    "pass_threshold": 0.7,
                },
            ),
            rubric=(
                "Correct multiple is 8.0x. Must show that number and reference "
                "the metric name."
            ),
        ),
    ]


# NOTE: A synthetic-data trading task was previously included here using
# SimulatedTradingTaskBuilder with a perfect-foresight oracle (sign of next
# bar's return). Real-world testing showed BOTH Haiku 4.5 and Sonnet 4.5
# fail it identically on the same window — because the underlying
# random-walk synthetic data has no signal to extract. The task was testing
# coin-flip luck, not reasoning. Dropped from the reference corpus on
# 2026-05-28. Replace with a deterministic trading scenario (e.g., explicit
# technical indicators with a textbook-correct response) if you want a real
# trading task in the corpus.


def build_taskset() -> TaskSet:
    tasks = hand_written_tasks()

    quick = ["q-bond-pv-001", "q-acct-ratio-001", "q-currency-001"]
    korean = [t.metadata.task_id for t in tasks if t.metadata.language == "ko"]
    auto_gradable = [
        t.metadata.task_id
        for t in tasks
        if t.grader_spec.type in {"exact_match", "regex", "composite"}
    ]
    all_ids = [t.metadata.task_id for t in tasks]

    return TaskSet(
        name="omniforge-reference-v0",
        version="0.1.0",
        description=(
            "Reference corpus shipped with omniforge v0.2. Ten tasks spanning "
            "English / Korean / Japanese finance and language reasoning, "
            "exercising every reference grader type. Demonstrates the "
            "platform — not a production eval corpus."
        ),
        tasks=tasks,
        splits={
            "quick": quick,
            "auto_gradable": auto_gradable,
            "korean": korean,
            "all": all_ids,
        },
        provenance={
            "license": "Apache-2.0",
            "build_script": "scripts/build_reference_corpus.py",
            "build_date": NOW.isoformat(),
            "note": "Reference / smoke-test corpus.",
        },
    )


def main() -> None:
    taskset = build_taskset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(json.loads(taskset.model_dump_json()), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(taskset.tasks)} tasks to {OUT_PATH}")
    print(f"  splits: {', '.join(taskset.splits)}")
    for s, ids in taskset.splits.items():
        print(f"    {s}: {len(ids)}")


if __name__ == "__main__":
    main()
