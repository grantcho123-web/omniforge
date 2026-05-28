"""Build the v0 'discriminating' task corpus.

Produces ``corpora/discriminating-v0/manifest.json`` — a harder corpus
intended to surface real disagreement between frontier models. Where
reference-v0 is the "platform works" demo (every task is settled
textbook material that any modern LLM solves), this corpus targets:

- multi-step quantitative reasoning where one early mistake compounds
- specialized formula recall (Macaulay duration, Black-Scholes put,
  optimal hedge ratio) that smaller models sometimes garble
- probability traps (base-rate fallacy, Bayesian inversion, birthday
  paradox) where even strong models can flub the framing
- multilingual reasoning that combines language fluency with domain
  knowledge (K-IFRS related-party rules, Japanese keigo, Chinese
  financial vocabulary)

Run from the project root:

    python scripts/build_discriminating_corpus.py

All answers are validated by hand below. If a model gets a task right
and the grader marks it wrong, suspect the grader before the model.
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

OUT_DIR = Path("corpora/discriminating-v0")
OUT_PATH = OUT_DIR / "manifest.json"

NOW = datetime(2026, 5, 28, tzinfo=timezone.utc)


def _meta(
    task_id: str,
    domain: str,
    difficulty: str = "hard",
    tags: list[str] | None = None,
    language: str = "en",
) -> TaskMetadata:
    return TaskMetadata(
        task_id=task_id,
        version="0.1.0",
        domain=domain,
        difficulty=difficulty,
        author="omniforge-discriminating-v0",
        created_at=NOW,
        tags=tags or [],
        language=language,
    )


# ==================================================================== TASKS


def finance_tasks() -> list[Task]:
    return [
        # --------------------------------------------------------- YTM solver
        Task(
            metadata=_meta(
                "f-ytm-001",
                "finance.quant.bonds",
                tags=["ytm", "iterative"],
            ),
            prompt=(
                "A 10-year corporate bond pays a $40 annual coupon on a "
                "$1,000 face value. It currently trades at $920. What is "
                "its yield to maturity? Answer as a percentage to 2 decimal "
                "places (e.g., 5.25)."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.10}
            ),
            reference_answer="5.04",
        ),
        # --------------------------------------------------- Beta from returns
        Task(
            metadata=_meta(
                "f-beta-001",
                "finance.quant.statistics",
                tags=["beta", "covariance"],
            ),
            prompt=(
                "Compute the beta of asset A relative to market M, given "
                "these 5 monthly returns:\n"
                "  A: [0.02, -0.03, 0.05, 0.01, -0.02]\n"
                "  M: [0.01, -0.02, 0.03, 0.005, -0.015]\n"
                "Beta = Cov(A, M) / Var(M). Answer to 2 decimal places."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.10}
            ),
            reference_answer="1.59",
        ),
        # ----------------------------------------------- Multi-period DCF
        Task(
            metadata=_meta(
                "f-dcf-001",
                "finance.quant.valuation",
                tags=["dcf", "multi-period"],
            ),
            prompt=(
                "A project's projected free cash flows are: Year 1 = $100M, "
                "Year 2 = $120M, Year 3 = $150M. The terminal value at end of "
                "Year 3 is $1,500M. The discount rate is 10%. Compute the "
                "present value of the project in $M, rounded to the nearest "
                "whole number. Answer with just the number (no units)."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 5.0}
            ),
            reference_answer="1430",
        ),
        # --------------------------------------------- Black-Scholes put
        Task(
            metadata=_meta(
                "f-bs-put-001",
                "finance.quant.options",
                tags=["black-scholes", "put"],
            ),
            prompt=(
                "Compute the Black-Scholes European put option price with:\n"
                "  Spot S = 100\n"
                "  Strike K = 110\n"
                "  Risk-free rate r = 0.05 (continuous compounding)\n"
                "  Time to maturity T = 1 year\n"
                "  Volatility sigma = 0.25\n"
                "  No dividends.\n"
                "Round to 2 decimal places. Answer with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.5}
            ),
            reference_answer="12.66",
        ),
        # ---------------------------------------------- Macaulay duration
        Task(
            metadata=_meta(
                "f-duration-001",
                "finance.quant.bonds",
                tags=["duration", "macaulay"],
            ),
            prompt=(
                "Compute the Macaulay duration of a 3-year bond paying $50 "
                "annual coupons on $1,000 face value, with the market yield "
                "at 5%. Macaulay duration in years, rounded to 2 decimal "
                "places. Answer with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.05}
            ),
            reference_answer="2.86",
        ),
        # ------------------------------------------ Optimal hedge ratio
        Task(
            metadata=_meta(
                "f-hedge-001",
                "finance.quant.hedging",
                tags=["hedge-ratio"],
            ),
            prompt=(
                "An airline wants to hedge jet fuel cost exposure using "
                "crude oil futures. Given:\n"
                "  Std dev of jet fuel returns = 0.15\n"
                "  Std dev of crude oil returns = 0.12\n"
                "  Correlation between jet fuel and crude = 0.85\n"
                "What is the optimal hedge ratio (units of crude oil per "
                "unit of jet fuel)? Round to 2 decimal places. Answer with "
                "just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.02}
            ),
            reference_answer="1.06",
        ),
    ]


def puzzle_tasks() -> list[Task]:
    return [
        # ----------------------------------------- 5-door Monty Hall
        Task(
            metadata=_meta(
                "p-monty-5-001",
                "logic.probability",
                tags=["monty-hall", "probability"],
            ),
            prompt=(
                "In a 5-door Monty Hall variant: one door hides a car, four "
                "hide goats. You pick a door. The host (who knows what's "
                "behind every door) opens 3 of the remaining 4 doors, all "
                "revealing goats. You can keep your original choice or "
                "switch to the one unopened door you didn't pick. If you "
                "switch, what is your probability of winning the car? "
                "Answer as a decimal to 3 decimal places."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.01}
            ),
            reference_answer="0.800",
        ),
        # ----------------------------------------- Fence posts counting
        Task(
            metadata=_meta(
                "p-fence-001",
                "logic.counting",
                difficulty="medium",
                tags=["counting"],
            ),
            prompt=(
                "A square fence has sides of 10 meters each. Wooden posts "
                "are placed every 2 meters along each side, including at "
                "every corner. How many posts are there in total? Answer "
                "with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.5}
            ),
            reference_answer="20",
        ),
        # ----------------------------------------- Knight corner-to-corner
        Task(
            metadata=_meta(
                "p-knight-001",
                "logic.chess",
                tags=["chess", "shortest-path"],
            ),
            prompt=(
                "A knight on a standard 8x8 chessboard starts at corner a1. "
                "What is the minimum number of moves required to reach the "
                "opposite corner h8? A knight moves in an L-shape: 2 squares "
                "in one direction (horizontal or vertical) and 1 square in "
                "the perpendicular direction. Answer with just the number."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.5}
            ),
            reference_answer="6",
        ),
    ]


def probability_tasks() -> list[Task]:
    return [
        # ----------------------------------------- Medical test base rate
        Task(
            metadata=_meta(
                "pr-medical-001",
                "logic.probability.bayes",
                tags=["base-rate", "bayes"],
            ),
            prompt=(
                "A medical test for a rare disease has 99% sensitivity (true "
                "positive rate) and 99% specificity (true negative rate). "
                "The disease occurs in 1 out of every 10,000 people. A "
                "randomly selected person from the general population tests "
                "positive. What is the probability they actually have the "
                "disease? Answer as a decimal to 4 decimal places."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.001}
            ),
            reference_answer="0.0098",
        ),
        # ----------------------------------------- Birthday paradox
        Task(
            metadata=_meta(
                "pr-birthday-001",
                "logic.probability",
                tags=["birthday-paradox"],
            ),
            prompt=(
                "In a group of 23 people, what is the probability that at "
                "least two share the same birthday? Assume 365 days, "
                "uniformly distributed, no leap years. Answer as a decimal "
                "to 3 decimal places."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.01}
            ),
            reference_answer="0.507",
        ),
        # ----------------------------------------- Spam classifier Bayes
        Task(
            metadata=_meta(
                "pr-bayes-001",
                "logic.probability.bayes",
                tags=["bayes", "spam"],
            ),
            prompt=(
                "10% of emails arriving in an inbox are spam. 90% of spam "
                "emails contain the word 'free'. 5% of legitimate (non-spam) "
                "emails contain the word 'free'. An email contains the word "
                "'free'. What is the probability that it is spam? Answer as "
                "a decimal to 3 decimal places."
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 0.01}
            ),
            reference_answer="0.667",
        ),
    ]


def multilingual_tasks() -> list[Task]:
    return [
        # ------------------------------------------- K-IFRS related party
        Task(
            metadata=_meta(
                "ko-related-001",
                "finance.korean.kifrs",
                tags=["korean", "kifrs", "multiple-choice"],
                language="ko",
            ),
            prompt=(
                "한국채택국제회계기준(K-IFRS) 제1024호 '특수관계자 공시'에 따를 때, "
                "다음 중 회사의 '특수관계자'에 일반적으로 해당하지 않는 사람은 누구입니까?\n"
                "A) 회사의 최대주주\n"
                "B) 회사 주요 경영진의 배우자\n"
                "C) 회사의 일반 직원\n"
                "D) 회사의 종속기업의 임원\n"
                "정답의 알파벳 한 글자만 답하세요."
            ),
            grader_spec=GraderSpec(type="exact_match"),
            reference_answer="C",
        ),
        # ------------------------------------------- Japanese keigo
        Task(
            metadata=_meta(
                "ja-keigo-001",
                "language.japanese.keigo",
                tags=["japanese", "keigo", "multiple-choice"],
                language="ja",
            ),
            prompt=(
                "ビジネスメールで、上司から送付された資料を「読んで内容を確認した」と "
                "返信する際に、最も丁寧で適切な表現はどれですか。\n"
                "A) 確認しました。\n"
                "B) 確認させていただきました。\n"
                "C) 確認致しました。\n"
                "D) 拝見いたしました。\n"
                "正解のアルファベット一文字だけで答えてください。"
            ),
            grader_spec=GraderSpec(type="exact_match"),
            reference_answer="D",
        ),
        # ------------------------------------------- Chinese P/E ratio
        Task(
            metadata=_meta(
                "zh-pe-001",
                "finance.chinese.ratios",
                tags=["chinese", "pe-ratio"],
                language="zh-CN",
            ),
            prompt=(
                "某上市公司去年净利润为人民币5000万元，发行的总股本为1亿股，"
                "目前每股股价为30元。请计算该公司的市盈率（P/E ratio），"
                "保留一位小数。只回答数字。"
            ),
            grader_spec=GraderSpec(
                type="exact_match", config={"numeric_tolerance": 1.0}
            ),
            reference_answer="60",
        ),
    ]


# ===================================================================== BUILD


def build_taskset() -> TaskSet:
    tasks = (
        finance_tasks()
        + puzzle_tasks()
        + probability_tasks()
        + multilingual_tasks()
    )

    by_domain = {
        "finance": [t.metadata.task_id for t in finance_tasks()],
        "puzzles": [t.metadata.task_id for t in puzzle_tasks()],
        "probability": [t.metadata.task_id for t in probability_tasks()],
        "multilingual": [t.metadata.task_id for t in multilingual_tasks()],
    }
    all_ids = [t.metadata.task_id for t in tasks]
    # A small "quick" split for fast iteration during corpus development.
    quick = ["f-dcf-001", "p-fence-001", "pr-bayes-001"]

    return TaskSet(
        name="omniforge-discriminating-v0",
        version="0.1.0",
        description=(
            "Harder 15-task corpus intended to surface real disagreement "
            "between frontier models. Mix of multi-step quant, formula "
            "recall, probability traps, and multilingual reasoning that "
            "combines language fluency with domain knowledge."
        ),
        tasks=tasks,
        splits={
            "quick": quick,
            "finance": by_domain["finance"],
            "puzzles": by_domain["puzzles"],
            "probability": by_domain["probability"],
            "multilingual": by_domain["multilingual"],
            "all": all_ids,
        },
        provenance={
            "license": "Apache-2.0",
            "build_script": "scripts/build_discriminating_corpus.py",
            "build_date": NOW.isoformat(),
            "note": (
                "Tasks hand-authored with answers verified analytically. "
                "If a model is marked wrong and the math/logic checks out, "
                "suspect the grader spec (extraction, tolerance) before "
                "assuming the model failed."
            ),
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
