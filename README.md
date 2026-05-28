# omniforge

[![ci](https://github.com/grantcho123-web/omniforge/actions/workflows/ci.yml/badge.svg)](https://github.com/grantcho123-web/omniforge/actions/workflows/ci.yml)

**RL data infrastructure for AI labs, built by [ebit](https://ebitglobal.ai).**

omniforge is the open-source platform behind ebit's RL data offering — the
same shape of product Scale AI, Mercor, and Surge sell to frontier AI labs,
focused on East Asian languages and finance and graded by domain experts.

## What this is

A frontier AI lab training the next generation of LLMs needs three things:

1. A **task corpus** — thousands of prompts with verifiable rubrics, written
   by people who actually do the job being evaluated.
2. A **grading workforce** — domain experts who can score open-ended
   responses where no auto-grader suffices.
3. A **platform** that lets the lab run their model over the corpus, capture
   graded attempts, and export training-ready data in their fine-tune format.

omniforge is open-source piece #3. Pieces #1 and #2 — the curated corpora and
the expert workforce — are what ebit sells, with the platform as the
visible credential.

## Positioning

| Category | Examples | ebit |
|---|---|---|
| Breadth-first US incumbents | Scale, Mercor, Surge, Invisible | We don't compete with them on breadth |
| Vertically focused on... | (varies) | **East Asian finance, language, and reasoning** |
| Native graders | (limited) | **Korean / Japanese / Chinese domain experts** |
| Customer focus | Mostly US frontier labs | **Asian frontier labs first** (Naver, Kakao, LG AI, Upstage, Krafton; Sakana, Preferred Networks; eligible Chinese labs) |

The wedge: every task is written and graded by people who have done the
work — at hedge funds, prop shops, banks, K-content studios, Korean
chip foundries. We don't do law. We don't do medicine. We do East Asian
finance and reasoning, and we do them better than anyone selling
breadth.

## Architecture

```
                   ┌──────────────────────────┐
                   │  omniforge  CLI / API     │
                   └────────────┬─────────────┘
                                │
                   ┌────────────┴─────────────┐
                   │  AttemptRunner           │
                   │  (load → ask → record)   │
                   └────────────┬─────────────┘
                                │
        ┌───────────────────────┴────────────────────┐
        │                                            │
┌───────┴──────────┐                       ┌─────────┴────────┐
│  ModelAdapter    │                       │  Grader          │
│  • OpenAI        │                       │  • ExactMatch    │
│  • Anthropic     │                       │  • Regex         │
│  • Upstage       │                       │  • LLMJudge      │
│  • Mock          │                       │  • Composite     │
└──────────────────┘                       │  • Human         │
                                           └──────────────────┘
                                ▲
                                │
                   ┌────────────┴─────────────┐
                   │  Task / TaskSet /        │
                   │  Attempt / GradingResult │
                   │  (versioned schemas)     │
                   └──────────────────────────┘
                                │
                   ┌────────────┴─────────────┐
                   │  Export adapters         │
                   │  • OpenAI fine-tune JSONL│
                   │  • Anthropic JSONL       │
                   │  • Generic JSONL         │
                   └──────────────────────────┘
```

Everything inside omniforge speaks four versioned types: `Task`, `TaskSet`,
`Attempt`, `GradingResult`. Adding a new model provider, grader, task
domain, or export format is a single self-contained file.

## Install

```bash
# From source (development):
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e ".[dev,models]"

# Add real data + RL training extras if you want them:
uv pip install -e ".[dev,models,data,train]"
```

PyPI release lands with v0.2.1.

## Quickstart

```bash
# Inspect the shipped reference corpus
omniforge inspect-taskset corpora/reference-v0/manifest.json

# Run a mock model against the quick split (offline, no API key needed)
omniforge eval \
  --task-set corpora/reference-v0/manifest.json \
  --model mock:default \
  --split quick

# Run a real model (needs ANTHROPIC_API_KEY)
omniforge eval \
  --task-set corpora/reference-v0/manifest.json \
  --model anthropic:claude-4.6-sonnet \
  --split auto_gradable \
  --output runs/claude_v1.json

# Export passing attempts as OpenAI fine-tune training data
omniforge eval \
  --task-set corpora/reference-v0/manifest.json \
  --model anthropic:claude-4.6-sonnet \
  --output runs/claude_v1.json \
  --export openai \
  --export-path runs/finetune.jsonl
```

Python API:

```python
from omniforge.core import TaskSet
from omniforge.core.runner import AttemptRunner
from omniforge.core.grader import make_grader
from omniforge.core.model import make_model
import omniforge.graders  # registers reference graders
import omniforge.models    # registers reference adapters

taskset = TaskSet.model_validate_json(open("corpora/reference-v0/manifest.json").read())
model = make_model("anthropic:claude-4.6-sonnet")
runner = AttemptRunner(model)

for task in taskset.tasks:
    attempt = runner.run(task)
    if task.grader_spec.type != "llm_judge":  # llm_judge needs runtime callable
        grader = make_grader(task.grader_spec)
        result = grader.grade(task, attempt)
        print(task.metadata.task_id, result.score, result.passed)
```

## What's in the reference corpus

`corpora/reference-v0/manifest.json` — ten tasks, demonstrating the platform:

| Task ID | Domain | Language | Grader |
|---|---|---|---|
| q-bond-pv-001 | finance.quant.bonds | en | exact_match (numeric tolerance) |
| q-bs-call-001 | finance.quant.options | en | exact_match (numeric tolerance) |
| q-acct-ratio-001 | finance.accounting | en | exact_match (numeric tolerance) |
| q-ko-bond-001 | finance.korean.bonds | ko | exact_match |
| q-ko-culture-001 | language.korean.culture | ko | human |
| q-ko-idiom-001 | language.korean.reasoning | ko | human |
| q-ja-math-001 | math.basic | ja | exact_match |
| q-currency-001 | language.keyword | en | regex |
| q-deal-memo-001 | finance.banking.deal | en | composite |
| q-trading-syn-001 | finance.trading.single_asset | en | exact_match (numeric tolerance) |

Production corpora are 100–10,000× larger and curated under separate
confidentiality agreements with domain experts on contract.

Regenerate with:

```bash
python scripts/build_reference_corpus.py
```

## Layout

```
src/omniforge/
  core/                    Schemas + abstract interfaces (Task, Grader, Model, Runner, Export)
  graders/                 Reference graders (exact, regex, llm_judge, composite, human)
  models/                  Reference adapters (OpenAI, Anthropic, Upstage, Mock)
  tasks/                   Concrete task factories (trading + future domains)
  envs/                    Legacy v0.1 RL env, available for sequential training
  data/                    OHLCV data sources
  backtest/                Walk-forward harness
  eval/                    Risk-adjusted financial metrics (Sharpe, Sortino, etc.)
  cli.py                   `omniforge` shell entry point

corpora/
  reference-v0/manifest.json  Shipped reference corpus (10 tasks)

examples/
  quickstart_taskset.json  Tiny 5-task demo for first-time CLI users

scripts/
  build_reference_corpus.py  Regenerates the reference corpus
  spy_tearsheet.py           Legacy v0.1 RL walk-forward demo
  train_ppo_demo.py          Legacy v0.1 RL training demo

tests/                     127 tests, pytest + ruff in CI on 3.10/3.11/3.12
```

## Status & roadmap

**v0.2** (current): platform foundation. Schemas, graders, model adapters,
runner, CLI, exports, reference corpus.

**v0.3**: tool-using attempt protocols (so agents can browse/run code/query
APIs as part of a task), async parallelism over large task sets, reviewer
workbench v1 (real UI for the human-grader queue), HyperCLOVA adapter,
PyPI release.

**v0.4**: synthetic task generation pipeline (LLM-drafted, human-verified —
the standard pattern for scaling to 5,000+ tasks per domain), reward-model
training utilities, lab-side integration adapters.

**v1.0**: API stability. First production corpora under customer contracts.

## License

Apache-2.0.
