# omniforge

[![ci](https://github.com/grantcho123-web/omniforge/actions/workflows/ci.yml/badge.svg)](https://github.com/grantcho123-web/omniforge/actions/workflows/ci.yml)

A general-purpose platform for building, running, and grading LLM
evaluation tasks. Plug in any model, any grader, any task corpus,
export training-ready JSONL for any fine-tune pipeline.

## What this is

A small but complete framework for the work that goes into making an
LLM measurably better at something:

1. **Author tasks** with verifiable rubrics (numeric, regex, LLM-judged,
   composite, or human-graded).
2. **Run any model** over the task set — OpenAI, Anthropic, Upstage,
   or a custom adapter you write in ~30 lines.
3. **Grade attempts**, capture cost + latency + reasoning, dump full
   results to JSON.
4. **Export passing attempts** as JSONL in OpenAI fine-tune, Anthropic
   message, or generic format.

Built for myself as a learning project and a clean substrate for
experiments in eval design, RL data engineering, and LLM-as-judge
methodology.

## Architecture

```
                   ┌──────────────────────────┐
                   │  omniforge  CLI / API    │
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

Everything speaks four versioned types: `Task`, `TaskSet`, `Attempt`,
`GradingResult`. Adding a new model provider, grader, task domain, or
export format is a single self-contained file.

## Install

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e ".[dev,models]"

# Optional extras for the legacy RL trading code:
uv pip install -e ".[dev,models,data,train]"
```

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
  --model anthropic:claude-sonnet-4-5 \
  --split auto_gradable \
  --output runs/claude_v1.json

# Export passing attempts as OpenAI fine-tune training data
omniforge eval \
  --task-set corpora/reference-v0/manifest.json \
  --model anthropic:claude-sonnet-4-5 \
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
import omniforge.models   # registers reference adapters

taskset = TaskSet.model_validate_json(open("corpora/reference-v0/manifest.json").read())
model = make_model("anthropic:claude-sonnet-4-5")
runner = AttemptRunner(model)

for task in taskset.tasks:
    attempt = runner.run(task)
    if task.grader_spec.type != "llm_judge":  # llm_judge needs runtime callable
        grader = make_grader(task.grader_spec)
        result = grader.grade(task, attempt)
        print(task.metadata.task_id, result.score, result.passed)
```

## Reference corpus

`corpora/reference-v0/manifest.json` — nine tasks demonstrating every
grader type and three languages:

| Task ID | Domain | Language | Grader |
|---|---|---|---|
| q-bond-pv-001 | finance.quant.bonds | en | exact_match (numeric tolerance) |
| q-bs-call-001 | finance.quant.options | en | exact_match (numeric tolerance) |
| q-acct-ratio-001 | finance.accounting | en | exact_match (numeric tolerance) |
| q-ko-bond-001 | finance.korean.bonds | ko | exact_match (numeric tolerance) |
| q-ko-culture-001 | language.korean.culture | ko | human |
| q-ko-idiom-001 | language.korean.reasoning | ko | human |
| q-ja-math-001 | math.basic | ja | exact_match (numeric tolerance) |
| q-currency-001 | language.keyword | en | regex |
| q-deal-memo-001 | finance.banking.deal | en | composite |

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
  envs/                    Legacy RL env (single-asset trading), kept for sequential training
  data/                    OHLCV data sources
  backtest/                Walk-forward harness
  eval/                    Risk-adjusted financial metrics (Sharpe, Sortino, etc.)
  cli.py                   `omniforge` shell entry point

corpora/
  reference-v0/manifest.json  Reference corpus (10 tasks)

examples/
  quickstart_taskset.json  Tiny 5-task demo for first-time CLI users

scripts/
  build_reference_corpus.py  Regenerates the reference corpus
  spy_tearsheet.py           Legacy RL walk-forward demo on SPY
  train_ppo_demo.py          Legacy RL PPO training demo

tests/                     127 tests, pytest + ruff in CI on 3.10/3.11/3.12
```

## Roadmap

**v0.2** (current): platform foundation. Schemas, graders, model
adapters, runner, CLI, exports, reference corpus.

**v0.3** (planned): tool-using attempt protocols (so agents can browse,
run code, query APIs as part of a task), async parallelism over large
task sets, reviewer workbench v1 (real UI for the human-grader queue),
PyPI release.

**v0.4** (planned): synthetic task generation pipeline (LLM-drafted,
human-verified), reward-model training utilities, expanded adapter
catalog.

## License

Apache-2.0.
