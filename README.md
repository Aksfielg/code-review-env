---
title: Code Review Environment
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Code Review Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

## What This Solves

Manual code review consumes **15-20% of engineering time** in software companies.
Developers miss bugs under time pressure - especially security vulnerabilities.

This environment trains RL agents to do **automated Python code review**: find bugs,
identify security vulnerabilities (SQL injection), catch performance issues (O(n^2)),
and suggest concrete fixes - the same work a senior engineer does every day.

**Trained agents can plug into GitHub Actions / CI-CD pipelines.**

## Tasks

| Task ID | Type | Difficulty | What the agent must find |
|---------|------|------------|--------------------------|
| `easy_01` | Bug | Easy | ZeroDivisionError on empty list |
| `medium_01` | Bug | Medium | KeyError + typo in dict key |
| `hard_01` | Security | Hard | SQL injection + wrong operator + connection leak |
| `perf_01` | Performance | Medium | O(n^2) nested loop  O(n) fix |
| `logic_01` | Logic | Hard | 3 binary search bugs |

## Action Space

```python
CodeReviewAction(review_text="Bug on line 6: ZeroDivisionError when list is empty. Fix: check len() > 0.")
```

## Observation Space

```python
CodeReviewObservation(
    task_id="easy_01",
    difficulty="easy",
    task_type="bug",
    code_snippet="def calculate_average(numbers):\n    ...",
    task_description="Find all bugs. Explain cause. Suggest fix.",
    attempt_number=1,
    max_attempts=3,
    previous_reviews=[],
    hint="",
    last_reward=0.0,
    feedback="",
    is_done=False,
)
```

## Reward Function

Continuous [0.0, 1.0] - **never sparse**. Every attempt gets a signal.

```
Bug/Logic:     reward = bugs_found*0.65 + fixes_quality*0.35 + improvement_bonus  penalty
Security:      reward = bugs_found*0.40 + fixes_quality*0.20 + security_check*0.40
Performance:   reward = bugs_found*0.40 + fixes_quality*0.20 + perf_check*0.40
```

- `improvement_bonus` +0.1 per new bug found vs previous attempt
- `verbosity_penalty` 0.15 if review < 50 chars

## Quick Start

```bash
pip install openenv-core uv
git clone https://huggingface.co/spaces/rishabhisgod/code-review-env
cd code-review-env
uv sync
uv run server
```

```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/reset?task_id=easy_01"
```

## Web UI

Visit `/web` for an interactive browser-based code review playground.

## Docker

```bash
docker build -t code-review-env .
docker run -d -p 8000:8000 -e HF_TOKEN=your_token code-review-env
```

## Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_token"
python inference.py
```

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| easy_01 | gpt-4o-mini | 0.82 |
| medium_01 | gpt-4o-mini | 0.71 |
| hard_01 | gpt-4o-mini | 0.58 |
| perf_01 | gpt-4o-mini | 0.74 |
| logic_01 | gpt-4o-mini | 0.51 |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | **YES** | none | Hugging Face / API key |
| `LOCAL_IMAGE_NAME` | No | none | Local Docker image name |

## License

MIT - Team Prodigy
