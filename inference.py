"""
Code Review Environment - Inference Script
==========================================
OpenEnv Round 1 Hackathon submission - Team Prodigy
Runs all 5 tasks. Logs in mandatory START/STEP/END JSON format.
Runtime: < 20 minutes. Runs on vcpu=2, memory=8gb.
"""
from __future__ import annotations
import asyncio
import json
import os
from typing import List

from openai import OpenAI

#  MANDATORY variables - exact names, exact defaults 
# Checklist rule: API_BASE_URL and MODEL_NAME have defaults, HF_TOKEN does NOT
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")            # NO default - intentional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")    # optional

API_KEY = HF_TOKEN or "dummy-key"

#  Constants 
BENCHMARK              = "code-review-env"
MAX_STEPS              = 5
MAX_TOTAL_REWARD       = float(MAX_STEPS)           # 5.0
SUCCESS_SCORE_THRESHOLD = 0.6
ALL_TASKS              = ["easy_01", "easy_02", "medium_01", "hard_01", "perf_01", "logic_01"]


#  Mandatory log functions - field names must match exactly 

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type":  "START",
        "task":  task,
        "env":   env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    print(json.dumps({
        "type":   "STEP",
        "step":   step,
        "action": str(action)[:300],
        "reward": round(float(reward), 4),
        "done":   bool(done),
        "error":  error,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type":    "END",
        "success": bool(success),
        "steps":   steps,
        "score":   round(float(score), 4),
        "rewards": [round(float(r), 4) for r in rewards],
    }), flush=True)


#  Model interaction 

def get_model_review(
    client: OpenAI,
    code_snippet: str,
    task_description: str,
    hint: str,
    feedback: str,
    previous_reviews: List[str],
    history: List[str],
) -> str:
    """Ask the LLM to review the code and return the review text."""
    hint_block     = f"\nHINT: {hint}" if hint else ""
    feedback_block = f"\nFEEDBACK ON LAST ATTEMPT: {feedback}" if feedback else ""
    prev_block     = f"\nYOUR PREVIOUS REVIEW:\n{previous_reviews[-1][:400]}" if previous_reviews else ""
    history_str    = " | ".join(history[-3:]) if history else "first attempt"

    prompt = f"""You are a senior software engineer doing a thorough code review.

TASK: {task_description}{hint_block}{feedback_block}{prev_block}

CODE:
```python
{code_snippet}
```

History: {history_str}

Instructions:
- Find EVERY bug with its exact line number and root cause
- Rate severity: Critical / High / Medium / Low
- For security bugs: explicitly say "SQL injection" or "injection"
- For performance bugs: explicitly say "O(n^2)" or "quadratic complexity"
- Provide the exact corrected code for each fix
- Be specific and detailed - minimum 100 words

Write your complete code review:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        safe_response = {
            "review": "Error occurred but handled safely",
            "score": 0.0
        }
        return json.dumps(safe_response)


#  Per-task runner 

async def run_task(task_id: str) -> None:
    """Run one full episode for a single task_id."""
    from code_review_env import CodeReviewEnv, CodeReviewAction

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to running environment
    if LOCAL_IMAGE_NAME:
        env = await CodeReviewEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = await CodeReviewEnv.from_hub("rishabhisgod/code-review-env")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation

            review = get_model_review(
                client=client,
                code_snippet=obs.code_snippet,
                task_description=obs.task_description,
                hint=obs.hint,
                feedback=obs.feedback,
                previous_reviews=list(obs.previous_reviews),
                history=history,
            )

            result     = await env.step(CodeReviewAction(review_text=review))
            reward     = float(result.reward or 0.0)
            done       = bool(result.done)
            error      = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=review[:200], reward=reward, done=done, error=error)
            history.append(f"step={step} reward={reward:.3f}")

            if done:
                break

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


#  Main 

async def main() -> None:
    """Run all tasks. Total runtime must stay under 20 minutes."""
    try:
        print(json.dumps({
            "type": "INFO",
            "message": f"Starting {len(ALL_TASKS)} tasks",
            "tasks": ALL_TASKS,
            "model": MODEL_NAME,
        }), flush=True)

        for task_id in ALL_TASKS:
            await run_task(task_id)

        print(json.dumps({"type": "INFO", "message": "All tasks complete"}), flush=True)
    except Exception as e:
        safe_response = {
            "type": "END",
            "review": "Error occurred but handled safely",
            "score": 0.0,
            "error_detail": str(e)
        }
        print(json.dumps(safe_response), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        safe_response = {
            "type": "END",
            "review": "Error occurred but handled safely",
            "score": 0.0,
            "error_detail": str(e)
        }
        print(json.dumps(safe_response), flush=True)
