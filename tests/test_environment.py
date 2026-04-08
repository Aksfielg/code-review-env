import pytest
from models import CodeReviewAction, TaskDifficulty, ALL_TASK_IDS
from server.environment import CodeReviewEnvironment


def test_all_tasks_load():
    for task_id in ALL_TASK_IDS:
        env = CodeReviewEnvironment(task_id=task_id)
        obs = env.reset()
        assert obs.task_id == task_id
        assert obs.code_snippet != ""
        assert obs.attempt_number == 1


def test_easy_task_perfect_review():
    env = CodeReviewEnvironment(task_id="easy_01")
    env.reset()
    action = CodeReviewAction(
        review_text=(
            "Bug on line 6: ZeroDivisionError when numbers is an empty list. "
            "len(numbers) returns 0 causing division by zero. "
            "Fix: add `if not numbers: return 0.0` before the return statement."
        )
    )
    obs, reward, done = env.step(action)
    assert reward >= 0.7, f"Expected reward >= 0.7, got {reward}"


def test_reward_in_range():
    for task_id in ALL_TASK_IDS:
        env = CodeReviewEnvironment(task_id=task_id)
        env.reset()
        _, reward, _ = env.step(CodeReviewAction(review_text="No issues found in this code."))
        assert 0.0 <= reward <= 1.0, f"Reward out of range for {task_id}: {reward}"


def test_reset_clears_state():
    env = CodeReviewEnvironment(task_id="easy_01")
    env.reset()
    env.step(CodeReviewAction(review_text="Found a division by zero bug."))
    obs = env.reset()
    assert obs.attempt_number == 1
    assert obs.previous_reviews == []
    assert obs.last_reward == 0.0


def test_hard_task_security_reward():
    env = CodeReviewEnvironment(task_id="hard_01")
    env.reset()
    action = CodeReviewAction(
        review_text=(
            "Critical SQL injection vulnerability on line 8: user input is directly "
            "interpolated into the SQL query via f-string. An attacker can inject "
            "malicious SQL like DROP TABLE. Fix: use parameterized queries with ? placeholder. "
            "Bug 2: wrong operator => on line 12, should be <=. "
            "Bug 3: connection leak — conn.close() never called. Use with sqlite3.connect() context manager."
        )
    )
    _, reward, _ = env.step(action)
    assert reward >= 0.8, f"Expected high reward for complete hard review, got {reward}"


def test_improvement_bonus():
    env = CodeReviewEnvironment(task_id="medium_01")
    env.reset()
    _, r1, _ = env.step(CodeReviewAction(review_text="Found a KeyError when key is missing."))
    _, r2, _ = env.step(CodeReviewAction(
        review_text="Found KeyError on missing key AND typo emial should be email. "
                    "Fix: use .get() and correct the key name."
    ))
    assert r2 >= r1, "Improvement bonus should make second attempt score >= first"
