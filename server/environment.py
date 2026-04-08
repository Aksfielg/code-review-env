from __future__ import annotations
import math
from typing import Optional
from models import (
    TaskConfig, CodeReviewAction, CodeReviewObservation,
    CodeReviewReward, EpisodeStats, TaskDifficulty, TaskType,
    TASK_LIBRARY, get_task_by_id, ALL_TASK_IDS
)


class CodeReviewEnvironment:
    """
    OpenEnv-compliant Code Review environment.
    An AI agent reviews buggy Python code across multiple attempts,
    receiving shaped reward signals that reward finding bugs,
    suggesting fixes, catching security issues, and improving over attempts.
    """

    def __init__(self, task_id: str = "easy_01"):
        if task_id not in ALL_TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {ALL_TASK_IDS}")
        self.task_id = task_id
        self.task: Optional[TaskConfig] = None
        self.attempt = 0
        self.done = False
        self.previous_reviews: list[str] = []
        self.previous_rewards: list[float] = []
        self.best_reward: float = 0.0
        self._obs: Optional[CodeReviewObservation] = None

    #  Public API 

    def reset(self) -> CodeReviewObservation:
        self.task = get_task_by_id(self.task_id)
        self.attempt = 1
        self.done = False
        self.previous_reviews = []
        self.previous_rewards = []
        self.best_reward = 0.0
        self._obs = CodeReviewObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            task_type=self.task.task_type,
            code_snippet=self.task.code_snippet,
            task_description=self.task.task_description,
            attempt_number=1,
            max_attempts=self.task.max_attempts,
            previous_reviews=[],
            hint="",
            last_reward=0.0,
            is_done=False,
            feedback="",
        )
        return self._obs

    def step(self, action: CodeReviewAction) -> tuple[CodeReviewObservation, float, bool]:
        if self.done:
            return self._obs, 0.0, True

        review = action.review_text
        reward_obj = self._grade(review)
        scalar = reward_obj.total

        self.previous_reviews.append(review)
        self.previous_rewards.append(scalar)
        self.best_reward = max(self.best_reward, scalar)

        # Episode ends if agent solved it OR ran out of attempts
        self.done = scalar >= 0.85 or self.attempt >= self.task.max_attempts

        # Hint after N failed attempts
        hint = ""
        if (
            self.task.hint_after_attempt > 0
            and self.attempt >= self.task.hint_after_attempt
            and scalar < 0.5
        ):
            hint = self.task.hint_text

        # Feedback on what was missed
        feedback = reward_obj.reason if scalar < 0.85 else "Great review! All issues identified."

        self._obs = CodeReviewObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            task_type=self.task.task_type,
            code_snippet=self.task.code_snippet,
            task_description=self.task.task_description,
            attempt_number=self.attempt,
            max_attempts=self.task.max_attempts,
            previous_reviews=list(self.previous_reviews),
            hint=hint,
            last_reward=scalar,
            is_done=self.done,
            feedback=feedback,
        )

        self.attempt += 1
        return self._obs, scalar, self.done

    def state(self) -> CodeReviewObservation:
        if self._obs is None:
            return self.reset()
        return self._obs

    def episode_stats(self) -> EpisodeStats:
        total_bugs = len(self.task.required_keywords) if self.task else 0
        last_reward_obj = self._grade(self.previous_reviews[-1]) if self.previous_reviews else None
        bugs_found_count = 0
        if last_reward_obj and self.task:
            txt = self.previous_reviews[-1].lower()
            bugs_found_count = sum(
                1 for group in self.task.required_keywords
                if any(kw in txt for kw in group)
            )
        return EpisodeStats(
            task_id=self.task_id,
            total_attempts=len(self.previous_reviews),
            best_reward=self.best_reward,
            final_reward=self.previous_rewards[-1] if self.previous_rewards else 0.0,
            bugs_identified=bugs_found_count,
            total_bugs=total_bugs,
            security_found=bool(
                self.task and
                self.task.security_keywords and
                self.previous_reviews and
                any(kw in self.previous_reviews[-1].lower()
                    for kw in self.task.security_keywords)
            ),
            solved=self.best_reward >= 0.85,
        )

    #  Grader 

    def _grade(self, review_text: str) -> CodeReviewReward:
        txt = review_text.lower()
        task = self.task

        # 1. Bugs found - keyword groups, weighted by severity
        groups_matched = 0
        weighted_bug_score = 0.0
        total_weight = sum(task.severity_weights) if task.severity_weights else len(task.required_keywords)

        for i, group in enumerate(task.required_keywords):
            if any(kw in txt for kw in group):
                groups_matched += 1
                weight = task.severity_weights[i] if i < len(task.severity_weights) else 1.0
                weighted_bug_score += weight

        bugs_found = weighted_bug_score / total_weight if total_weight > 0 else 0.0
        bugs_found = round(min(bugs_found, 1.0), 4)

        # 2. Fix quality - did they suggest actual fixes?
        fix_count = sum(1 for kw in task.fix_keywords if kw in txt)
        fixes_quality = min(fix_count / max(len(task.fix_keywords) * 0.3, 1), 1.0)
        fixes_quality = round(fixes_quality, 4)

        # 3. Security check (hard/security tasks)
        security_check = 0.0
        if task.security_keywords:
            sec_count = sum(1 for kw in task.security_keywords if kw in txt)
            security_check = round(min(sec_count / max(len(task.security_keywords) * 0.3, 1), 1.0), 4)

        # 4. Performance check (performance tasks)
        performance_check = 0.0
        if task.performance_keywords:
            perf_count = sum(1 for kw in task.performance_keywords if kw in txt)
            performance_check = round(min(perf_count / max(len(task.performance_keywords) * 0.3, 1), 1.0), 4)

        # 5. Improvement bonus - reward if agent improves vs last attempt
        improvement_bonus = 0.0
        if self.previous_rewards:
            last = self.previous_rewards[-1]
            prev_txt = self.previous_reviews[-1].lower() if self.previous_reviews else ""
            prev_bugs = sum(1 for group in task.required_keywords if any(kw in prev_txt for kw in group))
            curr_bugs = groups_matched
            if curr_bugs > prev_bugs:
                improvement_bonus = round(0.1 * (curr_bugs - prev_bugs), 4)

        # 6. Verbosity penalty - too short = low effort
        verbosity_penalty = 0.0
        if len(review_text.strip()) < 50:
            verbosity_penalty = 0.15
        elif len(review_text.strip()) < 100:
            verbosity_penalty = 0.05

        # 7. Compute total by task type
        if task.task_type == TaskType.security:
            base = (bugs_found * 0.4) + (fixes_quality * 0.2) + (security_check * 0.4)
        elif task.task_type == TaskType.performance:
            base = (bugs_found * 0.4) + (fixes_quality * 0.2) + (performance_check * 0.4)
        else:
            base = (bugs_found * 0.65) + (fixes_quality * 0.35)

        total = round(min(max(base + improvement_bonus - verbosity_penalty, 0.0), 1.0), 4)

        # Human-readable reason
        bug_detail = f"{groups_matched}/{len(task.required_keywords)} bug groups found"
        reason = (
            f"{bug_detail} | bugs={bugs_found:.2f} fixes={fixes_quality:.2f} "
            f"security={security_check:.2f} perf={performance_check:.2f} "
            f"improvement_bonus={improvement_bonus:.2f} penalty={verbosity_penalty:.2f} "
            f" total={total:.4f}"
        )

        return CodeReviewReward(
            total=total,
            bugs_found=bugs_found,
            fixes_quality=fixes_quality,
            security_check=security_check,
            performance_check=performance_check,
            improvement_bonus=improvement_bonus,
            verbosity_penalty=verbosity_penalty,
            severity_score=weighted_bug_score / total_weight if total_weight > 0 else 0.0,
            reason=reason,
        )
