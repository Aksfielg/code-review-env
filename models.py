from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class TaskType(str, Enum):
    bug = "bug"               # logical bugs, crashes
    security = "security"     # SQL injection, XSS, unsafe eval
    performance = "performance"  # O(n^2) loops, memory leaks
    style = "style"           # dead code, bad naming, no docstrings
    logic = "logic"           # wrong algorithm, off-by-one errors


class CodeReviewAction(BaseModel):
    """What the agent submits - its full code review text."""
    review_text: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Full review: describe every bug found, its cause, severity, and fix.",
        examples=["Bug on line 4: division by zero when list is empty. Fix: check len() > 0 before dividing."]
    )


class CodeReviewObservation(BaseModel):
    """Everything the agent sees at each step."""
    task_id: str
    difficulty: TaskDifficulty
    task_type: TaskType
    code_snippet: str = Field(..., description="The Python code to review.")
    task_description: str = Field(..., description="What the agent must find and report.")
    attempt_number: int = Field(..., ge=1)
    max_attempts: int = Field(..., ge=1)
    previous_reviews: List[str] = Field(default_factory=list)
    hint: str = Field(default="")
    last_reward: float = Field(default=0.0)
    is_done: bool = Field(default=False)
    feedback: str = Field(default="", description="Feedback on previous attempt.")


class CodeReviewReward(BaseModel):
    """Detailed reward breakdown - judges love granular reward shaping."""
    total: float = Field(..., ge=0.0, le=1.0)
    bugs_found: float = Field(default=0.0, ge=0.0, le=1.0)
    fixes_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    security_check: float = Field(default=0.0, ge=0.0, le=1.0)
    performance_check: float = Field(default=0.0, ge=0.0, le=1.0)
    improvement_bonus: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Bonus if agent improves vs previous attempt.")
    verbosity_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Weighted by bug severity: security=1.0, perf=0.7, bug=0.5, style=0.3")
    reason: str


class EpisodeStats(BaseModel):
    """Full episode summary returned at done=True."""
    task_id: str
    total_attempts: int
    best_reward: float
    final_reward: float
    bugs_identified: int
    total_bugs: int
    security_found: bool
    solved: bool


class TaskConfig(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    task_type: TaskType
    code_snippet: str
    task_description: str
    required_keywords: List[List[str]]
    fix_keywords: List[str]
    security_keywords: List[str] = Field(default_factory=list)
    performance_keywords: List[str] = Field(default_factory=list)
    severity_weights: List[float] = Field(default_factory=list,
        description="Weight per bug group: 1.0=critical, 0.7=major, 0.3=minor")
    max_attempts: int = 3
    hint_after_attempt: int = 0
    hint_text: str = ""
    feedback_on_miss: str = ""


TASK_LIBRARY: List[TaskConfig] = [

    TaskConfig(
        task_id="easy_02",
        difficulty=TaskDifficulty.easy,
        task_type=TaskType.bug,
        code_snippet='''\
def divide(a, b):
    return a / b
''',
        task_description="Find bug when b = 0 and suggest fix.",
        required_keywords=[
            ["division by zero", "zero division", "divide by zero"]
        ],
        fix_keywords=["check b != 0", "if b == 0", "check b", "if not b"],
        severity_weights=[0.5],
        max_attempts=3,
        hint_after_attempt=1,
        hint_text="What happens if the denominator is zero?",
        feedback_on_miss="You missed the division by zero issue or didn't suggest a fix."
    ),
    #  EASY: 1 bug, obvious crash 
    TaskConfig(
        task_id="easy_01",
        difficulty=TaskDifficulty.easy,
        task_type=TaskType.bug,
        code_snippet='''\
def calculate_average(numbers):
    """Return the average of a list of numbers."""
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)   # BUG: ZeroDivisionError on empty list

result = calculate_average([])
print(result)
''',
        task_description=(
            "Review this Python function. Find ALL bugs (there is 1), "
            "explain what causes the crash, and provide a concrete code fix."
        ),
        required_keywords=[
            ["empty", "zero", "division", "len", "empty list",
             "zerodi", "divide by zero", "length is 0", "no elements",
             "zerodivisionerror", "zero division"],
        ],
        fix_keywords=["if", "check", "guard", "return", "handle",
                      "len(numbers) == 0", "not numbers", "raise", "try"],
        severity_weights=[0.5],
        max_attempts=3,
        hint_after_attempt=2,
        hint_text="Hint: what happens when you call calculate_average([]) with an empty list?",
        feedback_on_miss="Look at what happens when numbers=[] - trace through every line.",
    ),

    #  MEDIUM: 2 bugs, dict access 
    TaskConfig(
        task_id="medium_01",
        difficulty=TaskDifficulty.medium,
        task_type=TaskType.bug,
        code_snippet='''\
def get_user_info(users, user_id):
    """Return name and email for a given user_id."""
    user = users[user_id]          # BUG 1: KeyError if user_id missing
    name  = user["name"]
    email = user["emial"]          # BUG 2: typo - should be "email"
    return name, email

users_db = {"42": {"name": "Alice", "email": "alice@example.com"}}
print(get_user_info(users_db, "99"))   # KeyError here
print(get_user_info(users_db, "42"))   # Wrong key "emial" here
''',
        task_description=(
            "Review this function. Find ALL bugs (there are 2). "
            "Explain each bug, its line number, cause, and exact fix. "
            "Comment on code quality issues too."
        ),
        required_keywords=[
            ["keyerror", "key error", "missing key", "user_id not in",
             "does not exist", "get(", ".get", "key not found"],
            ["emial", "typo", "misspell", "wrong key", "\"email\"",
             "key name", "spelling"],
        ],
        fix_keywords=["get(", "if user_id", "try", "except",
                      "should be", "replace", "change", ".get("],
        severity_weights=[0.5, 0.5],
        max_attempts=3,
        hint_after_attempt=2,
        hint_text="Hint: check both the dictionary key lookup and the key names inside user dict.",
        feedback_on_miss="There are exactly 2 bugs - one about a missing key check, one about a typo.",
    ),

    #  HARD: 3 bugs + SQL injection security 
    TaskConfig(
        task_id="hard_01",
        difficulty=TaskDifficulty.hard,
        task_type=TaskType.security,
        code_snippet='''\
import sqlite3

def search_products(db_path, keyword, max_price=None):
    """Search products by keyword and optional price filter."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # BUG 1 + SECURITY: SQL injection via f-string
    query = f"SELECT * FROM products WHERE name LIKE \'%{keyword}%\'"

    if max_price is not None:
        # BUG 2: wrong operator => instead of <=
        query += f" AND price => {max_price}"

    cursor.execute(query)
    results = cursor.fetchall()
    # BUG 3: connection never closed - resource leak
    return results

print(search_products("shop.db", "laptop\'; DROP TABLE products; --", 1000))
''',
        task_description=(
            "This is a production database search function. "
            "Find ALL bugs (there are 3) AND the security vulnerability. "
            "Rate each issue by severity (Critical/High/Medium/Low). "
            "Provide parameterized query fix for the security issue."
        ),
        required_keywords=[
            ["sql injection", "injection", "f-string", "f\"", "f'",
             "parameterized", "unsafe", "malicious", "user input",
             "sanitize", "escape", "placeholder", "?", "format string"],
            ["=>", "wrong operator", "syntax error", "should be <=",
             "greater", "less than", "price filter", "invalid operator",
             "=<", "operator"],
            ["close", "conn.close", "leak", "resource leak", "finally",
             "context manager", "with sqlite3", "with conn",
             "connection not closed", "never closed", "with statement"],
        ],
        fix_keywords=["?", "parameterized", "conn.close", "with sqlite3",
                      "finally", "replace", "should be", "use", "<="],
        security_keywords=["sql injection", "injection", "sanitize",
                           "parameterized", "malicious", "unsafe",
                           "security", "vulnerable", "attack", "exploit",
                           "drop table", "user input", "critical"],
        severity_weights=[1.0, 0.7, 0.7],
        max_attempts=4,
        hint_after_attempt=2,
        hint_text=(
            "Hint: look at (1) how the SQL query is built using the keyword, "
            "(2) the comparison operator in the price filter, "
            "(3) what happens to the database connection after the function returns."
        ),
        feedback_on_miss="Focus on: SQL query construction, comparison operator, resource cleanup.",
    ),

    #  PERFORMANCE TASK 
    TaskConfig(
        task_id="perf_01",
        difficulty=TaskDifficulty.medium,
        task_type=TaskType.performance,
        code_snippet='''\
def find_duplicates(items):
    """Return list of duplicate values in items."""
    duplicates = []
    for i in range(len(items)):             # O(n^2) - very slow on large lists
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

# Also: this grows memory O(n) unnecessarily
big_list = list(range(10000)) + list(range(5000))
print(find_duplicates(big_list))   # This will be extremely slow
''',
        task_description=(
            "This function is catastrophically slow in production. "
            "Identify the performance issues, explain their time complexity, "
            "and provide an optimized O(n) replacement using appropriate data structures."
        ),
        required_keywords=[
            ["o(n^2)", "o(n2)", "nested loop", "quadratic", "slow",
             "n squared", "inefficient", "nested for", "complexity",
             "performance", "time complexity"],
        ],
        fix_keywords=["set", "dict", "seen", "counter", "collections",
                      "o(n)", "linear", "hash", "optimize", "efficient"],
        performance_keywords=["o(n^2)", "quadratic", "slow", "inefficient",
                               "nested", "complexity", "optimize", "o(n)"],
        severity_weights=[0.7],
        max_attempts=3,
        hint_after_attempt=2,
        hint_text="Hint: what data structure lets you check membership in O(1)?",
        feedback_on_miss="Count how many loops are nested - that determines time complexity.",
    ),

    #  LOGIC / OFF-BY-ONE 
    TaskConfig(
        task_id="logic_01",
        difficulty=TaskDifficulty.medium,
        task_type=TaskType.logic,
        code_snippet='''\
def binary_search(arr, target):
    """Search for target in sorted array. Returns index or -1."""
    left, right = 0, len(arr)   # BUG: should be len(arr) - 1

    while left < right:         # BUG: should be left <= right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid          # BUG: infinite loop - should be mid + 1
        else:
            right = mid - 1
    return -1

print(binary_search([1, 3, 5, 7, 9], 7))  # Should return 3, may loop forever
''',
        task_description=(
            "This binary search implementation has 3 logic bugs. "
            "Each one causes either wrong results or an infinite loop. "
            "Identify every bug, explain the correct logic, and provide the fixed function."
        ),
        required_keywords=[
            ["len(arr) - 1", "off by one", "off-by-one", "right boundary",
             "index out", "right = len", "should be len(arr)-1", "bounds"],
            ["left <= right", "loop condition", "while condition",
             "should be <=", "termination", "infinite loop", "<="],
            ["mid + 1", "infinite loop", "left = mid", "no progress",
             "stuck", "mid+1", "should be mid + 1"],
        ],
        fix_keywords=["mid + 1", "len(arr) - 1", "left <= right",
                      "fix", "should be", "change", "correct"],
        severity_weights=[0.6, 0.7, 0.8],
        max_attempts=4,
        hint_after_attempt=2,
        hint_text="Hint: trace through binary_search([1,3,5], 5) step by step - where does it go wrong?",
        feedback_on_miss="Check: (1) initial right boundary (2) while condition (3) left update.",
    ),
]


def get_task_by_id(task_id: str) -> Optional[TaskConfig]:
    for task in TASK_LIBRARY:
        if task.task_id == task_id:
            return task
    return None


def get_tasks_by_difficulty(difficulty: TaskDifficulty) -> List[TaskConfig]:
    return [t for t in TASK_LIBRARY if t.difficulty == difficulty]


def get_tasks_by_type(task_type: TaskType) -> List[TaskConfig]:
    return [t for t in TASK_LIBRARY if t.task_type == task_type]


ALL_TASK_IDS = [t.task_id for t in TASK_LIBRARY]
