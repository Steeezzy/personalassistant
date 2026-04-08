"""
goal_queue.py — Persistent goal queue backed by a JSON file.

Thread- / process-safe via fcntl file locking (macOS / Linux).
Goals are dequeued by priority (lower = higher priority), then by creation time.
"""

import fcntl
import json
import os
import uuid
from time import time

BASE_DIR = os.path.abspath(".")
QUEUE_FILE = os.path.join(BASE_DIR, "goal_queue.json")
STALE_TIMEOUT_SECONDS = 300  # 5 minutes — auto-reset stuck "running" goals
MAX_COMPLETED_GOALS = 50


def _lock_file(fh, exclusive=True):
    """Acquire an fcntl lock on an open file handle."""
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(fh, mode)


def _unlock_file(fh):
    """Release an fcntl lock."""
    fcntl.flock(fh, fcntl.LOCK_UN)


def _read_queue():
    """Read the queue file and return the list of goals."""
    if not os.path.exists(QUEUE_FILE):
        return []

    try:
        with open(QUEUE_FILE, "r", encoding="utf-8") as fh:
            _lock_file(fh, exclusive=False)
            try:
                data = json.load(fh)
            finally:
                _unlock_file(fh)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []
    return data


def _write_queue(goals):
    """Write the full goals list to disk with an exclusive lock."""
    with open(QUEUE_FILE, "w", encoding="utf-8") as fh:
        _lock_file(fh, exclusive=True)
        try:
            json.dump(goals, fh, ensure_ascii=False, indent=2)
        finally:
            _unlock_file(fh)


def _atomic_update(mutator_fn):
    """
    Read → mutate → write under an exclusive lock.
    *mutator_fn* receives the goals list and must return (goals, return_value).
    """
    # Use a separate lock file so we can open the data file for writing
    lock_path = QUEUE_FILE + ".lock"
    with open(lock_path, "a+") as lock_fh:
        _lock_file(lock_fh, exclusive=True)
        try:
            goals = _read_queue()
            goals, result = mutator_fn(goals)
            _write_queue(goals)
            return result
        finally:
            _unlock_file(lock_fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_goal(goal_text, priority=1, max_attempts=3):
    """
    Add a new goal to the queue.
    Returns the created goal dict.
    """
    now = time()
    goal = {
        "id": str(uuid.uuid4()),
        "goal": goal_text,
        "status": "pending",
        "priority": max(1, int(priority)),
        "created_at": now,
        "updated_at": now,
        "attempts": 0,
        "max_attempts": max(1, int(max_attempts)),
        "result": None,
        "error": None,
    }

    def _add(goals):
        goals.append(goal)
        return goals, goal

    return _atomic_update(_add)


def dequeue_next():
    """
    Find the highest-priority pending goal, mark it 'running', and return it.
    Also resets stale 'running' goals older than STALE_TIMEOUT_SECONDS.
    Returns the goal dict or None.
    """
    now = time()

    def _dequeue(goals):
        # Reset stale running goals
        for g in goals:
            if (
                g.get("status") == "running"
                and (now - g.get("updated_at", 0)) > STALE_TIMEOUT_SECONDS
            ):
                g["status"] = "pending"
                g["updated_at"] = now

        # Find best pending goal
        pending = [g for g in goals if g.get("status") == "pending"]
        if not pending:
            return goals, None

        # Sort: priority ascending, then created_at ascending
        pending.sort(key=lambda g: (g.get("priority", 99), g.get("created_at", 0)))
        chosen = pending[0]

        # Mark running
        for g in goals:
            if g["id"] == chosen["id"]:
                g["status"] = "running"
                g["updated_at"] = now
                g["attempts"] = g.get("attempts", 0) + 1
                chosen = g
                break

        return goals, chosen

    return _atomic_update(_dequeue)


def update_goal(goal_id, **fields):
    """
    Update specific fields on a goal by ID.
    Allowed fields: status, result, error, attempts, updated_at.
    Returns the updated goal or None if not found.
    """
    allowed_update_fields = {"status", "result", "error", "attempts", "updated_at"}

    def _update(goals):
        for g in goals:
            if g["id"] == goal_id:
                for key, value in fields.items():
                    if key in allowed_update_fields:
                        g[key] = value
                g["updated_at"] = time()

                # Trim completed/failed goals to keep queue bounded
                done = [
                    x
                    for x in goals
                    if x.get("status") in ("done", "failed")
                ]
                if len(done) > MAX_COMPLETED_GOALS:
                    # Remove oldest completed goals
                    done.sort(key=lambda x: x.get("updated_at", 0))
                    remove_ids = {x["id"] for x in done[: len(done) - MAX_COMPLETED_GOALS]}
                    goals = [x for x in goals if x["id"] not in remove_ids]

                return goals, g
        return goals, None

    return _atomic_update(_update)


def get_goal(goal_id):
    """Get a single goal by ID. Returns the goal dict or None."""
    goals = _read_queue()
    for g in goals:
        if g.get("id") == goal_id:
            return g
    return None


def list_goals(status=None):
    """
    List all goals, optionally filtered by status.
    Returns a list of goal dicts.
    """
    goals = _read_queue()
    if status:
        goals = [g for g in goals if g.get("status") == status]
    return goals


def delete_goal(goal_id):
    """
    Remove a goal from the queue.
    Only pending or done/failed goals can be deleted.
    Returns True if deleted, False otherwise.
    """

    def _delete(goals):
        for i, g in enumerate(goals):
            if g["id"] == goal_id:
                if g.get("status") == "running":
                    return goals, False  # Can't delete running goals
                goals.pop(i)
                return goals, True
        return goals, False

    return _atomic_update(_delete)


def mark_done(goal_id, result):
    """Convenience: mark a goal as done with its result."""
    return update_goal(goal_id, status="done", result=result)


def mark_failed(goal_id, error):
    """Convenience: mark a goal as failed with error info."""
    return update_goal(goal_id, status="failed", error=error)
