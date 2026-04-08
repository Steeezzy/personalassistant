"""
evaluator.py — Deterministic + LLM evaluator for goal completion.

Rules run first (fast, reliable).
LLM is only consulted when rules are ambiguous.
"""

import re

from shared import ask_llm, extract_json


# ---------------------------------------------------------------------------
# Decision Result
# ---------------------------------------------------------------------------


def _decision(complete=False, retry=False, replan=False, abort=False, reason=""):
    """Build a standardized decision dict."""
    return {
        "complete": complete,
        "retry": retry,
        "replan": replan,
        "abort": abort,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Rule-Based Checks
# ---------------------------------------------------------------------------


def _check_abort(state, goal_obj=None):
    """Abort if max iterations or max attempts exceeded."""
    max_iters = 3
    if state["iteration"] >= max_iters:
        return True, "Max iterations reached"

    if goal_obj and goal_obj.get("attempts", 0) >= goal_obj.get("max_attempts", 3):
        return True, "Max attempts exceeded"

    return False, ""


def _check_errors(state):
    """If the last iteration had errors, recommend retry or replan."""
    if not state.get("errors"):
        return None

    last_error = state["errors"][-1]
    if last_error.get("iteration") != state["iteration"]:
        return None  # Error is from a previous iteration, not current

    error_count = len(state["errors"])

    if error_count >= 2:
        return _decision(replan=True, reason=f"Multiple errors ({error_count}), replanning")

    return _decision(retry=True, reason=f"Error in iteration {state['iteration']}")


def _check_no_results(state):
    """If no results at all after execution, replan."""
    if state["iteration"] > 0 and not state.get("results"):
        return _decision(replan=True, reason="No results after execution")
    return None


def _check_multi_task(goal_text, state):
    """
    Compound goals (containing 'and') need at least 2 completed actions.
    This is the most common failure mode with Gemma 2B.
    """
    g = goal_text.lower()
    if " and " in g and len(set(state.get("completed_actions", []))) < 2:
        return _decision(
            replan=True,
            reason="Compound goal ('and') but fewer than 2 distinct actions completed",
        )
    return None


def _check_basic_completion(goal_text, state):
    """
    Simple heuristic: if we have results and no recent errors,
    the goal is likely complete for simple single-action goals.
    """
    g = goal_text.lower()

    # Single-action goals
    if " and " not in g and state.get("results") and not _has_recent_error(state):
        return _decision(complete=True, reason="Single goal with results, no errors")

    # Compound goals with enough actions
    if " and " in g:
        distinct_actions = len(set(state.get("completed_actions", [])))
        if distinct_actions >= 2 and state.get("results") and not _has_recent_error(state):
            return _decision(
                complete=True,
                reason=f"Compound goal with {distinct_actions} distinct actions completed",
            )

    return None


def _has_recent_error(state):
    """Check if the most recent iteration produced an error."""
    if not state.get("errors"):
        return False
    return state["errors"][-1].get("iteration") == state["iteration"]


# ---------------------------------------------------------------------------
# LLM Evaluation (last resort)
# ---------------------------------------------------------------------------


EVAL_PROMPT = """
You are a task completion evaluator.

GOAL: {goal}

COMPLETED ACTIONS: {actions}

RESULTS SUMMARY: {results}

ERRORS: {errors}

---

Is this goal fully complete based on the results?

Respond ONLY with valid JSON:
{{"complete": true}} or {{"complete": false}}
"""


def _llm_evaluate(goal_text, state):
    """Ask the LLM if the goal is complete. Fallback: assume incomplete."""
    import json

    prompt = EVAL_PROMPT.format(
        goal=goal_text,
        actions=json.dumps(state.get("completed_actions", []), ensure_ascii=False),
        results=json.dumps(state.get("results", []), ensure_ascii=False),
        errors=json.dumps(state.get("errors", []), ensure_ascii=False),
    )

    response = ask_llm(prompt, label="EVAL")
    if not response:
        return None

    data = extract_json(response)
    if isinstance(data, dict) and isinstance(data.get("complete"), bool):
        return data["complete"]

    return None


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------


def evaluate(goal_text, state, goal_obj=None):
    """
    Evaluate whether a goal is complete.

    Priority order:
    1. Abort check (max iterations / attempts)
    2. Error check
    3. No-results check
    4. Multi-task check
    5. Basic completion heuristic
    6. LLM evaluation (only if ambiguous)

    Args:
        goal_text: the goal string
        state: current GoalState dict
        goal_obj: optional goal queue entry (for attempt tracking)

    Returns:
        Decision dict: {complete, retry, replan, abort, reason}
    """

    # 1. Abort check
    should_abort, abort_reason = _check_abort(state, goal_obj)
    if should_abort:
        return _decision(abort=True, reason=abort_reason)

    # 2. Error check
    error_decision = _check_errors(state)
    if error_decision:
        return error_decision

    # 3. No results check
    no_results = _check_no_results(state)
    if no_results:
        return no_results

    # 4. Multi-task check
    multi_task = _check_multi_task(goal_text, state)
    if multi_task:
        return multi_task

    # 5. Basic completion heuristic
    basic = _check_basic_completion(goal_text, state)
    if basic:
        return basic

    # 6. LLM evaluation (ambiguous case)
    llm_complete = _llm_evaluate(goal_text, state)
    if llm_complete is True:
        return _decision(complete=True, reason="LLM confirmed completion")
    if llm_complete is False:
        return _decision(replan=True, reason="LLM says goal is incomplete")

    # 7. Fallback: continue if we still have iterations left
    return _decision(replan=True, reason="Ambiguous state, replanning")
