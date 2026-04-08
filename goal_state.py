"""
goal_state.py — Per-goal execution state management.

State lives in memory during a single goal's processing.
Arrays are bounded to prevent prompt bloat with Gemma 2B.
"""

import json


MAX_RESULTS = 2
MAX_ERRORS = 3
MAX_COMPLETED_ACTIONS = 20
MAX_SUMMARY_LENGTH = 500


def init_state(goal_text, memory=None):
    """
    Create a fresh execution state for a goal.

    Returns:
        dict with bounded arrays for steps, results, errors, etc.
    """
    return {
        "goal": goal_text,
        "memory": memory or {},
        "steps": [],
        "results": [],
        "completed_actions": [],
        "errors": [],
        "last_summary": "",
        "iteration": 0,
    }


def update_state(state, plan, execution_results):
    """
    Merge a plan and its execution results into state.
    Keeps arrays bounded to prevent prompt explosion.

    Args:
        state: current state dict
        plan: parsed controller output (mode, steps, complete, confidence)
        execution_results: list of {step, action, result} dicts or error dict
    """
    state["iteration"] += 1

    # Record the plan's steps
    if plan and plan.get("steps"):
        state["steps"] = plan["steps"]

    # Handle execution errors
    if isinstance(execution_results, dict) and execution_results.get("error"):
        state["errors"].append(
            {
                "iteration": state["iteration"],
                "error": execution_results.get("error"),
                "failed_step": execution_results.get("failed_step"),
            }
        )
        state["errors"] = state["errors"][-MAX_ERRORS:]
        return state

    # Handle successful results
    if isinstance(execution_results, list):
        for item in execution_results:
            action = item.get("action")
            if action:
                state["completed_actions"].append(action)

            result_summary = _summarize_result(item.get("result"))
            state["results"].append(
                {
                    "iteration": state["iteration"],
                    "step": item.get("step"),
                    "action": action,
                    "result": result_summary,
                }
            )

    # Bound arrays
    state["results"] = state["results"][-MAX_RESULTS:]
    state["completed_actions"] = state["completed_actions"][-MAX_COMPLETED_ACTIONS:]
    state["errors"] = state["errors"][-MAX_ERRORS:]

    # Update summary
    state["last_summary"] = _generate_summary(state)

    return state


def get_state_for_prompt(state):
    """
    Return a truncated view of state suitable for injecting into the LLM prompt.
    Keeps token count low for Gemma 2B.
    """
    return {
        "goal": state["goal"],
        "memory": state.get("memory", {}),
        "completed_actions": state["completed_actions"][-10:],
        "results": state["results"][-MAX_RESULTS:],
        "errors": state["errors"][-2:],
        "iteration": state["iteration"],
    }


def has_errors(state):
    """Check if the last iteration produced errors."""
    if not state["errors"]:
        return False
    last_error = state["errors"][-1]
    return last_error.get("iteration") == state["iteration"]


def has_results(state):
    """Check if any successful results exist."""
    return len(state["results"]) > 0


def get_completed_action_count(state):
    """Return the number of distinct completed actions."""
    return len(set(state["completed_actions"]))


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _summarize_result(result):
    """Truncate a single result value for state storage."""
    if result is None:
        return None

    if isinstance(result, str):
        if len(result) > MAX_SUMMARY_LENGTH:
            return result[:MAX_SUMMARY_LENGTH] + "...[truncated]"
        return result

    if isinstance(result, list):
        # Keep first 10 items, stringify
        items = result[:10]
        text = json.dumps(items, ensure_ascii=False)
        if len(text) > MAX_SUMMARY_LENGTH:
            return text[:MAX_SUMMARY_LENGTH] + "...[truncated]"
        return text

    if isinstance(result, dict):
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_SUMMARY_LENGTH:
            return text[:MAX_SUMMARY_LENGTH] + "...[truncated]"
        return text

    return str(result)


def _generate_summary(state):
    """Create a one-line summary of current progress."""
    parts = []
    parts.append(f"iteration={state['iteration']}")
    parts.append(f"actions={len(state['completed_actions'])}")
    parts.append(f"results={len(state['results'])}")
    parts.append(f"errors={len(state['errors'])}")
    return " | ".join(parts)
