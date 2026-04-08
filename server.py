"""
server.py — FastAPI server for Jarvis Agent.

Endpoints:
    POST /run          — Synchronous single-command execution (original)
    POST /goals        — Add a goal to the autonomous queue
    GET  /goals        — List all goals
    GET  /goals/{id}   — Get goal details + result
    DELETE /goals/{id} — Cancel / remove a goal

The autonomous worker (autonomous_worker.py) processes queued goals
in the background as a separate process.
"""

import json
import os
from time import time

from fastapi import FastAPI, Request

from shared import (
    MAX_ITERATIONS,
    MAX_STATE_RESULTS,
    ask_llm_with_retry,
    deterministic_plan_fallback,
    execute_steps,
    extract_goal,
    infer_forbidden_actions,
    infer_plan_gaps,
    is_truly_complete,
    load_memory,
    parse_controller_output,
    render_final_prompt,
    save_memory,
    summarize_execution_for_state,
    validate_steps,
)
from goal_queue import (
    add_goal,
    delete_goal,
    get_goal,
    list_goals,
    mark_done,
    mark_failed,
)

app = FastAPI()

API_KEY = os.getenv("JARVIS_API_KEY")
RATE_LIMIT_SECONDS = 1.0

last_call = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_client_id(request: Request):
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def is_rate_limited(client_id):
    now = time()
    if client_id in last_call and now - last_call[client_id] < RATE_LIMIT_SECONDS:
        return True

    last_call[client_id] = now
    return False


def check_auth(request: Request):
    """Validate API key. Returns error dict or None."""
    if not API_KEY:
        return {"error": "server_not_configured"}

    if request.headers.get("x-api-key") != API_KEY:
        return {"error": "unauthorized"}

    return None


# ---------------------------------------------------------------------------
# POST /run — Synchronous execution (original endpoint, unchanged logic)
# ---------------------------------------------------------------------------


@app.post("/run")
def run_task(data: dict, request: Request):
    auth_error = check_auth(request)
    if auth_error:
        return auth_error

    client_id = get_client_id(request)
    if is_rate_limited(client_id):
        return {"error": "rate_limited"}

    user_input = (data.get("input") or "").strip()
    if not user_input:
        return {"error": "Missing input"}

    state = {
        "goal": extract_goal(user_input),
        "memory": load_memory(),
        "completed_actions": [],
        "results": [],
    }

    # ── Deterministic-first: bypass LLM for well-known command patterns ───────
    fast_steps = deterministic_plan_fallback(user_input)
    if fast_steps:
        execution_result = execute_steps(fast_steps)
        if not (isinstance(execution_result, dict) and execution_result.get("error")):
            all_results = []
            for item in execution_result:
                all_results.append({
                    "iteration": 0,
                    "mode": "PLAN",
                    "confidence": 1.0,
                    "step": item["step"],
                    "action": item["action"],
                    "result": item["result"],
                })
            save_memory(state["goal"], [s.get("action") for s in fast_steps])
            return {"results": all_results}
    # ──────────────────────────────────────────────────────────────────────────

    previous_steps = []
    all_results = []
    raw_outputs = []

    for iteration in range(MAX_ITERATIONS):
        prompt = render_final_prompt(user_input, state, previous_steps)
        llm_output, attempts = ask_llm_with_retry(
            prompt,
            label_prefix=f"AGENT RAW {iteration}",
            max_attempts=2,
        )
        raw_outputs.append(attempts)

        if not llm_output:
            return {"error": "LLM failed", "raw": raw_outputs}

        controller = parse_controller_output(llm_output)
        if not controller:
            previous_steps = []
            state["results"] = [{"error": "Invalid controller output"}][
                -MAX_STATE_RESULTS:
            ]
            if iteration == MAX_ITERATIONS - 1:
                return {"error": "Invalid controller output", "raw": raw_outputs}
            continue

        mode = controller["mode"]
        complete = controller["complete"]
        confidence = controller["confidence"]
        steps = controller["steps"]
        thought = controller.get("thought")

        # ── Conversational short-circuit ──────────────────────────────────────
        # If the model returned no steps but gave us a thought, just speak it.
        if not steps and thought:
            return {"results": [], "thought": thought}
        # ──────────────────────────────────────────────────────────────────────

        valid, reason = validate_steps(steps)
        gaps = infer_plan_gaps(user_input, steps) if complete else []
        forbidden = infer_forbidden_actions(user_input, steps)
        if not valid or gaps or forbidden:
            fallback_steps = deterministic_plan_fallback(user_input)
            fallback_valid, fallback_reason = validate_steps(fallback_steps)
            fallback_forbidden = infer_forbidden_actions(user_input, fallback_steps)
            fallback_gaps = infer_plan_gaps(user_input, fallback_steps)
            if fallback_valid and not fallback_forbidden and not fallback_gaps:
                steps = fallback_steps
                mode = "REPAIR"
                complete = True
                confidence = min(confidence, 0.6)
                execution_result = execute_steps(steps)
                if isinstance(execution_result, dict) and execution_result.get(
                    "error"
                ):
                    return {
                        "error": "Execution failed",
                        "details": execution_result,
                        "results": all_results,
                        "raw": raw_outputs,
                    }

                for item in execution_result:
                    all_results.append(
                        {
                            "iteration": iteration,
                            "mode": mode,
                            "confidence": confidence,
                            "step": item["step"],
                            "action": item["action"],
                            "result": item["result"],
                        }
                    )

                state["completed_actions"].extend(
                    [
                        item.get("action")
                        for item in execution_result
                        if item.get("action")
                    ]
                )
                state["results"] = (
                    state["results"]
                    + summarize_execution_for_state(execution_result)
                )[-MAX_STATE_RESULTS:]

                save_memory(state["goal"], state["completed_actions"])
                return {"results": all_results, "fallback": True, "thought": thought}

            error_reason = reason if not valid else "Incomplete or unsafe plan"
            previous_steps = steps
            state["results"] = [
                {
                    "error": error_reason,
                    "gaps": gaps,
                    "forbidden": forbidden,
                }
            ][-MAX_STATE_RESULTS:]
            if iteration == MAX_ITERATIONS - 1:
                return {
                    "error": error_reason,
                    "gaps": gaps,
                    "forbidden": forbidden,
                    "raw": raw_outputs,
                }
            continue

        execution_result = execute_steps(steps)
        if isinstance(execution_result, dict) and execution_result.get("error"):
            return {
                "error": "Execution failed",
                "details": execution_result,
                "results": all_results,
                "raw": raw_outputs,
            }

        for item in execution_result:
            all_results.append(
                {
                    "iteration": iteration,
                    "mode": mode,
                    "confidence": confidence,
                    "step": item["step"],
                    "action": item["action"],
                    "result": item["result"],
                }
            )

        state["completed_actions"].extend(
            [
                item.get("action")
                for item in execution_result
                if item.get("action")
            ]
        )
        state["results"] = (
            state["results"] + summarize_execution_for_state(execution_result)
        )[-MAX_STATE_RESULTS:]

        if complete and is_truly_complete(user_input, steps):
            save_memory(state["goal"], state["completed_actions"])
            return {"results": all_results, "thought": thought}

        if complete:
            previous_steps = steps
            state["results"] = (
                state["results"]
                + [{"error": "Completion guard rejected complete=true"}]
            )[-MAX_STATE_RESULTS:]
            if iteration == MAX_ITERATIONS - 1:
                return {
                    "error": "Completion guard rejected complete=true",
                    "results": all_results,
                    "raw": raw_outputs,
                }
            continue

        previous_steps = steps

    return {
        "error": "Iteration limit reached",
        "results": all_results,
        "raw": raw_outputs,
    }


# ---------------------------------------------------------------------------
# POST /goals — Add a goal to the autonomous queue
# ---------------------------------------------------------------------------


@app.post("/goals")
def create_goal(data: dict, request: Request):
    auth_error = check_auth(request)
    if auth_error:
        return auth_error

    client_id = get_client_id(request)
    if is_rate_limited(client_id):
        return {"error": "rate_limited"}

    goal_text = (data.get("goal") or "").strip()
    if not goal_text:
        return {"error": "Missing 'goal' field"}

    priority = data.get("priority", 1)
    max_attempts = data.get("max_attempts", 3)

    try:
        priority = max(1, int(priority))
    except (TypeError, ValueError):
        priority = 1

    try:
        max_attempts = max(1, min(10, int(max_attempts)))
    except (TypeError, ValueError):
        max_attempts = 3

    goal = add_goal(goal_text, priority=priority, max_attempts=max_attempts)

    return {
        "success": True,
        "message": "Goal queued for autonomous processing",
        "goal": goal,
    }


# ---------------------------------------------------------------------------
# GET /goals — List all goals
# ---------------------------------------------------------------------------


@app.get("/goals")
def get_goals(request: Request, status: str = None):
    auth_error = check_auth(request)
    if auth_error:
        return auth_error

    goals = list_goals(status=status)

    return {
        "success": True,
        "goals": goals,
        "count": len(goals),
    }


# ---------------------------------------------------------------------------
# GET /goals/{goal_id} — Get goal details
# ---------------------------------------------------------------------------


@app.get("/goals/{goal_id}")
def get_goal_details(goal_id: str, request: Request):
    auth_error = check_auth(request)
    if auth_error:
        return auth_error

    goal = get_goal(goal_id)
    if not goal:
        return {"error": "Goal not found"}

    return {
        "success": True,
        "goal": goal,
    }


# ---------------------------------------------------------------------------
# DELETE /goals/{goal_id} — Cancel / remove a goal
# ---------------------------------------------------------------------------


@app.delete("/goals/{goal_id}")
def remove_goal(goal_id: str, request: Request):
    auth_error = check_auth(request)
    if auth_error:
        return auth_error

    deleted = delete_goal(goal_id)
    if not deleted:
        return {"error": "Goal not found or currently running"}

    return {
        "success": True,
        "message": "Goal deleted",
    }
