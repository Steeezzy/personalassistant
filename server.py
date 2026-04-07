from fastapi import FastAPI, Request
import json
import os
import re
import subprocess
import requests
from time import time

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma:2b"
API_KEY = os.getenv("JARVIS_API_KEY")

BASE_DIR = os.path.abspath(".")
MAX_STEPS = 5
MAX_FILE_SIZE = 10000
MAX_ITERATIONS = 3
MAX_STATE_RESULTS = 2
MAX_STATE_TEXT = 500
RATE_LIMIT_SECONDS = 1.0
MAX_MEMORY_ITEMS = 10
MAX_SEARCH_RESULTS = 20
MAX_SEARCH_FILE_BYTES = 50000
MEMORY_FILE = os.path.join(BASE_DIR, ".jarvis_memory.json")

ALLOWED_COMMANDS = ["ls", "pwd"]
ALLOWED_ACTIONS = ["list_files", "read_file", "run_command", "search_project"]
ALLOWED_MODES = ["PLAN", "REPAIR", "CONTINUE"]

last_call = {}

ALLOWED_FIELDS = {
    "list_files": {"action", "path"},
    "read_file": {"action", "path"},
    "run_command": {"action", "cmd"},
    "search_project": {"action", "query"},
}


FINAL_PROMPT = """
You are an autonomous system controller.

Your role is to:
- Understand the user goal
- Use available tools to complete the task
- Track progress using memory and previous results
- Plan, repair, and continue execution until completion

You must act deterministically and safely.

---

INPUT:

USER INPUT:
{user_input}

GOAL:
{goal}

MEMORY:
{memory}

COMPLETED ACTIONS:
{completed_actions}

PREVIOUS STEPS:
{previous_steps}

RESULTS:
{results}

---

AVAILABLE ACTIONS:

1. list_files
     {"action":"list_files","path":"."}

2. read_file
     {"action":"read_file","path":"file.txt"}

3. run_command
     {"action":"run_command","cmd":"ls"}

4. search_project
     {"action":"search_project","query":"keyword"}

---

RULES:

- Output ONLY valid JSON
- No markdown or explanation
- Max 5 steps
- No destructive actions
- run_command only allows: ls, pwd
- Never hallucinate files or paths
- Always use previous results to guide next steps

---

TASK MODES:

Choose one:

PLAN -> create full step plan
REPAIR -> fix incomplete or incorrect plan
CONTINUE -> continue unfinished execution

---

COMPLETION RULE:

Set "complete": true ONLY IF:
- All parts of USER INPUT are satisfied
- No further steps are required

If uncertain -> complete must be false

---

FAILURE HANDLING:

If:
- results show error
- steps incomplete
- goal not achieved

Then:
- switch to CONTINUE or REPAIR
- generate next steps

---

MEMORY USAGE:

- Use MEMORY to recall past actions and preferences
- Do NOT repeat actions unnecessarily
- Prefer efficient execution

---

OUTPUT FORMAT:

{
    "mode": "PLAN | REPAIR | CONTINUE",
    "complete": true/false,
    "confidence": 0.0-1.0,
    "steps": [
        {"action":"action_name","param":"value"}
    ]
}

---

EXAMPLES:

User: show files
{
    "mode": "PLAN",
    "complete": true,
    "confidence": 0.9,
    "steps": [
        {"action":"list_files","path":"."}
    ]
}

User: read server.py and list files
{
    "mode": "PLAN",
    "complete": true,
    "confidence": 0.9,
    "steps": [
        {"action":"read_file","path":"server.py"},
        {"action":"list_files","path":"."}
    ]
}

User: continue task
{
    "mode": "CONTINUE",
    "complete": false,
    "confidence": 0.6,
    "steps": [
        {"action":"list_files","path":"."}
    ]
}

---

FAIL SAFE:

If unclear or unsafe:
{
    "mode": "PLAN",
    "complete": false,
    "confidence": 0.0,
    "steps": []
}
"""


def ask_llm(prompt, label="RAW"):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        data = res.json()
        print(label + ":", data)
        return data.get("response")
    except Exception as e:
        print("LLM ERROR:", str(e))
        return None


def ask_llm_with_retry(prompt, label_prefix="RAW", max_attempts=2):
    responses = []
    for attempt in range(max_attempts):
        response = ask_llm(prompt, label=f"{label_prefix} attempt {attempt}")
        responses.append(response)
        if response:
            return response, responses
    return None, responses


def load_memory():
    default_memory = {
        "recent_goals": [],
        "recent_actions": [],
    }

    if not os.path.exists(MEMORY_FILE):
        return default_memory

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as memory_file:
            data = json.load(memory_file)
    except Exception:
        return default_memory

    if not isinstance(data, dict):
        return default_memory

    recent_goals = data.get("recent_goals", [])
    recent_actions = data.get("recent_actions", [])
    if not isinstance(recent_goals, list):
        recent_goals = []
    if not isinstance(recent_actions, list):
        recent_actions = []

    return {
        "recent_goals": recent_goals[-MAX_MEMORY_ITEMS:],
        "recent_actions": recent_actions[-MAX_MEMORY_ITEMS:],
    }


def save_memory(goal, completed_actions):
    memory = load_memory()
    if goal:
        memory["recent_goals"].append(goal)

    for action in completed_actions:
        if action:
            memory["recent_actions"].append(action)

    memory["recent_goals"] = memory["recent_goals"][-MAX_MEMORY_ITEMS:]
    memory["recent_actions"] = memory["recent_actions"][-MAX_MEMORY_ITEMS:]

    with open(MEMORY_FILE, "w", encoding="utf-8") as memory_file:
        json.dump(memory, memory_file, ensure_ascii=True, indent=2)


def extract_goal(user_input):
    return user_input.lower()


def summarize_for_state(value):
    if isinstance(value, str):
        if len(value) > MAX_STATE_TEXT:
            return value[:MAX_STATE_TEXT] + "...[truncated]"
        return value

    if isinstance(value, list):
        return [summarize_for_state(item) for item in value[:10]]

    if isinstance(value, dict):
        summarized = {}
        for key, val in value.items():
            if key in {"step", "action", "error", "failed_step", "gaps", "mode", "iteration", "result", "confidence"}:
                summarized[key] = summarize_for_state(val)
        if summarized:
            return summarized
        return {"summary": "object"}

    return value


def summarize_execution_for_state(execution_result):
    summaries = []
    for item in execution_result:
        summaries.append(
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "result": summarize_for_state(item.get("result")),
            }
        )
    return summaries


def render_final_prompt(user_input, state, previous_steps):
    prompt = FINAL_PROMPT
    replacements = {
        "{user_input}": user_input,
        "{goal}": state["goal"],
        "{memory}": json.dumps(state["memory"], ensure_ascii=False),
        "{completed_actions}": json.dumps(state["completed_actions"], ensure_ascii=False),
        "{previous_steps}": json.dumps(previous_steps, ensure_ascii=False),
        "{results}": json.dumps(state["results"], ensure_ascii=False),
    }

    for key, value in replacements.items():
        prompt = prompt.replace(key, str(value))

    return prompt


def extract_json(text):
    if not text:
        return None

    cleaned = re.sub(r"```(?:json|python)?", "", text)
    cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(cleaned)):
            char = cleaned[idx]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

        start = cleaned.find("{", start + 1)

    return None


def parse_controller_output(text):
    data = extract_json(text)
    if not isinstance(data, dict):
        return None

    mode = data.get("mode")
    complete = data.get("complete")
    confidence = data.get("confidence")
    steps = data.get("steps")

    if mode not in ALLOWED_MODES:
        return None
    if not isinstance(complete, bool):
        return None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    if not isinstance(steps, list):
        return None

    return {
        "mode": mode,
        "complete": complete,
        "confidence": confidence,
        "steps": steps,
    }


def infer_plan_gaps(user_input, steps):
    text = user_input.lower()
    gaps = []

    if not steps:
        return ["no_steps"]

    actions = [step.get("action") for step in steps if isinstance(step, dict)]

    if re.search(r"\b(show|list)\s+files\b", text) and "list_files" not in actions:
        gaps.append("missing:list_files")

    if re.search(r"\bread\b", text) and "read_file" not in actions:
        gaps.append("missing:read_file")

    if re.search(r"\b(search|find|lookup|look\s+for)\b", text) and "search_project" not in actions:
        gaps.append("missing:search_project")

    if re.search(r"\b(show\s+)?current\s+directory\b|\bpwd\b", text):
        has_pwd = any(
            isinstance(step, dict)
            and step.get("action") == "run_command"
            and str(step.get("cmd", "")).strip() == "pwd"
            for step in steps
        )
        if not has_pwd:
            gaps.append("missing:run_command:pwd")

    if re.search(r"\brun\s+command\s+ls\b", text):
        has_ls = any(
            isinstance(step, dict)
            and step.get("action") == "run_command"
            and str(step.get("cmd", "")).strip() == "ls"
            for step in steps
        )
        if not has_ls:
            gaps.append("missing:run_command:ls")

    if " and " in text and len(steps) < 2:
        gaps.append("likely_incomplete:multi_task")

    return gaps


def infer_forbidden_actions(user_input, steps):
    forbidden = []

    for step in steps:
        if isinstance(step, dict) and step.get("action") == "write_file":
            forbidden.append("unexpected:write_file")

    return forbidden


def deterministic_plan_fallback(user_input):
    text = user_input.lower()
    steps = []

    if re.search(r"\bread\b", text) and "server.py" in text:
        steps.append({"action": "read_file", "path": "server.py"})

    if re.search(r"\b(show|list)\s+files\b", text):
        steps.append({"action": "list_files", "path": "."})

    if re.search(r"\b(show\s+)?current\s+directory\b|\bpwd\b", text):
        steps.append({"action": "run_command", "cmd": "pwd"})

    search_match = re.search(r"\b(?:search|find|lookup|look\s+for)\b(?:\s+project)?(?:\s+for)?\s+(.+)", text)
    if search_match:
        query_value = search_match.group(1).strip()
        if query_value:
            steps.append({"action": "search_project", "query": query_value})

    if not steps:
        return []

    deduped = []
    seen = set()
    for step in steps:
        key = json.dumps(step, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(step)

    return deduped[:MAX_STEPS]


def is_truly_complete(user_input, steps):
    user = user_input.lower()

    if " and " in user and len(steps) < 2:
        return False

    return True


def safe_path(path):
    if not isinstance(path, str) or not path.strip():
        raise Exception("Invalid path")

    full = os.path.abspath(os.path.join(BASE_DIR, path))
    if not (full == BASE_DIR or full.startswith(BASE_DIR + os.sep)):
        raise Exception("Access denied")
    return full


def validate_step(step):
    if not isinstance(step, dict):
        return False

    action = step.get("action")
    if action not in ALLOWED_ACTIONS:
        return False

    if set(step.keys()) - ALLOWED_FIELDS[action]:
        return False

    if action == "read_file" and "path" not in step:
        return False

    if action == "run_command" and "cmd" not in step:
        return False

    if action == "search_project" and "query" not in step:
        return False

    if action == "list_files" and "path" in step and not isinstance(step.get("path"), str):
        return False

    if action == "read_file" and not isinstance(step.get("path"), str):
        return False

    if action == "run_command":
        cmd = step.get("cmd")
        if not isinstance(cmd, str):
            return False
        parts = cmd.split()
        if len(parts) != 1 or parts[0] not in ALLOWED_COMMANDS:
            return False

    if action == "search_project" and not isinstance(step.get("query"), str):
        return False

    return True


def validate_steps(steps):
    if not isinstance(steps, list) or not steps:
        return False, "No valid steps"

    if len(steps) > MAX_STEPS:
        return False, "Too many steps"

    for index, step in enumerate(steps):
        if not validate_step(step):
            return False, f"Invalid step at index {index}"

    return True, ""


def list_files(path="."):
    path = safe_path(path)
    return os.listdir(path)


def read_file(path):
    path = safe_path(path)
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as file_obj:
        if size > MAX_FILE_SIZE:
            return file_obj.read(MAX_FILE_SIZE) + "\n...[truncated]"
        return file_obj.read()


def run_command(cmd):
    parts = cmd.split()

    if len(parts) == 0:
        return "blocked"

    if parts[0] not in ALLOWED_COMMANDS:
        return "blocked"

    if len(parts) > 1:
        return "blocked"

    return subprocess.getoutput(cmd)


def search_project(query):
    if not isinstance(query, str) or not query.strip():
        return {"error": "invalid_query"}

    query_text = query.strip().lower()
    matches = []
    excluded_dirs = {".git", "__pycache__", ".qodo"}

    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in excluded_dirs]

        for file_name in files:
            if len(matches) >= MAX_SEARCH_RESULTS:
                return matches

            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, BASE_DIR)

            if query_text in rel_path.lower():
                matches.append({"path": rel_path, "match": "path"})
                continue

            try:
                if os.path.getsize(abs_path) > MAX_SEARCH_FILE_BYTES:
                    continue

                with open(abs_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    for line_no, line in enumerate(file_obj, start=1):
                        if query_text in line.lower():
                            matches.append(
                                {
                                    "path": rel_path,
                                    "match": "content",
                                    "line": line_no,
                                    "snippet": line.strip()[:120],
                                }
                            )
                            break
            except Exception:
                continue

    return matches


def execute(action):
    if action["action"] == "list_files":
        return list_files(action.get("path", "."))

    if action["action"] == "read_file":
        return read_file(action["path"])

    if action["action"] == "run_command":
        return run_command(action["cmd"])

    if action["action"] == "search_project":
        return search_project(action["query"])

    return "no action"


def normalize_step_result(result):
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered.startswith("error:") or lowered == "blocked":
            return {"error": result}

    return result


def execute_steps(steps):
    results = []

    for index, step in enumerate(steps):
        if not validate_step(step):
            return {"error": f"Invalid step at index {index}", "step": step}

        try:
            result = execute(step)
            results.append(
                {
                    "step": index,
                    "action": step["action"],
                    "result": normalize_step_result(result),
                }
            )
        except Exception as error:
            return {
                "error": str(error),
                "failed_step": index,
                "step": step,
            }

    return results


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


@app.post("/run")
def run_task(data: dict, request: Request):
    if not API_KEY:
        return {"error": "server_not_configured"}

    if request.headers.get("x-api-key") != API_KEY:
        return {"error": "unauthorized"}

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
            state["results"] = [{"error": "Invalid controller output"}][-MAX_STATE_RESULTS:]
            if iteration == MAX_ITERATIONS - 1:
                return {"error": "Invalid controller output", "raw": raw_outputs}
            continue

        mode = controller["mode"]
        complete = controller["complete"]
        confidence = controller["confidence"]
        steps = controller["steps"]

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
                    [item.get("action") for item in execution_result if item.get("action")]
                )
                state["results"] = (
                    state["results"] + summarize_execution_for_state(execution_result)
                )[-MAX_STATE_RESULTS:]

                save_memory(state["goal"], state["completed_actions"])
                return {"results": all_results, "fallback": True}

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
            [item.get("action") for item in execution_result if item.get("action")]
        )
        state["results"] = (
            state["results"] + summarize_execution_for_state(execution_result)
        )[-MAX_STATE_RESULTS:]

        if complete and is_truly_complete(user_input, steps):
            save_memory(state["goal"], state["completed_actions"])
            return {"results": all_results}

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
