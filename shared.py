"""
shared.py — Shared infrastructure for Jarvis Agent.

Contains: constants, LLM interface, JSON extraction, tool functions,
step validation, state summarization, memory, plan inference,
deterministic fallback, and the autonomous prompt template.

Imported by server.py and autonomous_worker.py.
"""

import json
import os
import re
import subprocess

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-7b-it:free")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")

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

ALLOWED_COMMANDS = ["ls", "pwd", "date"]
ALLOWED_ACTIONS = ["list_files", "read_file", "run_command", "search_project", "open_app"]
ALLOWED_MODES = ["PLAN", "REPAIR", "CONTINUE"]

ALLOWED_FIELDS = {
    "list_files": {"action", "path"},
    "read_file": {"action", "path"},
    "run_command": {"action", "cmd"},
    "search_project": {"action", "query"},
    "open_app": {"action", "name"},
}

# ---------------------------------------------------------------------------
# Autonomous Prompt
# ---------------------------------------------------------------------------

FINAL_PROMPT = """
You are an autonomous system controller.

You must act deterministically and safely, but also speak naturally to the user.
Use the "thought" field to explain what you are doing, ask for clarification, or respond to greetings.

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

5. open_app
     {"action":"open_app","name":"Google Chrome"}

5. open_app
     {"action":"open_app","name":"Google Chrome"}

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
    "thought": "A natural sentence talking to the user.",
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

# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------


def ask_openrouter(prompt, label="RAW"):
    """Send a prompt to OpenRouter and return response content text."""
    if not OPENROUTER_API_KEY:
        print("OPENROUTER SKIP: missing OPENROUTER_API_KEY")
        return None

    try:
        res = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "Return ONLY JSON"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        print(label + " [openrouter]:", data)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content

        return None
    except Exception as e:
        print("OPENROUTER ERROR:", str(e))
        return None


def ask_local(prompt, label="RAW"):
    """Send a prompt to the local Ollama LLM and return the response text."""
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        data = res.json()
        print(label + " [local]:", data)
        return data.get("response")
    except Exception as e:
        print("LOCAL LLM ERROR:", str(e))
        return None


def ask_llm_with_fallback(prompt, label="RAW"):
    """Try OpenRouter first, then fallback to local Ollama."""
    cloud_response = ask_openrouter(prompt, label=label)
    if cloud_response:
        return cloud_response

    return ask_local(prompt, label=label)


def ask_llm(prompt, label="RAW"):
    """Primary LLM gateway used by the controller retry loop."""
    return ask_llm_with_fallback(prompt, label=label)


def ask_llm_with_retry(prompt, label_prefix="RAW", max_attempts=2):
    """Try the LLM up to *max_attempts* times, returning first non-None."""
    responses = []
    for attempt in range(max_attempts):
        response = ask_llm(prompt, label=f"{label_prefix} attempt {attempt}")
        responses.append(response)
        if response:
            return response, responses
    return None, responses


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------


def extract_json(text):
    """Extract the most likely valid JSON object from *text*."""
    if not text:
        return None

    # Remove markdown code blocks
    cleaned = re.sub(r"```(?:json|python)?", "", text)
    cleaned = cleaned.replace("```", "").strip()

    candidates = []
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
                    candidate_str = cleaned[start : idx + 1]
                    try:
                        obj = json.loads(candidate_str)
                        if isinstance(obj, dict):
                            # Score candidates based on presence of expected keys
                            score = 0
                            if any(k in obj for k in ["mode", "complete", "steps"]): score += 2
                            if any(k in obj for k in ["action", "result", "error"]): score += 1
                            candidates.append((score, obj))
                    except Exception:
                        pass
                    break
        start = cleaned.find("{", start + 1)

    if not candidates:
        return None

    # Return the candidate with the highest "validity" score
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def parse_controller_output(text):
    """Parse LLM text into a validated controller dict or conversational fallback."""
    clean_text = text.strip() if text else ""
    
    # The ultimate conversational fallback
    fallback = {
        "mode": "PLAN",
        "complete": True,
        "confidence": 1.0,
        "steps": [],
        "thought": clean_text,
    }

    data = extract_json(clean_text)
    if not isinstance(data, dict):
        return fallback if clean_text else None

    mode = data.get("mode")
    complete = data.get("complete")
    confidence = data.get("confidence")
    steps = data.get("steps")
    thought = data.get("thought", clean_text)

    if mode not in ALLOWED_MODES:
        return fallback
    if not isinstance(complete, bool):
        return fallback
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return fallback
    if confidence < 0.0 or confidence > 1.0:
        return fallback
    if not isinstance(steps, list):
        return fallback

    return {
        "mode": mode,
        "complete": complete,
        "confidence": confidence,
        "steps": steps,
        "thought": str(thought) if thought else None,
    }


# ---------------------------------------------------------------------------
# Path Safety
# ---------------------------------------------------------------------------


def safe_path(path):
    """Resolve *path* against BASE_DIR and reject escapes."""
    if not isinstance(path, str) or not path.strip():
        raise Exception("Invalid path")

    full = os.path.abspath(os.path.join(BASE_DIR, path))
    if not (full == BASE_DIR or full.startswith(BASE_DIR + os.sep)):
        raise Exception("Access denied")
    return full


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------


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


def open_app(name):
    """Open a macOS application by name."""
    try:
        subprocess.run(["open", "-a", name], check=True)
        return f"Opened {name}"
    except Exception as e:
        return f"Error opening {name}: {str(e)}"


def execute(action):
    """Dispatch a single validated action dict to the appropriate tool."""
    if action["action"] == "list_files":
        return list_files(action.get("path", "."))

    if action["action"] == "read_file":
        return read_file(action["path"])

    if action["action"] == "run_command":
        return run_command(action["cmd"])

    if action["action"] == "search_project":
        return search_project(action["query"])

    if action["action"] == "open_app":
        return open_app(action["name"])

    return "no action"


# ---------------------------------------------------------------------------
# Step Validation
# ---------------------------------------------------------------------------


def validate_step(step):
    """Return True if *step* is a valid action dict."""
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
    """Validate a list of steps. Returns (is_valid, reason)."""
    if not isinstance(steps, list) or not steps:
        return False, "No valid steps"

    if len(steps) > MAX_STEPS:
        return False, "Too many steps"

    for index, step in enumerate(steps):
        if not validate_step(step):
            return False, f"Invalid step at index {index}"

    return True, ""


def normalize_step_result(result):
    """Tag error-like results with an error key."""
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered.startswith("error:") or lowered == "blocked":
            return {"error": result}

    return result


def execute_steps(steps):
    """Execute a list of validated steps and return results or error dict."""
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


# ---------------------------------------------------------------------------
# State Summarization
# ---------------------------------------------------------------------------


def summarize_for_state(value):
    """Truncate large values so state stays bounded."""
    if isinstance(value, str):
        if len(value) > MAX_STATE_TEXT:
            return value[:MAX_STATE_TEXT] + "...[truncated]"
        return value

    if isinstance(value, list):
        return [summarize_for_state(item) for item in value[:10]]

    if isinstance(value, dict):
        summarized = {}
        for key, val in value.items():
            if key in {
                "step",
                "action",
                "error",
                "failed_step",
                "gaps",
                "mode",
                "iteration",
                "result",
                "confidence",
            }:
                summarized[key] = summarize_for_state(val)
        if summarized:
            return summarized
        return {"summary": "object"}

    return value


def summarize_execution_for_state(execution_result):
    """Produce compact summaries of execution results for state injection."""
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


# ---------------------------------------------------------------------------
# Memory (persistent across goals)
# ---------------------------------------------------------------------------


def load_memory():
    """Load cross-goal memory from disk."""
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
    """Persist goal and actions to cross-goal memory."""
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


# ---------------------------------------------------------------------------
# Goal Extraction
# ---------------------------------------------------------------------------


def extract_goal(user_input):
    """Normalize user input into a goal string."""
    return user_input.lower()


# ---------------------------------------------------------------------------
# Plan Inference & Fallback
# ---------------------------------------------------------------------------


def infer_plan_gaps(user_input, steps):
    """Detect missing actions that the user likely expects."""
    text = user_input.lower()
    gaps = []

    if not steps:
        return ["no_steps"]

    actions = [step.get("action") for step in steps if isinstance(step, dict)]

    if re.search(r"\b(show|list)\s+files\b", text) and "list_files" not in actions:
        gaps.append("missing:list_files")

    if re.search(r"\bread\b", text) and "read_file" not in actions:
        gaps.append("missing:read_file")

    if (
        re.search(r"\b(search|find|lookup|look\s+for)\b", text)
        and "search_project" not in actions
    ):
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
    """Flag actions that should never appear."""
    forbidden = []

    for step in steps:
        if isinstance(step, dict) and step.get("action") == "write_file":
            forbidden.append("unexpected:write_file")

    return forbidden


def deterministic_plan_fallback(user_input):
    """Build a plan from regex pattern matching — no LLM needed."""
    text = user_input.lower()
    steps = []

    # ── Time queries ──────────────────────────────────────────────────────────
    if re.search(r"\b(what'?s?\s+the\s+time|what\s+is\s+the\s+time|what\s*time|current\s+time|tell\s+me\s+the\s+time|check.*time|device\s+time|time\s+(now|today|right\s+now))\b", text):
        steps.append({"action": "run_command", "cmd": "date"})

    # ── Open app requests ─────────────────────────────────────────────────────
    KNOWN_APPS = {
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
        "safari": "Safari",
        "firefox": "Firefox",
        "spotify": "Spotify",
        "terminal": "Terminal",
        "finder": "Finder",
        "notes": "Notes",
        "calculator": "Calculator",
        "photos": "Photos",
        "messages": "Messages",
        "mail": "Mail",
        "calendar": "Calendar",
        "maps": "Maps",
        "vlc": "VLC",
        "vscode": "Visual Studio Code",
    }
    if re.search(r"\b(open|launch|start)\b", text):
        for key, app_name in KNOWN_APPS.items():
            if key in text:
                steps.append({"action": "open_app", "name": app_name})
                break

    # ── File operations ───────────────────────────────────────────────────────
    if re.search(r"\bread\b", text) and "server.py" in text:
        steps.append({"action": "read_file", "path": "server.py"})

    if re.search(r"\b(show|list)\s+files\b", text):
        steps.append({"action": "list_files", "path": "."})

    if re.search(r"\b(show\s+)?current\s+directory\b|\bpwd\b", text):
        steps.append({"action": "run_command", "cmd": "pwd"})

    search_match = re.search(
        r"\b(?:search|find|lookup|look\s+for)\b(?:\s+project)?(?:\s+for)?\s+(.+)", text
    )
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
    """Extra guard: reject premature completion for compound goals."""
    user = user_input.lower()

    if " and " in user and len(steps) < 2:
        return False

    return True


# ---------------------------------------------------------------------------
# Prompt Rendering
# ---------------------------------------------------------------------------


def render_final_prompt(user_input, state, previous_steps):
    """Fill the autonomous prompt template with current state."""
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
