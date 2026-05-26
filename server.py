#!/usr/bin/env python3
"""
Hermes Kanban Dashboard + Multi-Agent Chat API
- Kanban board from kanban.db
- Agent Chat: POST /api/chat/<agent>  (user → agent)
- Agent forward: POST /api/chat/<agent>/forward  (agent → agent)
- Chat history: GET /api/chat/<agent>/history
- Agent list: GET /api/agents
- Agent config: GET/POST /api/chat/<agent>/config  (model + api key)
- Model presets: GET /api/model-presets
"""
import sqlite3
import json
import os
import time
import subprocess
import base64
import re
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

DB_PATH = os.path.expanduser("~/.hermes/kanban.db")
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
HERMES_HOME = os.path.expanduser("~/.hermes")
PROFILES_DIR = os.path.join(HERMES_HOME, "profiles")

# ─── Comprehensive Model Presets ───
MODEL_PRESETS = {
    "deepseek": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "label": "DeepSeek",
        "icon": "🧠",
        "models": [
            {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash (推荐)", "context": "64K"},
            {"id": "deepseek-reasoner", "name": "DeepSeek R1 (推理)", "context": "64K"},
        ]
    },
    "openai": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "label": "OpenAI",
        "icon": "🤖",
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o", "context": "128K"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini (经济)", "context": "128K"},
            {"id": "o3-mini", "name": "o3-mini (推理)", "context": "200K"},
            {"id": "gpt-4.1", "name": "GPT-4.1", "context": "1M"},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "context": "1M"},
        ]
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "label": "Anthropic",
        "icon": "🔮",
        "models": [
            {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "context": "200K"},
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (0506)", "context": "200K"},
            {"id": "claude-haiku-3-5", "name": "Claude Haiku 3.5", "context": "200K"},
        ]
    },
    "openrouter": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "label": "OpenRouter",
        "icon": "🌐",
        "models": [
            {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "context": "200K"},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "context": "128K"},
            {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "context": "1M"},
            {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3", "context": "128K"},
            {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "context": "128K"},
            {"id": "qwen/qwen-plus", "name": "Qwen+", "context": "131K"},
            {"id": "cognitivecomputations/dolphin3.0-r1-mistral-24b", "name": "Dolphin 3.0 R1", "context": "32K"},
        ]
    },
    "google": {
        "provider": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "label": "Google Gemini",
        "icon": "🔵",
        "models": [
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context": "1M"},
            {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "context": "1M"},
            {"id": "gemini-2.5-flash-preview-04-17", "name": "Gemini 2.5 Flash", "context": "1M"},
            {"id": "gemini-2.5-pro-preview-03-25", "name": "Gemini 2.5 Pro", "context": "1M"},
        ]
    },
    "zhipu": {
        "provider": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "label": "智谱 GLM",
        "icon": "🟤",
        "models": [
            {"id": "glm-5-flash", "name": "GLM-5-Flash (免费)", "context": "128K"},
            {"id": "glm-5", "name": "GLM-5", "context": "128K"},
            {"id": "glm-5-air", "name": "GLM-5-Air (经济)", "context": "128K"},
        ]
    },
    "moonshot": {
        "provider": "moonshot",
        "base_url": "https://api.moonshot.cn/v1",
        "label": "月之暗面 Kimi",
        "icon": "🌙",
        "models": [
            {"id": "moonshot-v1-8k", "name": "Moonshot v1 8K", "context": "8K"},
            {"id": "moonshot-v1-32k", "name": "Moonshot v1 32K", "context": "32K"},
            {"id": "moonshot-v1-128k", "name": "Moonshot v1 128K", "context": "128K"},
        ]
    },
    "alibaba": {
        "provider": "alibaba",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "label": "阿里通义千问",
        "icon": "☁️",
        "models": [
            {"id": "qwen-turbo-2025-04-25", "name": "Qwen Turbo", "context": "1M"},
            {"id": "qwen-plus-2025-04-25", "name": "Qwen Plus", "context": "131K"},
            {"id": "qwen-max-2025-04-25", "name": "Qwen Max", "context": "32K"},
            {"id": "qwen-coder-plus", "name": "Qwen Coder Plus", "context": "128K"},
            {"id": "qwen-coder-max-latest", "name": "Qwen Coder Max", "context": "128K"},
        ]
    },
    "siliconflow": {
        "provider": "custom",
        "base_url": "https://api.siliconflow.cn/v1",
        "label": "SiliconFlow (硅基流动)",
        "icon": "💎",
        "models": [
            {"id": "Pro/deepseek-ai/DeepSeek-V3", "name": "DeepSeek V3", "context": "64K"},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "context": "64K"},
            {"id": "Qwen/Qwen2.5-72B-Instruct-128K", "name": "Qwen2.5 72B", "context": "128K"},
            {"id": "deepseek-ai/DeepSeek-V2.5", "name": "DeepSeek V2.5", "context": "128K"},
            {"id": "THUDM/glm-4-9b-chat", "name": "GLM-4 9B", "context": "128K"},
        ]
    },
    "groq": {
        "provider": "custom",
        "base_url": "https://api.groq.com/openai/v1",
        "label": "Groq (超快)",
        "icon": "⚡",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "context": "128K"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B (极速)", "context": "128K"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "context": "32K"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "context": "8K"},
        ]
    },
    "together": {
        "provider": "custom",
        "base_url": "https://api.together.xyz/v1",
        "label": "Together AI",
        "icon": "🟢",
        "models": [
            {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "name": "Mixtral 8x22B", "context": "64K"},
            {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "Llama 3.3 70B", "context": "128K"},
            {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek V3", "context": "128K"},
        ]
    },
    "xai": {
        "provider": "xai",
        "base_url": "https://api.x.ai/v1",
        "label": "xAI Grok",
        "icon": "🕳️",
        "models": [
            {"id": "grok-2-latest", "name": "Grok 2", "context": "128K"},
            {"id": "grok-2-vision-latest", "name": "Grok 2 Vision", "context": "128K"},
        ]
    },
    "local": {
        "provider": "custom",
        "base_url": "http://localhost:11434/v1",
        "label": "本地 Ollama",
        "icon": "💻",
        "models": [
            {"id": "llama3.2:3b", "name": "Llama 3.2 3B", "context": "128K"},
            {"id": "qwen2.5:7b", "name": "Qwen 2.5 7B", "context": "128K"},
            {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B", "context": "128K"},
        ]
    },
    "custom": {
        "provider": "custom",
        "base_url": "",
        "label": "自定义端点",
        "icon": "🔧",
        "models": [
            {"id": "custom-model", "name": "自定义模型", "context": "—"},
        ]
    }
}

# ─── Agent Definitions ───
AGENTS = [
    {"id": "team-lead",          "name": "负责人",          "icon": "👑", "color": "#fbbf24"},
    {"id": "project-director",   "name": "项目总监",        "icon": "🎯", "color": "#f97316"},
    {"id": "product-manager",    "name": "产品经理",        "icon": "📋", "color": "#6366f1"},
    {"id": "designer",           "name": "设计师",          "icon": "🎨", "color": "#22d3ee"},
    {"id": "architect",          "name": "架构师",          "icon": "🏗️", "color": "#34d399"},
    {"id": "frontend-engineer",  "name": "前端工程师",      "icon": "⚛️", "color": "#14b8a6"},
    {"id": "backend-engineer",   "name": "后端工程师",      "icon": "⚙️", "color": "#a78bfa"},
    {"id": "qa-tester",          "name": "测试工程师",      "icon": "🧪", "color": "#eab308"},
    {"id": "devops",             "name": "运维",            "icon": "🐳", "color": "#ec4899"},
]
AGENT_IDS = {a["id"] for a in AGENTS}

# ─── Active Session Tracking ───
# Track which agents are currently being chatted with
_active_sessions = {}  # agent_id → {"last_active": timestamp, "message_count": int}

def track_activity(agent_id: str):
    """Record that an agent was just used."""
    _active_sessions[agent_id] = {
        "last_active": time.time(),
        "message_count": _active_sessions.get(agent_id, {}).get("message_count", 0) + 1
    }

def get_agent_activity() -> dict:
    """Return activity status for all agents."""
    now = time.time()
    result = {}
    for a in AGENTS:
        aid = a["id"]
        session = _active_sessions.get(aid)
        if session:
            age = now - session["last_active"]
            result[aid] = {
                "active": age < 300,  # active if within 5 minutes
                "last_active_seconds": int(age),
                "message_count": session["message_count"],
            }
        else:
            result[aid] = {"active": False, "last_active_seconds": None, "message_count": 0}
    return result

# ─── ENV Variable Names ───
# Map provider → Hermes env var for API key
PROVIDER_ENV_MAP = {
    "deepseek":  "DEEPSEEK_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter":"OPENROUTER_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "zhipu":     "GLM_API_KEY",
    "glm":       "GLM_API_KEY",
    "moonshot":  "KIMI_API_KEY",
    "alibaba":   "DASHSCOPE_API_KEY",
    "xai":       "XAI_API_KEY",
}

# ─── Agent Config Management ───
def get_agent_profile_dir(agent_id: str) -> str:
    return os.path.join(PROFILES_DIR, agent_id)

def get_agent_config(agent_id: str) -> dict:
    """Read current model config + env from a profile."""
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir, "config.yaml")
    env_path = os.path.join(profile_dir, ".env")

    model_id = "deepseek-v4-flash"
    provider = "deepseek"
    base_url = "https://api.deepseek.com"
    api_key = ""

    # Parse config.yaml — support both nested YAML and flat formats
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
        # Flat format: model.default: value
        for line in lines:
            s = line.strip()
            if s.startswith("model.default:"):
                val = s.split(":", 1)[1].strip().strip("\"'")
                if val: model_id = val
            elif s.startswith("model.provider:"):
                val = s.split(":", 1)[1].strip().strip("\"'")
                if val: provider = val
            elif s.startswith("model.base_url:"):
                val = s.split(":", 1)[1].strip().strip("\"'")
                if val: base_url = val
        # Nested format: model:\n  default: ...\n  provider: ...\n  base_url: ...
        for i, line in enumerate(lines):
            if line.strip() == "model:":
                # Look at next lines that are indented (children of model:)
                for j in range(i+1, min(i+10, len(lines))):
                    raw_line = lines[j]
                    if raw_line[0:1] != " " and raw_line.strip():  # not indented, not empty
                        break
                    s = raw_line.strip()
                    if s.startswith("default:"):
                        val = s.split(":", 1)[1].strip().strip("\"'")
                        if val: model_id = val
                    elif s.startswith("provider:"):
                        val = s.split(":", 1)[1].strip().strip("\"'")
                        if val: provider = val
                    elif s.startswith("base_url:"):
                        val = s.split(":", 1)[1].strip().strip("\"'")
                        if val: base_url = val

    # Parse .env — look for the env var name matching this provider
    if os.path.isfile(env_path):
        expected_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(expected_var + "="):
                    k, v = line.split("=", 1)
                    v = v.strip().strip("\"'")
                    if v:
                        api_key = f"{k}={v}"
                    break
            # Fallback: scan all non-comment lines for any API key
            if not api_key:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#") and "API_KEY" in line:
                        k, v = line.split("=", 1)
                        v = v.strip().strip("\"'")
                        if v:
                            api_key = f"{k}={v}"

    return {
        "model_id": model_id,
        "provider": provider,
        "base_url": base_url,
        "api_key_configured": bool(api_key),
    }

def save_agent_config(agent_id: str, model_id: str, provider: str, base_url: str, api_key: str) -> dict:
    """Save model config + api key to a profile's config.yaml and .env."""
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir, "config.yaml")
    env_path = os.path.join(profile_dir, ".env")

    os.makedirs(profile_dir, exist_ok=True)

    # ── Update config.yaml ──
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    new_lines = []
    found_model = found_provider = found_base_url = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("model.default:"):
            new_lines.append(f"model.default: \"{model_id}\"\n")
            found_model = True
        elif stripped.startswith("model.provider:"):
            new_lines.append(f"model.provider: \"{provider}\"\n")
            found_provider = True
        elif stripped.startswith("model.base_url:"):
            new_lines.append(f"model.base_url: \"{base_url}\"\n")
            found_base_url = True
        else:
            new_lines.append(line)

    if not found_model:
        new_lines.append(f"model.default: \"{model_id}\"\n")
    if not found_provider:
        new_lines.append(f"model.provider: \"{provider}\"\n")
    if not found_base_url:
        new_lines.append(f"model.base_url: \"{base_url}\"\n")

    with open(config_path, "w") as f:
        f.writelines(new_lines)

    # ── Update .env ──
    # Determine correct env var name based on provider
    env_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
    if provider == "custom":
        # For custom provider, use a standard key
        env_var = "CUSTOM_API_KEY"

    new_env_lines = []
    found_key = False
    if os.path.isfile(env_path):
        with open(env_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(env_var + "="):
                    if api_key:
                        new_env_lines.append(f"{env_var}={api_key}\n")
                    found_key = True
                else:
                    new_env_lines.append(line)

    if not found_key and api_key:
        new_env_lines.append(f"{env_var}={api_key}\n")

    with open(env_path, "w") as f:
        f.writelines(new_env_lines)

    return {"success": True, "message": f"已保存 {agent_id} 的配置"}

# ─── Chat Session Manager ───
_chat_sessions = {}
_HERMES_BIN = None

def get_hermes_bin():
    global _HERMES_BIN
    if _HERMES_BIN is None:
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(p, "hermes")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                _HERMES_BIN = candidate
                break
        if _HERMES_BIN is None:
            _HERMES_BIN = "hermes"
    return _HERMES_BIN

def get_session(agent_id: str) -> dict:
    if agent_id not in _chat_sessions:
        _chat_sessions[agent_id] = {
            "id": agent_id,
            "messages": [],
            "created_at": time.time()
        }
    return _chat_sessions[agent_id]

def call_agent(agent_id: str, user_message: str) -> dict:
    session = get_session(agent_id)
    hermes = get_hermes_bin()

    history = session["messages"][-20:]
    ctx_lines = []
    for m in history:
        role_label = "用户" if m["role"] == "user" else "助手"
        ctx_lines.append(f"{role_label}: {m['content'][:500]}")
    context_str = ""
    if ctx_lines:
        context_str = "以下是本次对话的最近历史记录，请参考上下文回答：\n" + "\n".join(ctx_lines) + "\n\n---\n\n"

    full_prompt = context_str + user_message

    try:
        proc = subprocess.run(
            [hermes, "--yolo", "-p", agent_id, "chat", "-q", full_prompt, "--quiet"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "HERMES_QUIET": "1", "TERM": "dumb"}
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        response = stdout
        if not response:
            response = stderr if stderr else "(Agent 无响应)"

        for pattern in [
            r"^\s*━━━.*━━━\s*$", r"^\s*─+.*─+\s*$",
            r"^✦\s+.*", r"^╭─.*", r"^╰─.*", r"^│.*",
        ]:
            response = re.sub(pattern, "", response, flags=re.MULTILINE)
        response = response.strip()

        # Remove session_id line if present
        response = re.sub(r"^session_id:\s+\S+\s*\n?", "", response).strip()

        success = proc.returncode == 0 and len(response) > 0
    except subprocess.TimeoutExpired:
        response = f"(⏱️ Agent {agent_id} 超时，180秒未响应)"
        success = False
    except Exception as e:
        response = f"(❌ 错误: {str(e)})"
        success = False

    session["messages"].append({"role": "user", "content": user_message, "timestamp": time.time()})
    session["messages"].append({"role": "assistant", "content": response, "timestamp": time.time()})
    if len(session["messages"]) > 100:
        session["messages"] = session["messages"][-100:]

    # Track activity
    track_activity(agent_id)

    return {
        "success": success,
        "response": response,
        "agent": agent_id,
        "agent_name": next((a["name"] for a in AGENTS if a["id"] == agent_id), agent_id)
    }

def forward_to_agent(from_agent: str, to_agent: str, message: str) -> dict:
    if to_agent not in AGENT_IDS:
        return {"success": False, "response": f"未知目标Agent: {to_agent}", "agent": to_agent}

    from_name = next((a["name"] for a in AGENTS if a["id"] == from_agent), from_agent)
    to_name = next((a["name"] for a in AGENTS if a["id"] == to_agent), to_agent)

    # Build rich briefing context
    forward_msg = (
        f"【任务协调 - 来自 {from_name}({from_agent}) → 转交 {to_name}({to_agent})】\n\n"
        f"上游分析/结论：\n{message}\n\n"
        f"请基于上述信息执行你的职责范围内的分析，给出具体建议或方案。"
        f"如果需要补充信息或需要其他Agent协助，请明确指出。"
    )
    # Track activity for both agents
    track_activity(from_agent)
    result = call_agent(to_agent, forward_msg)
    track_activity(to_agent)
    return result

# ─── Kanban DB ───
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_tasks():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT id, title, body, assignee, status, priority,
                   created_by, created_at, started_at, completed_at,
                   workspace_kind, workspace_path, result,
                   consecutive_failures, last_failure_error,
                   current_run_id, skills, model_override
            FROM tasks ORDER BY created_at DESC
        """).fetchall()
        tasks = []
        for r in rows:
            task = dict(r)
            tid = r["id"]
            runs = conn.execute("""
                SELECT id, profile, status, started_at, ended_at,
                       outcome, summary, error
                FROM task_runs WHERE task_id = ?
                ORDER BY started_at DESC LIMIT 10
            """, (tid,)).fetchall()
            task["runs"] = [dict(run) for run in runs]
            cc = conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,)).fetchone()[0]
            task["comment_count"] = cc
            parents = conn.execute("SELECT parent_id FROM task_links WHERE child_id = ?", (tid,)).fetchall()
            task["parents"] = [p["parent_id"] for p in parents]
            children = conn.execute("SELECT child_id FROM task_links WHERE parent_id = ?", (tid,)).fetchall()
            task["children"] = [c["child_id"] for c in children]
            tasks.append(task)
        return tasks
    finally:
        conn.close()

def get_stats():
    conn = get_db()
    try:
        stats = {}
        for status in ["todo", "ready", "running", "blocked", "done", "archived"]:
            stats[status] = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (status,)
            ).fetchone()[0]
        stats["total"] = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        profiles = conn.execute("""
            SELECT assignee, status, COUNT(*) as cnt
            FROM tasks WHERE assignee IS NOT NULL
            GROUP BY assignee, status
        """).fetchall()
        profile_stats = {}
        for p in profiles:
            name = p["assignee"]
            if name not in profile_stats:
                profile_stats[name] = {}
            profile_stats[name][p["status"]] = p["cnt"]
        stats["profiles"] = profile_stats
        return stats
    finally:
        conn.close()

def get_events(limit=50):
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT id, task_id, run_id, kind, payload, created_at
            FROM task_events ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ─── HTTP Handler ───
class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/tasks":
            self.send_json(get_all_tasks())
        elif path == "/api/stats":
            self.send_json(get_stats())
        elif path == "/api/events":
            self.send_json(get_events())
        elif path == "/api/health":
            self.send_json({"status": "ok", "db": os.path.exists(DB_PATH), "time": int(time.time())})
        elif path == "/api/agents":
            self.send_json({"agents": AGENTS, "total": len(AGENTS)})
        elif path == "/api/model-presets":
            self.send_json(MODEL_PRESETS)
        elif re.match(r"^/api/chat/[\w-]+/history$", path):
            agent = path.split("/")[3]
            self.send_json(get_session(agent))
        elif re.match(r"^/api/chat/[\w-]+/config$", path):
            agent = path.split("/")[3]
            if agent not in AGENT_IDS:
                self.send_json({"error": "未知Agent"}, 404)
                return
            self.send_json(get_agent_config(agent))
        elif path == "/api/agents/status":
            # Check each agent's readiness
            status_list = []
            for a in AGENTS:
                cfg = get_agent_config(a["id"])
                status_list.append({
                    "id": a["id"],
                    "name": a["name"],
                    "icon": a["icon"],
                    "configured": cfg["api_key_configured"],
                    "model": cfg["model_id"],
                    "provider": cfg["provider"],
                })
            self.send_json({"agents": status_list})
        elif path == "/api/agents/activity":
            self.send_json(get_agent_activity())
        elif path == "/" or path == "/index.html":
            self.serve_file("index.html", "text/html")
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # POST /api/chat/<agent>  — User sends message
        chat_match = re.match(r"^/api/chat/([\w-]+)$", path)
        if chat_match:
            agent_id = chat_match.group(1)
            if agent_id not in AGENT_IDS:
                self.send_json({"success": False, "error": f"未知Agent: {agent_id}"}, 404)
                return
            body = self.read_body()
            if not body or "message" not in body:
                self.send_json({"success": False, "error": "缺少 message 字段"}, 400)
                return
            user_msg = body["message"].strip()
            if not user_msg:
                self.send_json({"success": False, "error": "消息不能为空"}, 400)
                return
            result = call_agent(agent_id, user_msg)
            self.send_json(result)
            return

        # POST /api/chat/<agent>/forward
        forward_match = re.match(r"^/api/chat/([\w-]+)/forward$", path)
        if forward_match:
            from_agent = forward_match.group(1)
            body = self.read_body()
            if not body or "to" not in body or "message" not in body:
                self.send_json({"success": False, "error": "需要 to 和 message 字段"}, 400)
                return
            result = forward_to_agent(from_agent, body["to"], body["message"])
            self.send_json(result)
            return

        # POST /api/chat/<agent>/config  — Save agent config
        config_match = re.match(r"^/api/chat/([\w-]+)/config$", path)
        if config_match:
            agent_id = config_match.group(1)
            if agent_id not in AGENT_IDS:
                self.send_json({"success": False, "error": f"未知Agent: {agent_id}"}, 404)
                return
            body = self.read_body()
            model_id = body.get("model_id", "deepseek-chat")
            provider = body.get("provider", "deepseek")
            base_url = body.get("base_url", "https://api.deepseek.com")
            api_key = body.get("api_key", "")
            result = save_agent_config(agent_id, model_id, provider, base_url, api_key)
            self.send_json(result)
            return

        self.send_json({"success": False, "error": "未找到"}, 404)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, filename, content_type):
        filepath = os.path.join(STATIC_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8765))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Hermes Multi-Agent Dashboard           ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Dashboard:  http://localhost:{port}         ║")
    print(f"║  Agents:     {len(AGENTS)} configured             ║")
    print(f"║  Providers:  {len(MODEL_PRESETS)} available             ║")
    print(f"║  Chat API:   /api/chat/<agent>          ║")
    print(f"║  Config API: /api/chat/<agent>/config   ║")
    print(f"╚══════════════════════════════════════════╝")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
