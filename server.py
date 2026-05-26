#!/usr/bin/env python3
"""
Hermes Kanban Dashboard + Multi-Agent Chat API

修复列表 (2025-05):
  [FIX-1] ThreadingMixIn — 多线程并发，防止长请求阻塞
  [FIX-2] CORS 限制为 localhost，防止跨域攻击
  [FIX-3] pyyaml 替代手写 YAML 解析，兼容 flat/nested 两种格式
  [FIX-4] API key 读取 bug 修复 (f.seek/fallback 逻辑错误)
  [FIX-5] GLM provider 映射补全 ("glm" → GLM_API_KEY)
  [FIX-6] chat 历史持久化到 SQLite (chat_messages 表)
  [FIX-7] 所有路由严格校验 agent_id
  [FIX-8] API 响应不暴露 key 值，只返回 bool
"""

import sqlite3
import json
import os
import time
import subprocess
import re
import yaml
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn          # [FIX-1]
from urllib.parse import urlparse

DB_PATH      = os.path.expanduser("~/.hermes/kanban.db")
STATIC_DIR   = os.path.dirname(os.path.abspath(__file__))
HERMES_HOME  = os.path.expanduser("~/.hermes")
PROFILES_DIR = os.path.join(HERMES_HOME, "profiles")

# [FIX-2] 只允许来自 localhost 的跨域请求
ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}

# ─── Model Presets ──────────────────────────────────────────────────────────

MODEL_PRESETS = {
    "deepseek": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "label": "DeepSeek", "icon": "🧠",
        "models": [
            {"id": "deepseek-v4-flash",  "name": "DeepSeek V4 Flash (推荐)", "context": "64K"},
            {"id": "deepseek-reasoner",  "name": "DeepSeek R1 (推理)",        "context": "64K"},
        ],
    },
    "openai": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "label": "OpenAI", "icon": "🤖",
        "models": [
            {"id": "gpt-4o",       "name": "GPT-4o",            "context": "128K"},
            {"id": "gpt-4o-mini",  "name": "GPT-4o Mini (经济)", "context": "128K"},
            {"id": "o3-mini",      "name": "o3-mini (推理)",      "context": "200K"},
            {"id": "gpt-4.1",      "name": "GPT-4.1",            "context": "1M"},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini",       "context": "1M"},
        ],
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "label": "Anthropic", "icon": "🔮",
        "models": [
            {"id": "claude-sonnet-4",          "name": "Claude Sonnet 4",       "context": "200K"},
            {"id": "claude-sonnet-4-20250514",  "name": "Claude Sonnet 4 (0514)","context": "200K"},
            {"id": "claude-haiku-3-5",          "name": "Claude Haiku 3.5",      "context": "200K"},
        ],
    },
    "openrouter": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "label": "OpenRouter", "icon": "🌐",
        "models": [
            {"id": "anthropic/claude-sonnet-4",                          "name": "Claude Sonnet 4",   "context": "200K"},
            {"id": "openai/gpt-4o",                                      "name": "GPT-4o",            "context": "128K"},
            {"id": "google/gemini-2.0-flash-001",                        "name": "Gemini 2.0 Flash",  "context": "1M"},
            {"id": "deepseek/deepseek-chat",                             "name": "DeepSeek V3",       "context": "128K"},
            {"id": "deepseek/deepseek-r1",                               "name": "DeepSeek R1",       "context": "128K"},
            {"id": "qwen/qwen-plus",                                     "name": "Qwen+",             "context": "131K"},
            {"id": "cognitivecomputations/dolphin3.0-r1-mistral-24b",    "name": "Dolphin 3.0 R1",    "context": "32K"},
        ],
    },
    "google": {
        "provider": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "label": "Google Gemini", "icon": "🔵",
        "models": [
            {"id": "gemini-2.0-flash",               "name": "Gemini 2.0 Flash",     "context": "1M"},
            {"id": "gemini-2.0-flash-lite",           "name": "Gemini 2.0 Flash Lite","context": "1M"},
            {"id": "gemini-2.5-flash-preview-04-17",  "name": "Gemini 2.5 Flash",     "context": "1M"},
            {"id": "gemini-2.5-pro-preview-03-25",    "name": "Gemini 2.5 Pro",       "context": "1M"},
        ],
    },
    "zhipu": {
        "provider": "zhipu",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "label": "智谱 GLM", "icon": "🟤",
        "models": [
            {"id": "glm-5-flash", "name": "GLM-5-Flash (免费)", "context": "128K"},
            {"id": "glm-5",       "name": "GLM-5",              "context": "128K"},
            {"id": "glm-5-air",   "name": "GLM-5-Air (经济)",   "context": "128K"},
        ],
    },
    "moonshot": {
        "provider": "moonshot",
        "base_url": "https://api.moonshot.cn/v1",
        "label": "月之暗面 Kimi", "icon": "🌙",
        "models": [
            {"id": "moonshot-v1-8k",   "name": "Moonshot v1 8K",   "context": "8K"},
            {"id": "moonshot-v1-32k",  "name": "Moonshot v1 32K",  "context": "32K"},
            {"id": "moonshot-v1-128k", "name": "Moonshot v1 128K", "context": "128K"},
        ],
    },
    "alibaba": {
        "provider": "alibaba",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "label": "阿里通义千问", "icon": "☁️",
        "models": [
            {"id": "qwen-turbo-2025-04-25",  "name": "Qwen Turbo",       "context": "1M"},
            {"id": "qwen-plus-2025-04-25",   "name": "Qwen Plus",        "context": "131K"},
            {"id": "qwen-max-2025-04-25",    "name": "Qwen Max",         "context": "32K"},
            {"id": "qwen-coder-plus",        "name": "Qwen Coder Plus",  "context": "128K"},
            {"id": "qwen-coder-max-latest",  "name": "Qwen Coder Max",   "context": "128K"},
        ],
    },
    "siliconflow": {
        "provider": "custom",
        "base_url": "https://api.siliconflow.cn/v1",
        "label": "SiliconFlow (硅基流动)", "icon": "💎",
        "models": [
            {"id": "Pro/deepseek-ai/DeepSeek-V3",        "name": "DeepSeek V3",   "context": "64K"},
            {"id": "deepseek-ai/DeepSeek-R1",            "name": "DeepSeek R1",   "context": "64K"},
            {"id": "Qwen/Qwen2.5-72B-Instruct-128K",     "name": "Qwen2.5 72B",  "context": "128K"},
            {"id": "deepseek-ai/DeepSeek-V2.5",          "name": "DeepSeek V2.5","context": "128K"},
            {"id": "THUDM/glm-4-9b-chat",                "name": "GLM-4 9B",     "context": "128K"},
        ],
    },
    "groq": {
        "provider": "custom",
        "base_url": "https://api.groq.com/openai/v1",
        "label": "Groq (超快)", "icon": "⚡",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B",    "context": "128K"},
            {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B (极速)","context": "128K"},
            {"id": "mixtral-8x7b-32768",      "name": "Mixtral 8x7B",     "context": "32K"},
            {"id": "gemma2-9b-it",            "name": "Gemma 2 9B",       "context": "8K"},
        ],
    },
    "together": {
        "provider": "custom",
        "base_url": "https://api.together.xyz/v1",
        "label": "Together AI", "icon": "🟢",
        "models": [
            {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1",          "name": "Mixtral 8x22B", "context": "64K"},
            {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",        "name": "Llama 3.3 70B", "context": "128K"},
            {"id": "deepseek-ai/DeepSeek-V3",                        "name": "DeepSeek V3",   "context": "128K"},
        ],
    },
    "xai": {
        "provider": "xai",
        "base_url": "https://api.x.ai/v1",
        "label": "xAI Grok", "icon": "🕳️",
        "models": [
            {"id": "grok-2-latest",        "name": "Grok 2",        "context": "128K"},
            {"id": "grok-2-vision-latest", "name": "Grok 2 Vision", "context": "128K"},
        ],
    },
    "local": {
        "provider": "custom",
        "base_url": "http://localhost:11434/v1",
        "label": "本地 Ollama", "icon": "💻",
        "models": [
            {"id": "llama3.2:3b",    "name": "Llama 3.2 3B",    "context": "128K"},
            {"id": "qwen2.5:7b",     "name": "Qwen 2.5 7B",     "context": "128K"},
            {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B",  "context": "128K"},
        ],
    },
    "custom": {
        "provider": "custom",
        "base_url": "",
        "label": "自定义端点", "icon": "🔧",
        "models": [
            {"id": "custom-model", "name": "自定义模型", "context": "—"},
        ],
    },
}

# ─── Agent Definitions ───────────────────────────────────────────────────────

AGENTS = [
    {"id": "team-lead",         "name": "负责人",     "icon": "👑", "color": "#fbbf24"},
    {"id": "project-director",  "name": "项目总监",   "icon": "🎯", "color": "#f97316"},
    {"id": "product-manager",   "name": "产品经理",   "icon": "📋", "color": "#6366f1"},
    {"id": "designer",          "name": "设计师",     "icon": "🎨", "color": "#22d3ee"},
    {"id": "architect",         "name": "架构师",     "icon": "🏗️", "color": "#34d399"},
    {"id": "frontend-engineer", "name": "前端工程师", "icon": "⚛️", "color": "#14b8a6"},
    {"id": "backend-engineer",  "name": "后端工程师", "icon": "⚙️", "color": "#a78bfa"},
    {"id": "qa-tester",         "name": "测试工程师", "icon": "🧪", "color": "#eab308"},
    {"id": "devops",            "name": "运维",       "icon": "🐳", "color": "#ec4899"},
]
AGENT_IDS = {a["id"] for a in AGENTS}

# ─── Activity Tracking ───────────────────────────────────────────────────────

_active_sessions: dict = {}

def track_activity(agent_id: str) -> None:
    _active_sessions[agent_id] = {
        "last_active":   time.time(),
        "message_count": _active_sessions.get(agent_id, {}).get("message_count", 0) + 1,
    }

def get_agent_activity() -> dict:
    now = time.time()
    result = {}
    for a in AGENTS:
        aid     = a["id"]
        session = _active_sessions.get(aid)
        if session:
            age = now - session["last_active"]
            result[aid] = {
                "active":               age < 300,
                "last_active_seconds":  int(age),
                "message_count":        session["message_count"],
            }
        else:
            result[aid] = {"active": False, "last_active_seconds": None, "message_count": 0}
    return result

# ─── Provider → ENV var mapping ──────────────────────────────────────────────
# [FIX-5] 补全 "glm" 别名，与 zhipu 指向同一个 env var

PROVIDER_ENV_MAP = {
    "deepseek":   "DEEPSEEK_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "zhipu":      "GLM_API_KEY",
    "glm":        "GLM_API_KEY",       # [FIX-5] 补全别名
    "moonshot":   "KIMI_API_KEY",
    "alibaba":    "DASHSCOPE_API_KEY",
    "xai":        "XAI_API_KEY",
}

# ─── Agent Config Management ─────────────────────────────────────────────────

def get_agent_profile_dir(agent_id: str) -> str:
    return os.path.join(PROFILES_DIR, agent_id)


def _parse_yaml_config(config_path: str) -> dict:
    """
    [FIX-3] 用 pyyaml 解析配置文件，同时兼容 flat (model.default: x)
    和 nested (model:\\n  default: x) 两种格式。
    """
    defaults = {"model_id": "deepseek-v4-flash", "provider": "deepseek",
                "base_url": "https://api.deepseek.com"}
    if not os.path.isfile(config_path):
        return defaults

    with open(config_path, "r") as f:
        raw = f.read()

    # 先尝试 pyyaml 解析（nested 格式）
    try:
        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            model_block = data.get("model", {})
            if isinstance(model_block, dict):
                defaults["model_id"] = model_block.get("default", defaults["model_id"])
                defaults["provider"] = model_block.get("provider", defaults["provider"])
                defaults["base_url"] = model_block.get("base_url",  defaults["base_url"])
                # 如果嵌套解析成功则直接返回
                if any(k in model_block for k in ("default", "provider", "base_url")):
                    return defaults
    except yaml.YAMLError:
        pass

    # 回退：逐行解析 flat 格式（model.default: x）
    for line in raw.splitlines():
        s = line.strip()
        for key, field in [("model.default:", "model_id"),
                            ("model.provider:", "provider"),
                            ("model.base_url:", "base_url")]:
            if s.startswith(key):
                val = s[len(key):].strip().strip("\"'")
                if val:
                    defaults[field] = val

    return defaults


def get_agent_config(agent_id: str) -> dict:
    """读取 agent 的模型配置，返回时不暴露 key 原文。[FIX-4][FIX-8]"""
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir, "config.yaml")
    env_path    = os.path.join(profile_dir, ".env")

    cfg      = _parse_yaml_config(config_path)
    model_id = cfg["model_id"]
    provider = cfg["provider"]
    base_url = cfg["base_url"]

    # [FIX-4] 用 readlines 一次读完，避免 seek/fallback 混乱
    api_key_configured = False
    if os.path.isfile(env_path):
        expected_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
        with open(env_path, "r") as f:
            lines = f.readlines()

        # 优先匹配当前 provider 的 key
        for line in lines:
            line = line.strip()
            if line.startswith(expected_var + "="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val:
                    api_key_configured = True
                    break

        # 回退：任意非空 API_KEY 行
        if not api_key_configured:
            for line in lines:
                line = line.strip()
                if "API_KEY" in line and "=" in line and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        api_key_configured = True
                        break

    return {
        "model_id":          model_id,
        "provider":          provider,
        "base_url":          base_url,
        "api_key_configured": api_key_configured,  # [FIX-8] 只返回 bool，不返回 key 原文
    }


def save_agent_config(agent_id: str, model_id: str, provider: str,
                      base_url: str, api_key: str) -> dict:
    """保存模型配置和 API key 到 profile 目录。"""
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir, "config.yaml")
    env_path    = os.path.join(profile_dir, ".env")
    os.makedirs(profile_dir, exist_ok=True)

    # ── 更新 config.yaml（flat 格式写回，兼容旧文件）──
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    new_lines = []
    found = {"model": False, "provider": False, "base_url": False}
    for line in lines:
        s = line.strip()
        if s.startswith("model.default:"):
            new_lines.append(f'model.default: "{model_id}"\n'); found["model"] = True
        elif s.startswith("model.provider:"):
            new_lines.append(f'model.provider: "{provider}"\n'); found["provider"] = True
        elif s.startswith("model.base_url:"):
            new_lines.append(f'model.base_url: "{base_url}"\n'); found["base_url"] = True
        else:
            new_lines.append(line)

    if not found["model"]:    new_lines.append(f'model.default: "{model_id}"\n')
    if not found["provider"]: new_lines.append(f'model.provider: "{provider}"\n')
    if not found["base_url"]: new_lines.append(f'model.base_url: "{base_url}"\n')

    with open(config_path, "w") as f:
        f.writelines(new_lines)

    # ── 更新 .env ──
    env_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
    existing: list[str] = []
    if os.path.isfile(env_path):
        with open(env_path, "r") as f:
            existing = f.readlines()

    new_env: list[str] = []
    replaced = False
    for line in existing:
        if line.strip().startswith(env_var + "="):
            if api_key:
                new_env.append(f"{env_var}={api_key}\n")
            replaced = True
        else:
            new_env.append(line)
    if not replaced and api_key:
        new_env.append(f"{env_var}={api_key}\n")

    with open(env_path, "w") as f:
        f.writelines(new_env)

    return {"success": True, "message": f"已保存 {agent_id} 的配置"}

# ─── Chat History (SQLite) ───────────────────────────────────────────────────
# [FIX-6] chat 历史持久化到 kanban.db，进程重启后不丢失

def _ensure_chat_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id   TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at REAL    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_agent ON chat_messages(agent_id)")
    conn.commit()


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_chat_table(conn)
    return conn


def load_chat_history(agent_id: str, limit: int = 20) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_messages "
            "WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))
    finally:
        conn.close()


def append_chat_message(agent_id: str, role: str, content: str) -> None:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO chat_messages (agent_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (agent_id, role, content, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

# ─── Hermes CLI Wrapper ──────────────────────────────────────────────────────

_HERMES_BIN: str | None = None

def get_hermes_bin() -> str:
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


def call_agent(agent_id: str, user_message: str) -> dict:
    """调用 hermes CLI，上下文从 SQLite 加载。"""
    history = load_chat_history(agent_id, limit=20)   # [FIX-6]

    ctx_lines = [
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:500]}"
        for m in history
    ]
    context_str = (
        "以下是本次对话的最近历史记录，请参考上下文回答：\n"
        + "\n".join(ctx_lines)
        + "\n\n---\n\n"
        if ctx_lines else ""
    )
    full_prompt = context_str + user_message

    try:
        proc = subprocess.run(
            [get_hermes_bin(), "--yolo", "-p", agent_id, "chat", "-q", full_prompt, "--quiet"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "HERMES_QUIET": "1", "TERM": "dumb"},
        )
        response = proc.stdout.strip() or proc.stderr.strip() or "(Agent 无响应)"
        for pattern in [
            r"^\s*━━━.*━━━\s*$", r"^\s*─+.*─+\s*$",
            r"^✦\s+.*", r"^╭─.*", r"^╰─.*", r"^│.*",
        ]:
            response = re.sub(pattern, "", response, flags=re.MULTILINE)
        response = re.sub(r"^session_id:\s+\S+\s*\n?", "", response).strip()
        success  = proc.returncode == 0 and bool(response)
    except subprocess.TimeoutExpired:
        response = f"(⏱️ Agent {agent_id} 超时，180秒未响应)"
        success  = False
    except Exception as e:
        response = f"(❌ 错误: {e})"
        success  = False

    # 持久化 [FIX-6]
    append_chat_message(agent_id, "user",      user_message)
    append_chat_message(agent_id, "assistant", response)
    track_activity(agent_id)

    return {
        "success":    success,
        "response":   response,
        "agent":      agent_id,
        "agent_name": next((a["name"] for a in AGENTS if a["id"] == agent_id), agent_id),
    }


def forward_to_agent(from_agent: str, to_agent: str, message: str) -> dict:
    if to_agent not in AGENT_IDS:
        return {"success": False, "response": f"未知目标Agent: {to_agent}", "agent": to_agent}

    from_name  = next((a["name"] for a in AGENTS if a["id"] == from_agent), from_agent)
    to_name    = next((a["name"] for a in AGENTS if a["id"] == to_agent),   to_agent)
    forward_msg = (
        f"【任务协调 - 来自 {from_name}({from_agent}) → 转交 {to_name}({to_agent})】\n\n"
        f"上游分析/结论：\n{message}\n\n"
        f"请基于上述信息执行你的职责范围内的分析，给出具体建议或方案。"
        f"如果需要补充信息或需要其他Agent协助，请明确指出。"
    )
    track_activity(from_agent)
    result = call_agent(to_agent, forward_msg)
    track_activity(to_agent)
    return result

# ─── Kanban DB Queries ───────────────────────────────────────────────────────

def get_all_tasks() -> list[dict]:
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
            tid  = r["id"]
            runs = conn.execute("""
                SELECT id, profile, status, started_at, ended_at, outcome, summary, error
                FROM task_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 10
            """, (tid,)).fetchall()
            task["runs"]          = [dict(run) for run in runs]
            task["comment_count"] = conn.execute(
                "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,)
            ).fetchone()[0]
            task["parents"]  = [p["parent_id"] for p in
                                 conn.execute("SELECT parent_id FROM task_links WHERE child_id  = ?", (tid,)).fetchall()]
            task["children"] = [c["child_id"]  for c in
                                 conn.execute("SELECT child_id  FROM task_links WHERE parent_id = ?", (tid,)).fetchall()]
            tasks.append(task)
        return tasks
    finally:
        conn.close()


def get_stats() -> dict:
    conn = get_db()
    try:
        stats = {}
        for status in ["todo", "ready", "running", "blocked", "done", "archived"]:
            stats[status] = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = ?", (status,)
            ).fetchone()[0]
        stats["total"] = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        profiles_rows = conn.execute("""
            SELECT assignee, status, COUNT(*) as cnt
            FROM tasks WHERE assignee IS NOT NULL GROUP BY assignee, status
        """).fetchall()
        profile_stats: dict = {}
        for p in profiles_rows:
            name = p["assignee"]
            profile_stats.setdefault(name, {})[p["status"]] = p["cnt"]
        stats["profiles"] = profile_stats
        return stats
    finally:
        conn.close()


def get_events(limit: int = 50) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT id, task_id, run_id, kind, payload, created_at
            FROM task_events ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ─── HTTP Handler ────────────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):

    # [FIX-2] 根据请求 Origin 决定允许的 CORS 来源
    def _cors_origin(self) -> str:
        origin = self.headers.get("Origin", "")
        # 提取 scheme + host（不含端口以外部分）
        base = origin.rsplit(":", 1)[0] if origin.count(":") >= 2 else origin
        if base in ALLOWED_ORIGINS or not origin:
            return origin or "http://localhost"
        return "http://localhost"   # 非白名单来源降级为 localhost

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

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
        elif path == "/api/agents/status":
            status_list = []
            for a in AGENTS:
                cfg = get_agent_config(a["id"])
                status_list.append({
                    "id":         a["id"],
                    "name":       a["name"],
                    "icon":       a["icon"],
                    "configured": cfg["api_key_configured"],
                    "model":      cfg["model_id"],
                    "provider":   cfg["provider"],
                })
            self.send_json({"agents": status_list})
        elif path == "/api/agents/activity":
            self.send_json(get_agent_activity())

        # [FIX-7] 所有 agent 路由在拆出 agent_id 后立即校验
        elif re.match(r"^/api/chat/[\w-]+/history$", path):
            agent_id = path.split("/")[3]
            if agent_id not in AGENT_IDS:
                self.send_json({"error": "未知Agent"}, 404); return
            msgs = load_chat_history(agent_id, limit=50)
            self.send_json({"id": agent_id, "messages": msgs})

        elif re.match(r"^/api/chat/[\w-]+/config$", path):
            agent_id = path.split("/")[3]
            if agent_id not in AGENT_IDS:
                self.send_json({"error": "未知Agent"}, 404); return
            self.send_json(get_agent_config(agent_id))

        elif path in ("/", "/index.html"):
            self.serve_file("index.html", "text/html")
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # POST /api/chat/<agent>
        m = re.match(r"^/api/chat/([\w-]+)$", path)
        if m:
            agent_id = m.group(1)
            # [FIX-7]
            if agent_id not in AGENT_IDS:
                self.send_json({"success": False, "error": f"未知Agent: {agent_id}"}, 404); return
            body = self.read_body()
            user_msg = (body.get("message") or "").strip()
            if not user_msg:
                self.send_json({"success": False, "error": "message 不能为空"}, 400); return
            self.send_json(call_agent(agent_id, user_msg))
            return

        # POST /api/chat/<agent>/forward
        m = re.match(r"^/api/chat/([\w-]+)/forward$", path)
        if m:
            from_agent = m.group(1)
            if from_agent not in AGENT_IDS:
                self.send_json({"success": False, "error": f"未知 from-agent: {from_agent}"}, 404); return
            body = self.read_body()
            if "to" not in body or "message" not in body:
                self.send_json({"success": False, "error": "需要 to 和 message 字段"}, 400); return
            self.send_json(forward_to_agent(from_agent, body["to"], body["message"]))
            return

        # POST /api/chat/<agent>/config
        m = re.match(r"^/api/chat/([\w-]+)/config$", path)
        if m:
            agent_id = m.group(1)
            if agent_id not in AGENT_IDS:
                self.send_json({"success": False, "error": f"未知Agent: {agent_id}"}, 404); return
            body     = self.read_body()
            model_id = body.get("model_id",  "deepseek-chat")
            provider = body.get("provider",  "deepseek")
            base_url = body.get("base_url",  "https://api.deepseek.com")
            api_key  = body.get("api_key",   "")
            self.send_json(save_agent_config(agent_id, model_id, provider, base_url, api_key))
            return

        self.send_json({"success": False, "error": "未找到"}, 404)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # [FIX-2] 受限的 CORS 头
        self.send_header("Access-Control-Allow-Origin",  self._cors_origin())
        self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, filename: str, content_type: str) -> None:
        filepath = os.path.join(STATIC_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type",   content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        # [FIX-2]
        self.send_header("Access-Control-Allow-Origin",  self._cors_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def log_message(self, format, *args):
        pass   # 静默日志；如需调试可注释掉此行


# [FIX-1] 多线程服务器，防止长请求阻塞其他连接
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port   = int(os.environ.get("DASHBOARD_PORT", 8765))
    server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)

    print("╔══════════════════════════════════════════╗")
    print("║     Hermes Multi-Agent Dashboard         ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Dashboard : http://localhost:{port}      ║")
    print(f"║  Agents    : {len(AGENTS)} configured              ║")
    print(f"║  Providers : {len(MODEL_PRESETS)} available              ║")
    print("║  Chat API  : /api/chat/<agent>           ║")
    print("║  Config API: /api/chat/<agent>/config    ║")
    print("╚══════════════════════════════════════════╝")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
