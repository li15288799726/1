#!/usr/bin/env python3
"""
Hermes Kanban Dashboard + Multi-Agent Chat API + Autonomous Pipeline

新增功能:
  [PIPELINE] 自动流水线引擎 — team-lead 接收指令后自动分工，
             各 Agent 依次协作，project-director 审核，无需人工介入。

API:
  POST /api/pipeline/start          — 启动新流水线
  GET  /api/pipeline/<id>/stream    — SSE 实时进度流
  GET  /api/pipeline/<id>/status    — 查询流水线状态
  GET  /api/pipelines               — 所有流水线列表

历史修复:
  [FIX-1] ThreadingMixIn 多线程
  [FIX-2] CORS 限制 localhost
  [FIX-3] pyyaml 解析配置
  [FIX-4] API key 读取 bug
  [FIX-5] glm provider 别名
  [FIX-6] chat 历史持久化 SQLite
  [FIX-7] 路由严格校验 agent_id
  [FIX-8] 不暴露 key 原文
"""

import sqlite3
import json
import os
import re
import time
import threading
import queue
import uuid
import subprocess
import yaml
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

DB_PATH      = os.path.expanduser("~/.hermes/kanban.db")
STATIC_DIR   = os.path.dirname(os.path.abspath(__file__))
HERMES_HOME  = os.path.expanduser("~/.hermes")
PROFILES_DIR = os.path.join(HERMES_HOME, "profiles")

ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}

# ─── Model Presets ────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    "deepseek":   {"provider":"deepseek",  "base_url":"https://api.deepseek.com",                         "label":"DeepSeek",           "icon":"🧠",
                   "models":[{"id":"deepseek-v4-flash","name":"DeepSeek V4 Flash (推荐)","context":"64K"},
                              {"id":"deepseek-reasoner","name":"DeepSeek R1 (推理)","context":"64K"}]},
    "openai":     {"provider":"openai",    "base_url":"https://api.openai.com/v1",                         "label":"OpenAI",             "icon":"🤖",
                   "models":[{"id":"gpt-4o","name":"GPT-4o","context":"128K"},
                              {"id":"gpt-4o-mini","name":"GPT-4o Mini","context":"128K"},
                              {"id":"gpt-4.1","name":"GPT-4.1","context":"1M"}]},
    "anthropic":  {"provider":"anthropic", "base_url":"https://api.anthropic.com",                         "label":"Anthropic",          "icon":"🔮",
                   "models":[{"id":"claude-sonnet-4-20250514","name":"Claude Sonnet 4","context":"200K"},
                              {"id":"claude-haiku-3-5","name":"Claude Haiku 3.5","context":"200K"}]},
    "openrouter": {"provider":"openrouter","base_url":"https://openrouter.ai/api/v1",                      "label":"OpenRouter",         "icon":"🌐",
                   "models":[{"id":"anthropic/claude-sonnet-4","name":"Claude Sonnet 4","context":"200K"},
                              {"id":"deepseek/deepseek-chat","name":"DeepSeek V3","context":"128K"},
                              {"id":"google/gemini-2.0-flash-001","name":"Gemini 2.0 Flash","context":"1M"}]},
    "google":     {"provider":"google",    "base_url":"https://generativelanguage.googleapis.com/v1beta",  "label":"Google Gemini",      "icon":"🔵",
                   "models":[{"id":"gemini-2.5-flash-preview-04-17","name":"Gemini 2.5 Flash","context":"1M"},
                              {"id":"gemini-2.5-pro-preview-03-25","name":"Gemini 2.5 Pro","context":"1M"}]},
    "zhipu":      {"provider":"zhipu",     "base_url":"https://open.bigmodel.cn/api/paas/v4",             "label":"智谱 GLM",           "icon":"🟤",
                   "models":[{"id":"glm-5-flash","name":"GLM-5-Flash (免费)","context":"128K"},
                              {"id":"glm-5","name":"GLM-5","context":"128K"}]},
    "moonshot":   {"provider":"moonshot",  "base_url":"https://api.moonshot.cn/v1",                       "label":"月之暗面 Kimi",       "icon":"🌙",
                   "models":[{"id":"moonshot-v1-128k","name":"Moonshot v1 128K","context":"128K"}]},
    "alibaba":    {"provider":"alibaba",   "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","label":"阿里通义千问",        "icon":"☁️",
                   "models":[{"id":"qwen-plus-2025-04-25","name":"Qwen Plus","context":"131K"},
                              {"id":"qwen-coder-plus","name":"Qwen Coder Plus","context":"128K"}]},
    "groq":       {"provider":"custom",    "base_url":"https://api.groq.com/openai/v1",                   "label":"Groq (超快)",         "icon":"⚡",
                   "models":[{"id":"llama-3.3-70b-versatile","name":"Llama 3.3 70B","context":"128K"}]},
    "local":      {"provider":"custom",    "base_url":"http://localhost:11434/v1",                         "label":"本地 Ollama",         "icon":"💻",
                   "models":[{"id":"qwen2.5:7b","name":"Qwen 2.5 7B","context":"128K"}]},
    "custom":     {"provider":"custom",    "base_url":"",                                                  "label":"自定义端点",          "icon":"🔧",
                   "models":[{"id":"custom-model","name":"自定义模型","context":"—"}]},
}

# ─── Agent Definitions ────────────────────────────────────────────────────────

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
AGENT_IDS   = {a["id"] for a in AGENTS}
AGENT_META  = {a["id"]: a for a in AGENTS}

# ─── Activity Tracking ────────────────────────────────────────────────────────

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
            result[aid] = {"active": age < 300, "last_active_seconds": int(age),
                           "message_count": session["message_count"]}
        else:
            result[aid] = {"active": False, "last_active_seconds": None, "message_count": 0}
    return result

# ─── Provider ENV Map ─────────────────────────────────────────────────────────

PROVIDER_ENV_MAP = {
    "deepseek":   "DEEPSEEK_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "zhipu":      "GLM_API_KEY",
    "glm":        "GLM_API_KEY",
    "moonshot":   "KIMI_API_KEY",
    "alibaba":    "DASHSCOPE_API_KEY",
    "xai":        "XAI_API_KEY",
}

# ─── Agent Config ─────────────────────────────────────────────────────────────

def get_agent_profile_dir(agent_id: str) -> str:
    return os.path.join(PROFILES_DIR, agent_id)

def _parse_yaml_config(config_path: str) -> dict:
    defaults = {"model_id": "deepseek-v4-flash", "provider": "deepseek",
                "base_url": "https://api.deepseek.com"}
    if not os.path.isfile(config_path):
        return defaults
    with open(config_path) as f:
        raw = f.read()
    try:
        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            mb = data.get("model", {})
            if isinstance(mb, dict) and any(k in mb for k in ("default","provider","base_url")):
                defaults["model_id"] = mb.get("default",  defaults["model_id"])
                defaults["provider"] = mb.get("provider", defaults["provider"])
                defaults["base_url"] = mb.get("base_url", defaults["base_url"])
                return defaults
    except yaml.YAMLError:
        pass
    for line in raw.splitlines():
        s = line.strip()
        for key, field in [("model.default:","model_id"),("model.provider:","provider"),("model.base_url:","base_url")]:
            if s.startswith(key):
                val = s[len(key):].strip().strip("\"'")
                if val: defaults[field] = val
    return defaults

def get_agent_config(agent_id: str) -> dict:
    profile_dir = get_agent_profile_dir(agent_id)
    cfg      = _parse_yaml_config(os.path.join(profile_dir, "config.yaml"))
    env_path = os.path.join(profile_dir, ".env")
    api_key_configured = False
    if os.path.isfile(env_path):
        expected_var = PROVIDER_ENV_MAP.get(cfg["provider"], "CUSTOM_API_KEY")
        with open(env_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith(expected_var + "=") and line.split("=",1)[1].strip().strip("\"'"):
                api_key_configured = True; break
        if not api_key_configured:
            for line in lines:
                line = line.strip()
                if "API_KEY" in line and "=" in line and not line.startswith("#"):
                    if line.split("=",1)[1].strip().strip("\"'"):
                        api_key_configured = True; break
    return {**cfg, "api_key_configured": api_key_configured}

def save_agent_config(agent_id: str, model_id: str, provider: str, base_url: str, api_key: str) -> dict:
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir, "config.yaml")
    env_path    = os.path.join(profile_dir, ".env")
    os.makedirs(profile_dir, exist_ok=True)
    lines = open(config_path).readlines() if os.path.isfile(config_path) else []
    new_lines, found = [], {"model":False,"provider":False,"base_url":False}
    for line in lines:
        s = line.strip()
        if s.startswith("model.default:"):    new_lines.append(f'model.default: "{model_id}"\n');  found["model"] = True
        elif s.startswith("model.provider:"): new_lines.append(f'model.provider: "{provider}"\n'); found["provider"] = True
        elif s.startswith("model.base_url:"): new_lines.append(f'model.base_url: "{base_url}"\n'); found["base_url"] = True
        else:                                  new_lines.append(line)
    if not found["model"]:    new_lines.append(f'model.default: "{model_id}"\n')
    if not found["provider"]: new_lines.append(f'model.provider: "{provider}"\n')
    if not found["base_url"]: new_lines.append(f'model.base_url: "{base_url}"\n')
    open(config_path,"w").writelines(new_lines)
    env_var  = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
    existing = open(env_path).readlines() if os.path.isfile(env_path) else []
    new_env, replaced = [], False
    for line in existing:
        if line.strip().startswith(env_var + "="):
            if api_key: new_env.append(f"{env_var}={api_key}\n")
            replaced = True
        else:
            new_env.append(line)
    if not replaced and api_key: new_env.append(f"{env_var}={api_key}\n")
    open(env_path,"w").writelines(new_env)
    return {"success": True, "message": f"已保存 {agent_id} 的配置"}

# ─── SQLite ───────────────────────────────────────────────────────────────────

def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id   TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_agent ON chat_messages(agent_id);

        CREATE TABLE IF NOT EXISTS pipelines (
            id          TEXT PRIMARY KEY,
            goal        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'running',
            created_at  REAL NOT NULL,
            finished_at REAL
        );
        CREATE TABLE IF NOT EXISTS pipeline_steps (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id  TEXT NOT NULL,
            step_index   INTEGER NOT NULL,
            agent_id     TEXT NOT NULL,
            role_prompt  TEXT NOT NULL,
            input        TEXT NOT NULL,
            output       TEXT,
            status       TEXT NOT NULL DEFAULT 'pending',
            started_at   REAL,
            finished_at  REAL,
            FOREIGN KEY(pipeline_id) REFERENCES pipelines(id)
        );
        CREATE INDEX IF NOT EXISTS idx_steps_pipeline ON pipeline_steps(pipeline_id);
    """)
    conn.commit()

def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn

_db_lock = threading.Lock()

def db_exec(sql: str, params=()) -> list[dict]:
    with _db_lock:
        conn = get_db()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return [dict(r) for r in (cur.fetchall() if cur.description else [])]
        finally:
            conn.close()

def load_chat_history(agent_id: str, limit: int = 20) -> list[dict]:
    rows = db_exec(
        "SELECT role,content,created_at FROM chat_messages "
        "WHERE agent_id=? ORDER BY created_at DESC LIMIT ?", (agent_id, limit))
    return list(reversed(rows))

def append_chat_message(agent_id: str, role: str, content: str) -> None:
    db_exec("INSERT INTO chat_messages(agent_id,role,content,created_at) VALUES(?,?,?,?)",
            (agent_id, role, content, time.time()))

# ─── Hermes CLI ───────────────────────────────────────────────────────────────

_HERMES_BIN: str | None = None

def get_hermes_bin() -> str:
    global _HERMES_BIN
    if _HERMES_BIN is None:
        for p in os.environ.get("PATH","").split(os.pathsep):
            c = os.path.join(p, "hermes")
            if os.path.isfile(c) and os.access(c, os.X_OK):
                _HERMES_BIN = c; break
        if _HERMES_BIN is None: _HERMES_BIN = "hermes"
    return _HERMES_BIN

def _run_hermes(agent_id: str, prompt: str) -> tuple[bool, str]:
    """调用 hermes CLI，返回 (success, response)"""
    try:
        proc = subprocess.run(
            [get_hermes_bin(), "--yolo", "-p", agent_id, "chat", "-q", prompt, "--quiet"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "HERMES_QUIET": "1", "TERM": "dumb"})
        response = proc.stdout.strip() or proc.stderr.strip() or "(Agent 无响应)"
        for pat in [r"^\s*━━━.*━━━\s*$", r"^\s*─+.*─+\s*$",
                    r"^✦\s+.*", r"^╭─.*", r"^╰─.*", r"^│.*"]:
            response = re.sub(pat, "", response, flags=re.MULTILINE)
        response = re.sub(r"^session_id:\s+\S+\s*\n?", "", response).strip()
        return proc.returncode == 0 and bool(response), response
    except subprocess.TimeoutExpired:
        return False, f"(⏱️ {agent_id} 超时 180s)"
    except Exception as e:
        return False, f"(❌ {e})"


def call_agent(agent_id: str, user_message: str) -> dict:
    """普通调用（非 team-lead 或单条聊天）"""
    history   = load_chat_history(agent_id, limit=20)
    ctx_lines = [f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:500]}" for m in history]
    context_str = ("以下是本次对话的最近历史记录：\n" + "\n".join(ctx_lines) + "\n\n---\n\n") if ctx_lines else ""
    success, response = _run_hermes(agent_id, context_str + user_message)
    append_chat_message(agent_id, "user",      user_message)
    append_chat_message(agent_id, "assistant", response)
    track_activity(agent_id)
    return {"success": success, "response": response, "agent": agent_id,
            "agent_name": AGENT_META.get(agent_id, {}).get("name", agent_id)}


def forward_to_agent(from_agent: str, to_agent: str, message: str) -> dict:
    if to_agent not in AGENT_IDS:
        return {"success": False, "response": f"未知目标Agent: {to_agent}", "agent": to_agent}
    from_name = AGENT_META.get(from_agent, {}).get("name", from_agent)
    to_name   = AGENT_META.get(to_agent,   {}).get("name", to_agent)
    msg = (f"【任务协调 - 来自 {from_name} → 转交 {to_name}】\n\n"
           f"上游输出：\n{message}\n\n"
           f"请基于上述信息执行你职责范围内的工作，给出具体方案或产出。")
    track_activity(from_agent)
    result = call_agent(to_agent, msg)
    track_activity(to_agent)
    return result


# ─── Team-Lead 自动分工引擎 ───────────────────────────────────────────────────
#
# 用户直接跟 team-lead 聊天 → team-lead 自动分工 → 各 Agent 依次完成工作
# → project-director 验收 → team-lead 汇总回复给用户
#
# 所有进度通过 SSE 实时推送到 team-lead 的聊天窗口。

DELEGATION_PROMPTS = {
    "team-lead-kickoff": (
        "你是团队负责人（Team Lead）。用户对你提出了以下目标：\n\n"
        "【用户目标】{goal}\n\n"
        "请拆解目标，写一份清晰的项目启动令（含范围、验收标准），准备交给项目总监。"
        "直接输出启动令，不要废话。"
    ),
    "project-director": (
        "你是项目总监。收到负责人的启动令：\n\n{prev}\n\n"
        "请制定：阶段划分、各角色职责、技术风险及应对。"
    ),
    "product-manager": (
        "你是产品经理。项目背景：\n\n{prev}\n\n"
        "请输出：用户故事、功能清单、界面原型描述、数据结构定义。要可直接指导开发。"
    ),
    "architect": (
        "你是架构师。需求如下：\n\n{prev}\n\n"
        "请输出：技术选型、模块划分、核心数据结构与算法、文件结构、关键实现要点。"
    ),
    "designer": (
        "你是设计师。方案如下：\n\n{prev}\n\n"
        "请输出：配色方案（具体色值）、布局结构、交互动效、关键 CSS 样式要点。"
    ),
    "backend-engineer": (
        "你是后端工程师。需求和架构：\n\n{prev}\n\n"
        "请直接输出完整可运行的后端代码（含依赖说明）。只输出代码块。"
    ),
    "frontend-engineer": (
        "你是前端工程师。以下是完整需求和设计：\n\n{prev}\n\n"
        "请直接输出完整可运行的单文件 HTML（内联 CSS+JS），能直接浏览器打开运行。只输出代码块。"
    ),
    "qa-tester": (
        "你是测试工程师。项目全部输出：\n\n{prev}\n\n"
        "请列测试用例、检查明显 Bug、给出总体评价：【通过】或【需整改】。"
    ),
    "project-director-review": (
        "你是项目总监，做最终验收。\n目标：{goal}\n\n所有输出：\n{prev}\n\n"
        "逐项核对验收条件，给出明确结论：【验收通过】或【需返工】（需返工时说明具体问题）。"
    ),
    "team-lead-summary": (
        "你是团队负责人（Team Lead）。项目已完成，完整交付物如下：\n\n{prev}\n\n"
        "用户原始目标：{goal}\n\n"
        "请写一份面向用户的交付报告：完成情况摘要、主要交付物（代码完整列出）、使用说明、后续建议。"
        "语言友好简明，这就是你回复用户的最终内容。"
    ),
}

def _build_steps(goal: str) -> list[dict]:
    """根据目标动态决定需要哪些 Agent"""
    needs_backend = any(w in goal for w in ["后端","API","数据库","服务器","接口","登录","认证","存储"])
    needs_design  = any(w in goal for w in ["游戏","网页","界面","UI","前端","可视化","动画","页面"])
    steps = [
        {"agent": "team-lead",        "prompt_key": "team-lead-kickoff",        "label": "拆解目标，起草启动令"},
        {"agent": "project-director", "prompt_key": "project-director",         "label": "制定项目计划"},
        {"agent": "product-manager",  "prompt_key": "product-manager",          "label": "编写产品需求"},
        {"agent": "architect",        "prompt_key": "architect",                "label": "设计技术方案"},
    ]
    if needs_design:
        steps.append({"agent": "designer", "prompt_key": "designer", "label": "UI/UX 设计"})
    if needs_backend:
        steps.append({"agent": "backend-engineer", "prompt_key": "backend-engineer", "label": "后端开发"})
    steps += [
        {"agent": "frontend-engineer", "prompt_key": "frontend-engineer",       "label": "前端开发"},
        {"agent": "qa-tester",         "prompt_key": "qa-tester",               "label": "测试 & 质检"},
        {"agent": "project-director",  "prompt_key": "project-director-review", "label": "项目总监验收"},
        {"agent": "team-lead",         "prompt_key": "team-lead-summary",       "label": "汇总，回复用户"},
    ]
    return steps


# SSE 订阅管理（以 chat session id 为 key）
_tl_subscribers: dict[str, list[queue.Queue]] = {}
_tl_lock = threading.Lock()

def _tl_publish(session_id: str, event: dict) -> None:
    with _tl_lock:
        qs = list(_tl_subscribers.get(session_id, []))
    for q in qs:
        try: q.put_nowait(event)
        except queue.Full: pass

def _tl_subscribe(session_id: str) -> "queue.Queue[dict]":
    q: queue.Queue[dict] = queue.Queue(maxsize=300)
    with _tl_lock:
        _tl_subscribers.setdefault(session_id, []).append(q)
    return q

def _tl_unsubscribe(session_id: str, q: "queue.Queue[dict]") -> None:
    with _tl_lock:
        subs = _tl_subscribers.get(session_id, [])
        if q in subs: subs.remove(q)


def _run_delegation(session_id: str, goal: str) -> None:
    """后台线程：team-lead 自动分工协作"""
    steps   = _build_steps(goal)
    outputs: list[dict] = []   # {"agent": id, "output": str}
    retried = False

    _tl_publish(session_id, {
        "type": "start", "goal": goal,
        "total": len(steps), "timestamp": time.time(),
    })

    step_idx = 0
    while step_idx < len(steps):
        step     = steps[step_idx]
        agent_id = step["agent"]
        label    = step["label"]
        am       = AGENT_META.get(agent_id, {"name": agent_id, "icon": "🤖", "color": "#818cf8"})

        # 构建累积上下文
        prev_ctx = "\n\n".join(
            f"【{AGENT_META.get(o['agent'],{}).get('name', o['agent'])} 输出】\n{o['output']}"
            for o in outputs
        ) or "(无前序输出)"

        prompt = DELEGATION_PROMPTS[step["prompt_key"]].format(goal=goal, prev=prev_ctx)

        _tl_publish(session_id, {
            "type": "step_start", "step": step_idx, "total": len(steps),
            "agent_id": agent_id, "agent_name": am["name"],
            "agent_icon": am["icon"], "agent_color": am.get("color","#818cf8"),
            "label": label, "timestamp": time.time(),
        })

        success, output = _run_hermes(agent_id, prompt)
        append_chat_message(agent_id, "user",      prompt)
        append_chat_message(agent_id, "assistant", output)
        track_activity(agent_id)

        outputs.append({"agent": agent_id, "output": output})

        _tl_publish(session_id, {
            "type": "step_done", "step": step_idx, "total": len(steps),
            "agent_id": agent_id, "agent_name": am["name"],
            "agent_icon": am["icon"], "agent_color": am.get("color","#818cf8"),
            "label": label, "output": output, "success": success,
            "timestamp": time.time(),
        })

        # project-director 验收：若不通过且未重试，退回 engineer
        if step["prompt_key"] == "project-director-review" and not retried:
            if any(kw in output for kw in ["需返工","需要返工","不通过","reject","Reject"]):
                retried = True
                eng_steps = [s for s in steps if "engineer" in s["agent"]]
                if eng_steps:
                    rework = dict(eng_steps[-1])
                    rework["label"] = f"🔁 {rework['label']}（返工）"
                    steps.insert(step_idx + 1, rework)
                    steps.insert(step_idx + 2, dict(steps[step_idx]))
                    _tl_publish(session_id, {
                        "type": "rework",
                        "reason": "验收未通过，自动退回重做",
                        "timestamp": time.time(),
                    })

        step_idx += 1

    # 最终输出 = team-lead 的汇总（最后一步）
    final = outputs[-1]["output"] if outputs else "（无输出）"

    # 存入 team-lead 的聊天历史，用户在聊天窗口看到的最终回复
    append_chat_message("team-lead", "user",      goal)
    append_chat_message("team-lead", "assistant", final)

    _tl_publish(session_id, {
        "type": "done", "final": final, "timestamp": time.time(),
    })


def start_delegation(goal: str) -> str:
    """启动 team-lead 自动分工，返回 session_id 供 SSE 订阅"""
    session_id = str(uuid.uuid4())
    t = threading.Thread(target=_run_delegation, args=(session_id, goal), daemon=True)
    t.start()
    return session_id


# 保留旧接口兼容（非 team-lead 的 pipeline）
def start_pipeline(goal: str) -> str:
    return start_delegation(goal)

# ─── Kanban DB Queries ────────────────────────────────────────────────────────

def get_all_tasks() -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT id,title,body,assignee,status,priority,created_by,created_at,
                   started_at,completed_at,workspace_kind,workspace_path,result,
                   consecutive_failures,last_failure_error,current_run_id,skills,model_override
            FROM tasks ORDER BY created_at DESC
        """).fetchall()
        tasks = []
        for r in rows:
            task = dict(r); tid = r["id"]
            runs = conn.execute(
                "SELECT id,profile,status,started_at,ended_at,outcome,summary,error "
                "FROM task_runs WHERE task_id=? ORDER BY started_at DESC LIMIT 10",(tid,)).fetchall()
            task["runs"]          = [dict(run) for run in runs]
            task["comment_count"] = conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id=?",(tid,)).fetchone()[0]
            task["parents"]  = [p["parent_id"] for p in conn.execute("SELECT parent_id FROM task_links WHERE child_id=?" ,(tid,)).fetchall()]
            task["children"] = [c["child_id"]  for c in conn.execute("SELECT child_id  FROM task_links WHERE parent_id=?",(tid,)).fetchall()]
            tasks.append(task)
        return tasks
    finally:
        conn.close()

def get_stats() -> dict:
    conn = get_db()
    try:
        stats = {}
        for s in ["todo","ready","running","blocked","done","archived"]:
            stats[s] = conn.execute("SELECT COUNT(*) FROM tasks WHERE status=?",(s,)).fetchone()[0]
        stats["total"] = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        prows = conn.execute("SELECT assignee,status,COUNT(*) as cnt FROM tasks WHERE assignee IS NOT NULL GROUP BY assignee,status").fetchall()
        ps = {}
        for p in prows:
            ps.setdefault(p["assignee"],{})[p["status"]] = p["cnt"]
        stats["profiles"] = ps
        return stats
    finally:
        conn.close()

def get_events(limit: int = 50) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute("SELECT id,task_id,run_id,kind,payload,created_at FROM task_events ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):

    def _cors_origin(self) -> str:
        origin = self.headers.get("Origin","")
        base   = origin.rsplit(":",1)[0] if origin.count(":") >= 2 else origin
        return origin if base in ALLOWED_ORIGINS else "http://localhost"

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # Kanban
        if   path == "/api/tasks":            self.send_json(get_all_tasks())
        elif path == "/api/stats":            self.send_json(get_stats())
        elif path == "/api/events":           self.send_json(get_events())
        elif path == "/api/health":           self.send_json({"status":"ok","db":os.path.exists(DB_PATH),"time":int(time.time())})

        # Agents
        elif path == "/api/agents":           self.send_json({"agents":AGENTS,"total":len(AGENTS)})
        elif path == "/api/model-presets":    self.send_json(MODEL_PRESETS)
        elif path == "/api/agents/status":
            self.send_json({"agents":[{**a,"configured":get_agent_config(a["id"])["api_key_configured"],
                                       "model":get_agent_config(a["id"])["model_id"],
                                       "provider":get_agent_config(a["id"])["provider"]} for a in AGENTS]})
        elif path == "/api/agents/activity":  self.send_json(get_agent_activity())

        # Chat history / config
        elif re.match(r"^/api/chat/[\w-]+/history$", path):
            aid = path.split("/")[3]
            if aid not in AGENT_IDS: self.send_json({"error":"未知Agent"},404); return
            self.send_json({"id":aid,"messages":load_chat_history(aid,50)})
        elif re.match(r"^/api/chat/[\w-]+/config$", path):
            aid = path.split("/")[3]
            if aid not in AGENT_IDS: self.send_json({"error":"未知Agent"},404); return
            self.send_json(get_agent_config(aid))

        # ── Pipeline ──────────────────────────────────────────────────────
        elif path == "/api/pipelines":
            rows = db_exec("SELECT id,goal,status,created_at,finished_at FROM pipelines ORDER BY created_at DESC LIMIT 50")
            self.send_json(rows)

        elif re.match(r"^/api/pipeline/[\w-]+/status$", path):
            pid  = path.split("/")[3]
            rows = db_exec("SELECT * FROM pipelines WHERE id=?", (pid,))
            if not rows: self.send_json({"error":"未找到"},404); return
            steps = db_exec("SELECT * FROM pipeline_steps WHERE pipeline_id=? ORDER BY step_index",(pid,))
            self.send_json({**rows[0], "steps": steps})

        elif re.match(r"^/api/pipeline/[\w-]+/stream$", path):
            pid = path.split("/")[3]
            self._sse_stream(pid)
            return

        # team-lead 分工 SSE 流
        elif re.match(r"^/api/chat/team-lead/stream/[\w-]+$", path):
            session_id = path.split("/")[-1]
            self._tl_sse_stream(session_id)
            return

        elif path in ("/","/index.html"):     self.serve_file("index.html","text/html")
        else:                                  super().do_GET()

    # ── POST ─────────────────────────────────────────────────────────────────

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        # Chat
        m = re.match(r"^/api/chat/([\w-]+)$", path)
        if m:
            aid = m.group(1)
            if aid not in AGENT_IDS: self.send_json({"success":False,"error":f"未知Agent: {aid}"},404); return
            body = self.read_body(); msg = (body.get("message") or "").strip()
            if not msg: self.send_json({"success":False,"error":"message 不能为空"},400); return
            # team-lead 自动分工：返回 SSE stream url，前端实时看进度
            if aid == "team-lead":
                session_id = start_delegation(msg)
                self.send_json({"success":True,"delegating":True,
                                "session_id":session_id,
                                "stream_url":f"/api/chat/team-lead/stream/{session_id}"}); return
            self.send_json(call_agent(aid, msg)); return

        # Forward
        m = re.match(r"^/api/chat/([\w-]+)/forward$", path)
        if m:
            fa = m.group(1)
            if fa not in AGENT_IDS: self.send_json({"success":False,"error":f"未知Agent: {fa}"},404); return
            body = self.read_body()
            if "to" not in body or "message" not in body:
                self.send_json({"success":False,"error":"需要 to 和 message"},400); return
            self.send_json(forward_to_agent(fa, body["to"], body["message"])); return

        # Config
        m = re.match(r"^/api/chat/([\w-]+)/config$", path)
        if m:
            aid = m.group(1)
            if aid not in AGENT_IDS: self.send_json({"success":False,"error":f"未知Agent: {aid}"},404); return
            body = self.read_body()
            self.send_json(save_agent_config(
                aid, body.get("model_id","deepseek-chat"), body.get("provider","deepseek"),
                body.get("base_url","https://api.deepseek.com"), body.get("api_key",""))); return

        # ── Pipeline start ─────────────────────────────────────────────────
        if path == "/api/pipeline/start":
            body = self.read_body()
            goal = (body.get("goal") or body.get("message") or "").strip()
            if not goal:
                self.send_json({"success":False,"error":"请提供 goal 字段"},400); return
            pipeline_id = start_pipeline(goal)
            self.send_json({"success":True,"pipeline_id":pipeline_id,
                            "stream_url":f"/api/pipeline/{pipeline_id}/stream",
                            "status_url":f"/api/pipeline/{pipeline_id}/status"}); return

        self.send_json({"success":False,"error":"未找到"},404)

    # ── SSE Stream ────────────────────────────────────────────────────────────

    def _tl_sse_stream(self, session_id: str) -> None:
        """team-lead 分工过程的 SSE 实时推送"""
        self.send_response(200)
        self.send_header("Content-Type",              "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control",             "no-cache")
        self.send_header("X-Accel-Buffering",         "no")
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.end_headers()

        q = _tl_subscribe(session_id)
        try:
            while True:
                try:
                    event = q.get(timeout=25)
                    self._sse_write(event)
                    if event.get("type") == "done":
                        break
                except queue.Empty:
                    self._sse_write({"type": "heartbeat", "timestamp": time.time()})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _tl_unsubscribe(session_id, q)

    def _sse_stream(self, pipeline_id: str) -> None:
        """Server-Sent Events：实时推送流水线进度"""
        self.send_response(200)
        self.send_header("Content-Type",              "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control",             "no-cache")
        self.send_header("X-Accel-Buffering",         "no")
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.end_headers()

        q = _subscribe(pipeline_id)
        try:
            # 先推送已完成步骤（历史回放）
            rows = db_exec(
                "SELECT * FROM pipeline_steps WHERE pipeline_id=? AND status='done' ORDER BY step_index",
                (pipeline_id,))
            pl = db_exec("SELECT * FROM pipelines WHERE id=?",(pipeline_id,))
            if pl:
                self._sse_write({"type":"meta","pipeline_id":pipeline_id,
                                 "goal":pl[0]["goal"],"status":pl[0]["status"]})
            for row in rows:
                aid = row["agent_id"]
                am  = AGENT_META.get(aid,{"name":aid,"icon":"🤖"})
                self._sse_write({"type":"step_done","step_index":row["step_index"],
                                 "agent_id":aid,"agent_name":am["name"],"agent_icon":am["icon"],
                                 "output":row["output"],"success":True})

            # 若流水线已完成直接推 done 然后关闭
            if pl and pl[0]["status"] == "done":
                last = db_exec("SELECT output FROM pipeline_steps WHERE pipeline_id=? ORDER BY step_index DESC LIMIT 1",(pipeline_id,))
                self._sse_write({"type":"done","pipeline_id":pipeline_id,
                                 "final_output": last[0]["output"] if last else ""})
                return

            # 实时等待新事件
            while True:
                try:
                    event = q.get(timeout=25)
                    self._sse_write(event)
                    if event.get("type") == "done":
                        break
                except queue.Empty:
                    self._sse_write({"type":"heartbeat","timestamp":time.time()})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _unsubscribe(pipeline_id, q)

    def _sse_write(self, data: dict) -> None:
        line = f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0: return {}
        try:    return json.loads(self.rfile.read(length))
        except: return {}

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",              "application/json; charset=utf-8")
        self.send_header("Content-Length",            str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.send_header("Vary","Origin")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, filename: str, content_type: str) -> None:
        fp = os.path.join(STATIC_DIR, filename)
        if os.path.exists(fp):
            body = open(fp,"rb").read()
            self.send_response(200)
            self.send_header("Content-Type",   content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  self._cors_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary","Origin")
        self.end_headers()

    def log_message(self, format, *args): pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port   = int(os.environ.get("DASHBOARD_PORT", 8765))
    server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)
    print("╔══════════════════════════════════════════════╗")
    print("║     Hermes Multi-Agent Dashboard             ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Dashboard  : http://localhost:{port}          ║")
    print(f"║  Pipeline   : POST /api/pipeline/start       ║")
    print(f"║  SSE Stream : GET  /api/pipeline/<id>/stream ║")
    print(f"║  Agents     : {len(AGENTS)} configured                ║")
    print("╚══════════════════════════════════════════════╝")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
