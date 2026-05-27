#!/usr/bin/env python3
"""
Hermes Kanban Dashboard + Multi-Agent Chat API
每个 Agent 直接调用 LLM API，无外部依赖。
"""

import sqlite3, json, os, re, time, threading, queue, uuid, subprocess, yaml
import urllib.request as _urllib_req
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

DB_PATH      = os.path.expanduser("~/.hermes/kanban.db")
STATIC_DIR   = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(os.path.expanduser("~/.hermes"), "profiles")
ALLOWED_ORIGINS = {"http://localhost", "http://127.0.0.1"}

# ─── Model Presets ─────────────────────────────────────────────────────────────
MODEL_PRESETS = {
    "deepseek":   {"provider":"deepseek",  "base_url":"https://api.deepseek.com",
                   "label":"DeepSeek","icon":"🧠",
                   "models":[{"id":"deepseek-chat","name":"DeepSeek V3","context":"64K"},
                              {"id":"deepseek-reasoner","name":"DeepSeek R1","context":"64K"}]},
    "openai":     {"provider":"openai",    "base_url":"https://api.openai.com/v1",
                   "label":"OpenAI","icon":"🤖",
                   "models":[{"id":"gpt-4o","name":"GPT-4o","context":"128K"},
                              {"id":"gpt-4o-mini","name":"GPT-4o Mini","context":"128K"},
                              {"id":"gpt-4.1","name":"GPT-4.1","context":"1M"}]},
    "anthropic":  {"provider":"anthropic", "base_url":"https://api.anthropic.com",
                   "label":"Anthropic","icon":"🔮",
                   "models":[{"id":"claude-sonnet-4-20250514","name":"Claude Sonnet 4","context":"200K"},
                              {"id":"claude-haiku-3-5","name":"Claude Haiku 3.5","context":"200K"}]},
    "openrouter": {"provider":"openrouter","base_url":"https://openrouter.ai/api/v1",
                   "label":"OpenRouter","icon":"🌐",
                   "models":[{"id":"anthropic/claude-sonnet-4","name":"Claude Sonnet 4","context":"200K"},
                              {"id":"deepseek/deepseek-chat","name":"DeepSeek V3","context":"128K"},
                              {"id":"google/gemini-2.0-flash-001","name":"Gemini 2.0 Flash","context":"1M"}]},
    "google":     {"provider":"google",    "base_url":"https://generativelanguage.googleapis.com/v1beta",
                   "label":"Google Gemini","icon":"🔵",
                   "models":[{"id":"gemini-2.5-flash-preview-04-17","name":"Gemini 2.5 Flash","context":"1M"},
                              {"id":"gemini-2.5-pro-preview-03-25","name":"Gemini 2.5 Pro","context":"1M"}]},
    "zhipu":      {"provider":"zhipu",     "base_url":"https://open.bigmodel.cn/api/paas/v4",
                   "label":"智谱 GLM","icon":"🟤",
                   "models":[{"id":"glm-4-flash","name":"GLM-4-Flash（免费）","context":"128K"},
                              {"id":"glm-4","name":"GLM-4","context":"128K"}]},
    "moonshot":   {"provider":"moonshot",  "base_url":"https://api.moonshot.cn/v1",
                   "label":"月之暗面 Kimi","icon":"🌙",
                   "models":[{"id":"moonshot-v1-128k","name":"Moonshot v1 128K","context":"128K"}]},
    "alibaba":    {"provider":"alibaba",   "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
                   "label":"阿里通义千问","icon":"☁️",
                   "models":[{"id":"qwen-plus-2025-04-25","name":"Qwen Plus","context":"131K"},
                              {"id":"qwen-coder-plus","name":"Qwen Coder Plus","context":"128K"}]},
    "groq":       {"provider":"custom",    "base_url":"https://api.groq.com/openai/v1",
                   "label":"Groq（超快）","icon":"⚡",
                   "models":[{"id":"llama-3.3-70b-versatile","name":"Llama 3.3 70B","context":"128K"}]},
    "local":      {"provider":"custom",    "base_url":"http://localhost:11434/v1",
                   "label":"本地 Ollama","icon":"💻",
                   "models":[{"id":"qwen2.5:7b","name":"Qwen 2.5 7B","context":"128K"}]},
    "custom":     {"provider":"custom",    "base_url":"",
                   "label":"自定义端点","icon":"🔧",
                   "models":[{"id":"custom-model","name":"自定义模型","context":"—"}]},
}

# ─── Agent Definitions ─────────────────────────────────────────────────────────
AGENTS = [
    {"id":"team-lead",        "name":"负责人",    "icon":"👑","color":"#fbbf24"},
    {"id":"project-director", "name":"项目总监",  "icon":"🎯","color":"#f97316"},
    {"id":"product-manager",  "name":"产品经理",  "icon":"📋","color":"#6366f1"},
    {"id":"designer",         "name":"设计师",    "icon":"🎨","color":"#22d3ee"},
    {"id":"architect",        "name":"架构师",    "icon":"🏗️","color":"#34d399"},
    {"id":"frontend-engineer","name":"前端工程师","icon":"⚛️","color":"#14b8a6"},
    {"id":"backend-engineer", "name":"后端工程师","icon":"⚙️","color":"#a78bfa"},
    {"id":"qa-tester",        "name":"测试工程师","icon":"🧪","color":"#eab308"},
    {"id":"devops",           "name":"运维",      "icon":"🐳","color":"#ec4899"},
]
AGENT_IDS  = {a["id"] for a in AGENTS}
AGENT_META = {a["id"]: a for a in AGENTS}

# ─── Provider ENV Map ──────────────────────────────────────────────────────────
PROVIDER_ENV_MAP = {
    "deepseek":"DEEPSEEK_API_KEY","openai":"OPENAI_API_KEY",
    "anthropic":"ANTHROPIC_API_KEY","openrouter":"OPENROUTER_API_KEY",
    "google":"GOOGLE_API_KEY","zhipu":"GLM_API_KEY","glm":"GLM_API_KEY",
    "moonshot":"KIMI_API_KEY","alibaba":"DASHSCOPE_API_KEY","xai":"XAI_API_KEY",
}

# ─── Activity Tracking ─────────────────────────────────────────────────────────
_active_sessions: dict = {}

def track_activity(agent_id: str):
    _active_sessions[agent_id] = {
        "last_active": time.time(),
        "message_count": _active_sessions.get(agent_id,{}).get("message_count",0)+1,
    }

def get_agent_activity() -> dict:
    now = time.time(); result = {}
    for a in AGENTS:
        aid = a["id"]; s = _active_sessions.get(aid)
        if s:
            age = now - s["last_active"]
            result[aid] = {"active":age<300,"last_active_seconds":int(age),"message_count":s["message_count"]}
        else:
            result[aid] = {"active":False,"last_active_seconds":None,"message_count":0}
    return result

# ─── Agent Config ──────────────────────────────────────────────────────────────
def get_agent_profile_dir(agent_id: str) -> str:
    return os.path.join(PROFILES_DIR, agent_id)

def _parse_yaml_config(config_path: str) -> dict:
    defaults = {"model_id":"deepseek-chat","provider":"deepseek","base_url":"https://api.deepseek.com"}
    if not os.path.isfile(config_path): return defaults
    raw = open(config_path).read()
    try:
        data = yaml.safe_load(raw) or {}
        mb = data.get("model",{}) if isinstance(data,dict) else {}
        if isinstance(mb,dict) and any(k in mb for k in ("default","provider","base_url")):
            defaults["model_id"] = mb.get("default", defaults["model_id"])
            defaults["provider"] = mb.get("provider",defaults["provider"])
            defaults["base_url"] = mb.get("base_url", defaults["base_url"])
            return defaults
    except yaml.YAMLError:
        pass
    for line in raw.splitlines():
        s = line.strip()
        for key,field in [("model.default:","model_id"),("model.provider:","provider"),("model.base_url:","base_url")]:
            if s.startswith(key):
                val = s[len(key):].strip().strip("\"'")
                if val: defaults[field] = val
    return defaults

def _read_api_key_from_env_file(env_path: str, provider: str) -> str:
    """从 .env 文件读取 API key 原文"""
    if not os.path.isfile(env_path): return ""
    expected = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
    lines = open(env_path).readlines()
    # 优先匹配 provider 对应 key
    for line in lines:
        s = line.strip()
        if s.startswith(expected+"="):
            v = s.split("=",1)[1].strip().strip("\"'")
            if v: return v
    # 回退任意 API_KEY 行
    for line in lines:
        s = line.strip()
        if "API_KEY" in s and "=" in s and not s.startswith("#"):
            v = s.split("=",1)[1].strip().strip("\"'")
            if v: return v
    return ""

def get_agent_config(agent_id: str) -> dict:
    profile_dir = get_agent_profile_dir(agent_id)
    cfg = _parse_yaml_config(os.path.join(profile_dir,"config.yaml"))
    env_path = os.path.join(profile_dir,".env")
    key = _read_api_key_from_env_file(env_path, cfg["provider"])
    if not key:
        key = os.environ.get(PROVIDER_ENV_MAP.get(cfg["provider"],""),"")
    return {**cfg, "api_key_configured": bool(key)}

def save_agent_config(agent_id:str, model_id:str, provider:str, base_url:str, api_key:str) -> dict:
    profile_dir = get_agent_profile_dir(agent_id)
    config_path = os.path.join(profile_dir,"config.yaml")
    env_path    = os.path.join(profile_dir,".env")
    os.makedirs(profile_dir, exist_ok=True)
    lines = open(config_path).readlines() if os.path.isfile(config_path) else []
    new_lines, found = [], {"model":False,"provider":False,"base_url":False}
    for line in lines:
        s = line.strip()
        if   s.startswith("model.default:"):  new_lines.append(f'model.default: "{model_id}"\n');  found["model"]=True
        elif s.startswith("model.provider:"): new_lines.append(f'model.provider: "{provider}"\n'); found["provider"]=True
        elif s.startswith("model.base_url:"): new_lines.append(f'model.base_url: "{base_url}"\n'); found["base_url"]=True
        else: new_lines.append(line)
    if not found["model"]:    new_lines.append(f'model.default: "{model_id}"\n')
    if not found["provider"]: new_lines.append(f'model.provider: "{provider}"\n')
    if not found["base_url"]: new_lines.append(f'model.base_url: "{base_url}"\n')
    open(config_path,"w").writelines(new_lines)
    env_var  = PROVIDER_ENV_MAP.get(provider,"CUSTOM_API_KEY")
    existing = open(env_path).readlines() if os.path.isfile(env_path) else []
    new_env, replaced = [], False
    for line in existing:
        if line.strip().startswith(env_var+"="):
            if api_key: new_env.append(f"{env_var}={api_key}\n")
            replaced = True
        else: new_env.append(line)
    if not replaced and api_key: new_env.append(f"{env_var}={api_key}\n")
    open(env_path,"w").writelines(new_env)
    return {"success":True,"message":f"已保存 {agent_id} 的配置"}

# ─── SQLite ────────────────────────────────────────────────────────────────────
_db_lock = threading.Lock()

def _ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_agent ON chat_messages(agent_id);
        CREATE TABLE IF NOT EXISTS pipelines (
            id TEXT PRIMARY KEY, goal TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            created_at REAL NOT NULL, finished_at REAL
        );
        CREATE TABLE IF NOT EXISTS pipeline_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id TEXT NOT NULL, step_index INTEGER NOT NULL,
            agent_id TEXT NOT NULL, role_prompt TEXT NOT NULL,
            input TEXT NOT NULL, output TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            started_at REAL, finished_at REAL,
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

def db_exec(sql: str, params=()) -> list:
    with _db_lock:
        conn = get_db()
        try:
            cur = conn.execute(sql, params); conn.commit()
            return [dict(r) for r in (cur.fetchall() if cur.description else [])]
        finally:
            conn.close()

def load_chat_history(agent_id: str, limit: int = 20) -> list:
    rows = db_exec(
        "SELECT role,content,created_at FROM chat_messages "
        "WHERE agent_id=? ORDER BY created_at DESC LIMIT ?", (agent_id, limit))
    return list(reversed(rows))

def append_chat_message(agent_id:str, role:str, content:str):
    db_exec("INSERT INTO chat_messages(agent_id,role,content,created_at) VALUES(?,?,?,?)",
            (agent_id, role, content, time.time()))

# ─── Agent System Prompts（每个 Agent 只做本职工作）──────────────────────────
AGENT_SYSTEM_PROMPTS = {

    "team-lead": """你是团队负责人（Team Lead）。

【职责】只做协调和决策：
- 接收用户消息，判断是工作任务还是日常对话
- 工作任务：拆解子任务分配给各角色，明确验收标准，汇总交付物
- 日常对话：友好简短地回应，不超过3句话
- 绝不写代码、做设计、做测试，只做协调

【工作任务时输出格式】
1. 任务理解（一句话）
2. 各角色子任务分配（角色 → 具体任务）
3. 验收标准""",

    "project-director": """你是项目总监（Project Director）。

【职责】只做项目管理：
- 制定项目计划：WBS分解、里程碑、甘特图
- 风险管理：识别风险点，制定应对措施
- 资源协调：人员分工、工作量估算
- 最终验收：核对交付物是否达标

【输出要求】
- 项目计划有明确阶段、负责人、时间节点
- 风险有概率/影响评估和应对方案
- 验收标准可量化可检验
- 不写代码，不做设计，不做具体实现""",

    "product-manager": """你是产品经理（Product Manager）。

【职责】只做需求和产品设计：
- 用户故事：格式"作为[角色]，我想要[功能]，以便[价值]"
- 功能清单：优先级标注 P0/P1/P2，每条有验收标准（AC）
- PRD文档：功能描述、边界条件、异常情况
- 原型描述：界面布局、交互流程（文字描述）
- 数据结构：实体定义、字段和状态

【不做的事】不写代码，不做视觉设计（给设计师）""",

    "designer": """你是UI/UX设计师（Designer）。

【职责】只做视觉和交互设计：
- 配色：主色/辅色/背景色/文字色，必须给出十六进制色值（如 #1a1a2e）
- 字体：字体族、各级字号（px）、行高、字重
- 间距：组件内外边距具体数值（如 padding: 16px 24px）
- 布局：栅格、断点、空间比例
- 组件：按钮/卡片/表单/导航的视觉规范
- 动效：duration 和 easing（如 transition: all 300ms ease-in-out）

【输出要求】前端工程师拿到你的输出能直接写CSS，不需要再猜
【不做的事】不写代码，不做产品需求""",

    "architect": """你是系统架构师（Architect）。

【职责】只做技术架构设计：
- 技术选型：选择什么语言/框架/数据库，说明理由和对比
- 模块划分：系统有哪些模块，各自职责和边界
- 数据库设计：表名、字段名、字段类型、索引、关联关系
- API规范：method、path、请求体、响应体结构
- 关键算法：核心业务逻辑的实现思路
- 非功能：性能、安全、可扩展性方案

【输出要求】具体到工程师可以直接按此开发
【不做的事】不写完整业务代码，可写关键代码片段举例""",

    "frontend-engineer": """你是前端工程师（Frontend Engineer）。

【职责】只做前端开发：
- HTML/CSS/JavaScript 实现
- 单文件任务：所有CSS和JS内联到一个HTML文件
- 多文件任务：给出完整文件结构和每个文件内容
- 实现所有交互逻辑、状态管理、API对接
- 做好错误处理和边界情况

【输出要求】
- 代码必须完整可运行，绝不用"..."省略
- 变量和函数命名清晰，有必要的注释
- 指出需要哪些后端API（如果有）

【不做的事】不写后端代码，不做数据库设计""",

    "backend-engineer": """你是后端工程师（Backend Engineer）。

【职责】只做后端开发：
- API接口实现（RESTful）
- 业务逻辑、数据库操作
- 认证授权、数据验证
- 错误处理、日志记录

【输出要求】
- 代码完整可运行
- 包含 requirements.txt 或 package.json
- 包含环境变量列表（.env.example）
- 包含启动命令
- HTTP状态码使用正确

【不做的事】不写前端代码，不做基础设施配置（给运维）""",

    "qa-tester": """你是测试工程师（QA Tester）。

【职责】只做测试和质量保障：
- 设计测试用例：前置条件、操作步骤、预期结果
- 代码审查：找逻辑错误、边界条件缺失、异常未处理
- Bug报告：标题/严重级别/重现步骤/实际结果/预期结果
- 最终给出【通过】或【需整改】结论，并列出具体问题

【输出要求】
- 测试用例覆盖正常流程、边界值、异常情况
- Bug描述具体可重现，不能泛泛而谈

【不做的事】不修改被测代码，不做功能开发""",

    "devops": """你是运维工程师（DevOps Engineer）。

【职责】只做基础设施和部署：
- Dockerfile：多阶段构建，镜像优化
- docker-compose.yml：服务编排、网络、数据卷
- CI/CD：GitHub Actions/GitLab CI 流水线配置
- Nginx：反向代理、负载均衡、HTTPS、安全头
- 监控：Prometheus告警规则、Grafana Dashboard
- 部署脚本：具体可执行的 shell 命令

【输出要求】
- 配置文件完整可用，不留TODO占位符
- 提供具体命令，不说"配置XXX"这种废话

【不做的事】不写业务代码，不做功能开发""",
}

# ─── 意图识别（纯本地，不消耗 API）──────────────────────────────────────────
# 关键词全部小写，因为 msg 会被 lower()
_WORK_KW = {
    "开发","实现","做一个","做个","帮我做","帮我写","帮我设计","帮我搭",
    "功能","需求","项目","系统","代码","接口","api","数据库","部署",
    "测试","bug","报错","错误","崩溃","优化","重构","设计","架构",
    "方案","文档","原型","流程","上线","发布","自动化","爬虫","脚本",
    "网站","网页","页面","前端","后端","服务","服务器","容器",
    "游戏","应用","app","小程序","h5","安全","认证","登录",
    "用户管理","权限","支付","通知","推送","监控","日志","备份",
    "需要","要求","实现一个","写一个","建一个","搭一个","完成",
    "交付","sprint","迭代","版本","计划","排期","里程碑",
    "检查","排查","分析","调查","评估","审查","审核","复盘",
    "问题","故障","异常","性能","瓶颈","缺陷","改进",
    "进展","状态","情况","汇报","报告","总结","梳理","整理",
    "写","做","搞","弄","建","搭","改","修","查","看",
}
_CHAT_KW = {
    "你好","hi","hello","哈喽","早","午","晚","吃了吗",
    "你是谁","介绍一下","谢谢","感谢","厉害","牛","棒","nice",
    "哈哈","哈","呵呵","嗯","哦","啊","天气","心情","累了","休息",
}

def classify_intent(message: str) -> str:
    """返回 'work' 或 'chat'"""
    msg = message.strip().lower()
    if not msg: return "chat"
    # 极短消息（≤4字）且无工作词 → 聊天
    if len(message.strip()) <= 4 and not any(k in msg for k in _WORK_KW):
        return "chat"
    if any(k in msg for k in _WORK_KW): return "work"
    if any(k in msg for k in _CHAT_KW): return "chat"
    # 较长默认工作
    return "work" if len(message.strip()) > 15 else "chat"

# ─── LLM API 调用 ──────────────────────────────────────────────────────────────
def _get_key_and_cfg(agent_id: str):
    """返回 (api_key, provider, base_url, model_id)"""
    cfg      = get_agent_config(agent_id)
    provider = cfg["provider"]
    base_url = cfg["base_url"]
    model_id = cfg["model_id"]
    env_path = os.path.join(get_agent_profile_dir(agent_id), ".env")
    key = _read_api_key_from_env_file(env_path, provider)
    if not key: key = os.environ.get(PROVIDER_ENV_MAP.get(provider,""),"")
    return key, provider, base_url, model_id

def _call_openai_compat(endpoint:str, api_key:str, model_id:str,
                        messages:list, temperature:float=0.7) -> tuple:
    payload = json.dumps({"model":model_id,"messages":messages,
                          "max_tokens":4096,"temperature":temperature}).encode()
    req = _urllib_req.Request(endpoint, data=payload,
          headers={"Content-Type":"application/json",
                   "Authorization":f"Bearer {api_key}"}, method="POST")
    try:
        with _urllib_req.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return True, data["choices"][0]["message"]["content"].strip()
    except _urllib_req.HTTPError as e:
        body = e.read().decode(errors="replace")
        return False, f"❌ API {e.code}: {body[:300]}"
    except Exception as e:
        return False, f"❌ {e}"

def _call_anthropic(api_key:str, model_id:str, messages:list) -> tuple:
    system = ""; filtered = []
    for m in messages:
        if m["role"]=="system": system = m["content"]
        else: filtered.append(m)
    payload = json.dumps({"model":model_id,"max_tokens":4096,
                          "system":system,"messages":filtered}).encode()
    req = _urllib_req.Request("https://api.anthropic.com/v1/messages", data=payload,
          headers={"Content-Type":"application/json","x-api-key":api_key,
                   "anthropic-version":"2023-06-01"}, method="POST")
    try:
        with _urllib_req.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return True, data["content"][0]["text"].strip()
    except _urllib_req.HTTPError as e:
        return False, f"❌ Anthropic {e.code}: {e.read().decode(errors='replace')[:200]}"
    except Exception as e:
        return False, f"❌ {e}"

def call_llm(agent_id:str, user_prompt:str,
             history:list=None, force_work:bool=False) -> tuple:
    """
    调用该 Agent 配置的 LLM。
    history: [{"role":"user"|"assistant","content":str}, ...]
    返回 (success, response_text)
    """
    key, provider, base_url, model_id = _get_key_and_cfg(agent_id)
    if not key:
        name = AGENT_META.get(agent_id,{}).get("name",agent_id)
        return False, f"⚠️ {name} 未配置 API Key，请在 Agent 设置中配置。"

    system = AGENT_SYSTEM_PROMPTS.get(agent_id, "你是一个专业的AI助手。")

    # team-lead：根据意图追加提示
    if agent_id == "team-lead" and not force_work:
        intent = classify_intent(user_prompt)
        if intent == "chat":
            system += "\n\n当前消息是日常对话，请友好简短地回应（不超过3句），不要启动工作流程。"
        else:
            system += "\n\n当前消息是工作任务，请按格式：①任务理解 ②各角色子任务分配 ③验收标准。"

    # 构建 messages：system + 历史 + 当前
    messages = [{"role":"system","content":system}]
    if history:
        for h in history[-10:]:   # 最多带10轮历史
            if h["role"] in ("user","assistant"):
                messages.append({"role":h["role"],"content":h["content"][:800]})
    messages.append({"role":"user","content":user_prompt})

    if provider == "anthropic":
        return _call_anthropic(key, model_id, messages)

    endpoint = base_url.rstrip("/") + "/chat/completions"
    return _call_openai_compat(endpoint, key, model_id, messages)

# ─── Agent 对话接口 ────────────────────────────────────────────────────────────
def chat_with_agent(agent_id:str, user_message:str) -> dict:
    """普通对话：带历史，回复持久化"""
    history = load_chat_history(agent_id, limit=20)
    success, response = call_llm(agent_id, user_message, history=history)
    append_chat_message(agent_id, "user",      user_message)
    append_chat_message(agent_id, "assistant", response)
    track_activity(agent_id)
    return {"success":success,"response":response,"agent":agent_id,
            "agent_name":AGENT_META.get(agent_id,{}).get("name",agent_id)}

def forward_to_agent(from_agent:str, to_agent:str, message:str) -> dict:
    if to_agent not in AGENT_IDS:
        return {"success":False,"response":f"未知目标Agent: {to_agent}","agent":to_agent}
    from_name = AGENT_META.get(from_agent,{}).get("name",from_agent)
    to_name   = AGENT_META.get(to_agent,{}).get("name",to_agent)
    prompt = (f"【来自 {from_name} 的协作请求】\n\n{message}\n\n"
              f"请在你的职责范围内处理以上内容，给出专业输出。")
    track_activity(from_agent)
    result = chat_with_agent(to_agent, prompt)
    track_activity(to_agent)
    return result

# ─── 分工流水线（team-lead 自动协调）──────────────────────────────────────────
DELEGATION_PROMPTS = {
    "team-lead-kickoff": (
        "【用户目标】\n{goal}\n\n"
        "你是团队负责人，请拆解该目标，写一份项目启动令（含范围、验收标准），"
        "准备交给项目总监执行。直接输出启动令，不要废话。"
    ),
    "project-director-plan": (
        "【项目启动令】\n{prev}\n\n"
        "请制定详细项目计划：阶段划分、各角色职责、时间节点、风险及应对。"
    ),
    "product-manager": (
        "【项目背景】\n{prev}\n\n"
        "请输出完整PRD：用户故事（带AC）、功能清单（P0/P1/P2）、"
        "界面原型描述、数据结构定义。"
    ),
    "architect": (
        "【产品需求】\n{prev}\n\n"
        "请输出技术方案：技术选型（附理由）、模块划分、"
        "数据库表结构、API接口规范、关键实现要点。"
    ),
    "designer": (
        "【技术方案】\n{prev}\n\n"
        "请输出UI设计规范：配色（十六进制色值）、字体字号、间距规范、"
        "组件样式、动效参数。前端可直接按此写CSS。"
    ),
    "backend-engineer": (
        "【项目全部资料】\n{prev}\n\n"
        "请直接输出完整可运行的后端代码，包含依赖说明和启动命令。"
    ),
    "frontend-engineer": (
        "【项目全部资料】\n{prev}\n\n"
        "请直接输出完整可运行的前端代码（单HTML文件，内联CSS+JS）。"
        "代码必须完整，不得省略。"
    ),
    "qa-tester": (
        "【项目全部交付物】\n{prev}\n\n"
        "请列测试用例，审查代码中的Bug和问题，"
        "最终给出【通过】或【需整改】结论并列具体问题。"
    ),
    "project-director-review": (
        "【原始目标】\n{goal}\n\n"
        "【各角色交付物】\n{prev}\n\n"
        "作为项目总监，请逐项核对验收标准，"
        "给出明确结论：【验收通过】或【需返工】（需返工时列出具体问题）。"
    ),
    "team-lead-summary": (
        "【原始目标】\n{goal}\n\n"
        "【完整交付物】\n{prev}\n\n"
        "作为团队负责人，请写一份面向用户的交付报告：\n"
        "1. 完成情况摘要\n2. 完整交付物（代码完整列出）\n"
        "3. 使用说明\n4. 后续建议\n\n语言简明友好。"
    ),
}

def _build_steps(goal:str) -> list:
    needs_be = any(w in goal for w in ["后端","api","API","数据库","服务器","接口","登录","认证","存储"])
    needs_ui = any(w in goal for w in ["游戏","网页","界面","ui","UI","前端","可视化","动画","页面","网站"])
    steps = [
        {"agent":"team-lead",        "pk":"team-lead-kickoff",       "label":"拆解目标，起草启动令"},
        {"agent":"project-director", "pk":"project-director-plan",   "label":"制定项目计划"},
        {"agent":"product-manager",  "pk":"product-manager",         "label":"编写产品需求 PRD"},
        {"agent":"architect",        "pk":"architect",               "label":"设计技术架构"},
    ]
    if needs_ui:
        steps.append({"agent":"designer","pk":"designer","label":"UI/UX 设计规范"})
    if needs_be:
        steps.append({"agent":"backend-engineer","pk":"backend-engineer","label":"后端开发"})
    steps += [
        {"agent":"frontend-engineer","pk":"frontend-engineer",      "label":"前端开发"},
        {"agent":"qa-tester",        "pk":"qa-tester",              "label":"测试 & 质检"},
        {"agent":"project-director", "pk":"project-director-review","label":"项目总监验收"},
        {"agent":"team-lead",        "pk":"team-lead-summary",      "label":"汇总交付报告"},
    ]
    return steps

# SSE 订阅（team-lead 分工用）
_tl_subs: dict = {}
_tl_lock = threading.Lock()

def _tl_pub(sid:str, ev:dict):
    with _tl_lock: qs = list(_tl_subs.get(sid,[]))
    for q in qs:
        try: q.put_nowait(ev)
        except queue.Full: pass

def _tl_sub(sid:str) -> queue.Queue:
    q = queue.Queue(maxsize=300)
    with _tl_lock: _tl_subs.setdefault(sid,[]).append(q)
    return q

def _tl_unsub(sid:str, q:queue.Queue):
    with _tl_lock:
        subs = _tl_subs.get(sid,[])
        if q in subs: subs.remove(q)

def _run_delegation(session_id:str, goal:str):
    steps = _build_steps(goal)
    outputs = []   # [{"agent":id,"output":str}]
    retried = False

    _tl_pub(session_id,{"type":"start","goal":goal,"total":len(steps),"ts":time.time()})

    i = 0
    while i < len(steps):
        s = steps[i]
        aid, pk, label = s["agent"], s["pk"], s["label"]
        am = AGENT_META.get(aid,{"name":aid,"icon":"🤖","color":"#818cf8"})

        prev_ctx = "\n\n".join(
            f"【{AGENT_META.get(o['agent'],{}).get('name',o['agent'])} 输出】\n{o['output']}"
            for o in outputs) or "(无前序输出)"

        # team-lead-kickoff 不需要 prev
        if pk == "team-lead-kickoff":
            prompt = DELEGATION_PROMPTS[pk].format(goal=goal)
        else:
            prompt = DELEGATION_PROMPTS[pk].format(goal=goal, prev=prev_ctx)

        _tl_pub(session_id,{"type":"step_start","step":i,"total":len(steps),
                             "agent_id":aid,"agent_name":am["name"],
                             "agent_icon":am["icon"],"agent_color":am.get("color","#818cf8"),
                             "label":label,"ts":time.time()})

        # 流水线里强制 work，跳过意图检测
        success, output = call_llm(aid, prompt, force_work=True)
        append_chat_message(aid,"user",prompt)
        append_chat_message(aid,"assistant",output)
        track_activity(aid)
        outputs.append({"agent":aid,"output":output})

        _tl_pub(session_id,{"type":"step_done","step":i,"total":len(steps),
                             "agent_id":aid,"agent_name":am["name"],
                             "agent_icon":am["icon"],"agent_color":am.get("color","#818cf8"),
                             "label":label,"output":output,"success":success,"ts":time.time()})

        # 验收不通过，自动返工一次
        if pk=="project-director-review" and not retried:
            if any(kw in output for kw in ["需返工","需要返工","不通过","reject"]):
                retried = True
                eng = [x for x in steps if "engineer" in x["agent"]]
                if eng:
                    rw = dict(eng[-1]); rw["label"]="🔁 "+rw["label"]+"（返工）"
                    steps.insert(i+1, rw)
                    steps.insert(i+2, dict(steps[i]))
                    _tl_pub(session_id,{"type":"rework","reason":"验收未通过，自动返工","ts":time.time()})
        i += 1

    final = outputs[-1]["output"] if outputs else "（无输出）"
    append_chat_message("team-lead","user",goal)
    append_chat_message("team-lead","assistant",final)
    _tl_pub(session_id,{"type":"done","final":final,"ts":time.time()})

def start_delegation(goal:str) -> str:
    sid = str(uuid.uuid4())
    threading.Thread(target=_run_delegation,args=(sid,goal),daemon=True).start()
    return sid

# ─── Kanban（表不存在时优雅降级）─────────────────────────────────────────────
def _safe_query(sql:str, params=()) -> list:
    """执行查询，若表不存在返回空列表"""
    try: return db_exec(sql, params)
    except Exception: return []

def get_all_tasks() -> list:
    return _safe_query("SELECT * FROM tasks ORDER BY created_at DESC")

def get_stats() -> dict:
    stats = {"total":0,"todo":0,"ready":0,"running":0,"blocked":0,"done":0,"archived":0,"profiles":{}}
    for s in ["todo","ready","running","blocked","done","archived"]:
        rows = _safe_query("SELECT COUNT(*) as cnt FROM tasks WHERE status=?",(s,))
        stats[s] = rows[0]["cnt"] if rows else 0
    rows = _safe_query("SELECT COUNT(*) as cnt FROM tasks")
    stats["total"] = rows[0]["cnt"] if rows else 0
    return stats

def get_events(limit:int=50) -> list:
    return _safe_query("SELECT * FROM task_events ORDER BY created_at DESC LIMIT ?",(limit,))

# ─── HTTP Handler ──────────────────────────────────────────────────────────────
class DashboardHandler(SimpleHTTPRequestHandler):

    def _origin(self) -> str:
        o = self.headers.get("Origin","")
        base = o.rsplit(":",1)[0] if o.count(":")>=2 else o
        return o if base in ALLOWED_ORIGINS else "http://localhost"

    # GET ───────────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path

        # Kanban
        if   path=="/api/tasks":   self.json(get_all_tasks())
        elif path=="/api/stats":   self.json(get_stats())
        elif path=="/api/events":  self.json(get_events())
        elif path=="/api/health":  self.json({"status":"ok","db":os.path.exists(DB_PATH),"time":int(time.time())})
        # Agents
        elif path=="/api/agents":  self.json({"agents":AGENTS,"total":len(AGENTS)})
        elif path=="/api/model-presets": self.json(MODEL_PRESETS)
        elif path=="/api/agents/status":
            out=[]
            for a in AGENTS:
                cfg=get_agent_config(a["id"])
                out.append({**a,"configured":cfg["api_key_configured"],
                            "model":cfg["model_id"],"provider":cfg["provider"]})
            self.json({"agents":out})
        elif path=="/api/agents/activity": self.json(get_agent_activity())
        # Chat history
        elif re.match(r"^/api/chat/[\w-]+/history$",path):
            aid=path.split("/")[3]
            if aid not in AGENT_IDS: self.json({"error":"未知Agent"},404); return
            self.json({"id":aid,"messages":load_chat_history(aid,50)})
        # Agent config
        elif re.match(r"^/api/chat/[\w-]+/config$",path):
            aid=path.split("/")[3]
            if aid not in AGENT_IDS: self.json({"error":"未知Agent"},404); return
            self.json(get_agent_config(aid))
        # Team-lead SSE stream
        elif re.match(r"^/api/chat/team-lead/stream/[\w-]+$",path):
            self._tl_stream(path.split("/")[-1]); return
        # Static
        elif path in ("/","/index.html"): self.serve_file("index.html","text/html")
        else: super().do_GET()

    # POST ──────────────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path

        # Chat
        m = re.match(r"^/api/chat/([\w-]+)$",path)
        if m:
            aid = m.group(1)
            if aid not in AGENT_IDS: self.json({"success":False,"error":"未知Agent"},404); return
            body = self.body(); msg = (body.get("message") or "").strip()
            if not msg: self.json({"success":False,"error":"message不能为空"},400); return
            if aid=="team-lead":
                if classify_intent(msg)=="work":
                    sid = start_delegation(msg)
                    self.json({"success":True,"delegating":True,"session_id":sid,
                               "stream_url":f"/api/chat/team-lead/stream/{sid}"}); return
            self.json(chat_with_agent(aid, msg)); return

        # Forward
        m = re.match(r"^/api/chat/([\w-]+)/forward$",path)
        if m:
            fa=m.group(1)
            if fa not in AGENT_IDS: self.json({"success":False,"error":"未知Agent"},404); return
            body=self.body()
            if "to" not in body or "message" not in body:
                self.json({"success":False,"error":"需要 to 和 message"},400); return
            self.json(forward_to_agent(fa,body["to"],body["message"])); return

        # Config
        m = re.match(r"^/api/chat/([\w-]+)/config$",path)
        if m:
            aid=m.group(1)
            if aid not in AGENT_IDS: self.json({"success":False,"error":"未知Agent"},404); return
            body=self.body()
            self.json(save_agent_config(aid,
                body.get("model_id","deepseek-chat"),body.get("provider","deepseek"),
                body.get("base_url","https://api.deepseek.com"),body.get("api_key",""))); return

        self.json({"success":False,"error":"未找到"},404)

    # SSE ───────────────────────────────────────────────────────────────────────
    def _tl_stream(self, sid:str):
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream; charset=utf-8")
        self.send_header("Cache-Control","no-cache")
        self.send_header("X-Accel-Buffering","no")
        self.send_header("Access-Control-Allow-Origin",self._origin())
        self.end_headers()
        q = _tl_sub(sid)
        try:
            while True:
                try:
                    ev = q.get(timeout=25)
                    self._sse(ev)
                    if ev.get("type")=="done": break
                except queue.Empty:
                    self._sse({"type":"heartbeat","ts":time.time()})
        except (BrokenPipeError,ConnectionResetError): pass
        finally: _tl_unsub(sid,q)

    def _sse(self, data:dict):
        line = f"data: {json.dumps(data,ensure_ascii=False,default=str)}\n\n"
        self.wfile.write(line.encode()); self.wfile.flush()

    # Helpers ───────────────────────────────────────────────────────────────────
    def body(self) -> dict:
        n = int(self.headers.get("Content-Length",0))
        if not n: return {}
        try: return json.loads(self.rfile.read(n))
        except: return {}

    def json(self, data, status:int=200):
        body = json.dumps(data,ensure_ascii=False,default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Access-Control-Allow-Origin",self._origin())
        self.send_header("Vary","Origin")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, name:str, ct:str):
        fp = os.path.join(STATIC_DIR,name)
        if os.path.exists(fp):
            body=open(fp,"rb").read()
            self.send_response(200)
            self.send_header("Content-Type",ct)
            self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else: self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",self._origin())
        self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.send_header("Vary","Origin"); self.end_headers()

    def log_message(self,*a): pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__=="__main__":
    port = int(os.environ.get("DASHBOARD_PORT",8765))
    srv  = ThreadedHTTPServer(("0.0.0.0",port), DashboardHandler)
    print(f"╔══════════════════════════════════════╗")
    print(f"║  Hermes Multi-Agent Dashboard        ║")
    print(f"║  http://localhost:{port}               ║")
    print(f"║  {len(AGENTS)} Agents | 直接 LLM API 调用  ║")
    print(f"╚══════════════════════════════════════╝")
    try: srv.serve_forever()
    except KeyboardInterrupt: srv.server_close()
