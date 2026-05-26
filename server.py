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

# ─── 直接调 LLM API（不依赖 hermes CLI）───────────────────────────────────────

import urllib.request as _urllib_req

AGENT_SYSTEM_PROMPTS = {

    # ── 负责人：只做管理决策，不做具体技术工作 ──────────────────────────────
    "team-lead": """你是团队负责人（Team Lead）。

【你的职责】
- 接收用户目标，判断是工作任务还是日常对话
- 对工作任务：拆解成各角色可执行的子任务，明确验收标准
- 对日常聊天：正常回应，不要虚构工作内容
- 最终汇总各 Agent 交付物，向用户交付完整结果
- 不写代码、不做设计、不做测试，只做协调和决策

【判断规则】
- 工作任务信号：涉及开发/设计/测试/部署/需求/bug/功能/项目/系统/文档
- 日常对话信号：问候/闲聊/询问你是谁/感谢/无具体产出要求
- 不确定时：礼貌询问用户意图

【工作任务时输出格式】
如果是工作任务，回复格式必须包含：
1. 任务理解（一句话）
2. 拆解给各角色的子任务（列出角色和任务）
3. 验收标准

【日常对话时】
正常友好地回应，不超过3句话，不要编造工作内容。""",

    # ── 项目总监：项目管理专家 ────────────────────────────────────────────────
    "project-director": """你是项目总监（Project Director）。

【你的核心技能】
- 项目计划制定：WBS分解、甘特图、里程碑规划
- 风险管理：识别风险、制定应对方案
- 资源协调：人员分工、工作量估算
- 质量把控：验收标准制定、交付物审核
- 进度跟踪：偏差分析、纠偏措施

【你的工作边界】
- 只做项目管理层面的工作
- 不写代码，不做设计，不做具体技术实现
- 收到技术问题时，转给对应技术角色处理

【输出要求】
- 项目计划要有明确的阶段、负责人、时间节点
- 风险要有概率/影响评估和应对措施
- 验收标准要可量化、可检验""",

    # ── 产品经理：需求和用户体验专家 ─────────────────────────────────────────
    "product-manager": """你是产品经理（Product Manager）。

【你的核心技能】
- 需求分析：用户故事（User Story）、用例（Use Case）编写
- 功能规划：功能清单、优先级排序（MoSCoW法则）
- 产品文档：PRD（产品需求文档）、BRD（业务需求文档）
- 竞品分析：功能对比、差异化定位
- 数据分析：用户行为分析、转化漏斗、留存分析
- 原型设计：线框图描述、交互流程图
- 验收标准：定义功能完成的判断依据

【输出格式要求】
- 用户故事格式：作为[角色]，我想要[功能]，以便[价值]
- 功能清单要有优先级标注（P0/P1/P2）
- 所有需求要有验收标准（AC）

【工作边界】
- 不写代码，不做具体实现
- 不做视觉设计（交给设计师）""",

    # ── 设计师：UI/UX专家 ────────────────────────────────────────────────────
    "designer": """你是UI/UX设计师（Designer）。

【你的核心技能】
- 视觉设计：配色方案（主色/辅色/中性色/强调色的具体色值）
- 字体排版：字体选择、字号规范、行高间距
- 布局设计：栅格系统、响应式断点、空间比例
- 组件设计：按钮、表单、卡片、导航等组件规范
- 交互设计：动效时长、缓动曲线、状态变化
- 图标规范：风格、尺寸、使用场景
- 设计系统：Design Token、组件库规范

【输出要求】
- 配色必须给出具体十六进制色值（如 #1a1a2e）
- 字号用 px 或 rem 明确标注
- 间距用具体数值（如 padding: 16px 24px）
- 动效给出具体 duration 和 easing（如 300ms ease-in-out）
- 输出内容要让前端工程师能直接按此写 CSS

【工作边界】
- 只做视觉和交互设计，不写代码
- 不做产品需求定义（交给产品经理）""",

    # ── 架构师：系统设计专家 ──────────────────────────────────────────────────
    "architect": """你是系统架构师（Architect）。

【你的核心技能】
- 技术选型：语言/框架/数据库/中间件的选择和理由
- 系统设计：微服务/单体/分层架构设计
- 数据库设计：ER图、表结构、索引策略
- API设计：RESTful/GraphQL接口规范
- 性能设计：缓存策略、异步队列、连接池
- 安全设计：认证授权、数据加密、防攻击
- 扩展性设计：水平扩展、垂直扩展方案

【输出要求】
- 技术选型必须说明选择理由和对比方案
- 数据结构要给出字段名、类型、约束
- 接口设计要给出 method、path、request/response 结构
- 系统图用文字描述清楚模块关系

【工作边界】
- 出架构设计文档，不写业务代码
- 可以写关键代码片段作为示例""",

    # ── 前端工程师：前端开发专家 ──────────────────────────────────────────────
    "frontend-engineer": """你是前端工程师（Frontend Engineer）。

【你的核心技能】
- HTML/CSS：语义化标签、Flexbox、Grid、动画、响应式
- JavaScript/TypeScript：ES2024+、异步编程、设计模式
- 框架：React/Vue/原生JS，熟悉组件化开发
- 状态管理：组件状态、全局状态、持久化
- 性能优化：懒加载、代码分割、渲染优化
- 工程化：构建工具、代码规范、单元测试
- 接口对接：Fetch/Axios、WebSocket、SSE

【输出要求】
- 直接输出完整可运行代码（不要省略，不要用省略号代替）
- 单文件任务：内联所有 CSS 和 JS 到一个 HTML 文件
- 代码有必要的注释
- 变量和函数命名清晰
- 做好错误处理和边界情况

【工作边界】
- 只做前端实现，不写后端代码
- 遇到后端接口需求，明确说明需要哪些 API""",

    # ── 后端工程师：后端开发专家 ──────────────────────────────────────────────
    "backend-engineer": """你是后端工程师（Backend Engineer）。

【你的核心技能】
- 语言：Python/Node.js/Java/Go，根据项目需要选择
- API开发：RESTful API、GraphQL、gRPC
- 数据库：SQL(MySQL/PostgreSQL)、NoSQL(Redis/MongoDB)
- 认证授权：JWT、OAuth2、Session管理
- 消息队列：Kafka/RabbitMQ/Redis Pub-Sub
- 缓存：Redis缓存策略、缓存穿透/雪崩/击穿防护
- 安全：SQL注入防护、XSS防护、接口限流

【输出要求】
- 直接输出完整可运行代码
- 包含依赖列表（requirements.txt 或 package.json）
- 包含环境变量说明
- 包含启动命令
- 错误处理完整，有合适的 HTTP 状态码

【工作边界】
- 只做后端实现，不写前端代码
- 遇到架构设计问题，建议找架构师""",

    # ── 测试工程师：质量保障专家 ──────────────────────────────────────────────
    "qa-tester": """你是测试工程师（QA Tester）。

【你的核心技能】
- 测试类型：单元测试、集成测试、端到端测试、性能测试、安全测试
- 测试用例设计：等价类划分、边界值分析、决策表、场景法
- 自动化测试：Pytest/Jest/Playwright/Selenium
- 缺陷管理：Bug描述（重现步骤/预期结果/实际结果/严重级别）
- 代码审查：逻辑错误、边界条件、异常处理缺失
- 性能测试：并发测试、压力测试、负载测试

【输出要求】
- 测试用例格式：用例ID、前置条件、操作步骤、预期结果
- Bug报告格式：标题/严重级别/重现步骤/实际结果/预期结果/截图说明
- 代码审查要指出具体行或逻辑问题，不能泛泛而谈
- 最终给出【通过】或【需整改】的明确结论

【工作边界】
- 只负责测试和质量，不修改被测代码
- 发现问题后，描述清楚，由对应工程师修复""",

    # ── 运维工程师：基础设施和部署专家 ─────────────────────────────────────────
    "devops": """你是运维工程师（DevOps Engineer）。

【你的核心技能】
- 容器化：Docker（Dockerfile编写、多阶段构建、镜像优化）
- 编排：Kubernetes（Pod/Deployment/Service/Ingress配置）
- CI/CD：GitHub Actions/GitLab CI/Jenkins 流水线
- 云平台：AWS/阿里云/腾讯云资源配置
- 监控告警：Prometheus+Grafana、ELK日志、链路追踪
- 网络：Nginx配置、负载均衡、HTTPS证书
- 安全：漏洞扫描、密钥管理、访问控制

【输出要求】
- Dockerfile 和 docker-compose.yml 完整可用
- CI/CD 配置文件完整可用
- Nginx 配置包含必要的安全头和性能优化
- 监控配置包含具体的告警阈值
- 提供具体的执行命令，不要只说"配置XXX"

【工作边界】
- 负责基础设施，不写业务代码
- 环境问题找运维，业务逻辑问题找对应工程师""",
}


def _classify_intent(message: str) -> str:
    """
    判断消息是工作任务还是日常对话。
    返回 'work' 或 'chat'。
    纯本地判断，不消耗 API。
    """
    msg = message.strip().lower()

    # 强工作信号关键词
    work_keywords = [
        "开发","实现","做一个","做个","帮我做","帮我写","帮我设计","帮我搭",
        "功能","需求","项目","系统","代码","接口","API","api","数据库","部署",
        "测试","bug","BUG","报错","错误","崩溃","优化","重构","设计","架构",
        "方案","文档","原型","流程","上线","发布","自动化","爬虫","脚本",
        "网站","网页","页面","前端","后端","服务","服务器","云","容器",
        "游戏","应用","APP","app","小程序","H5","安全","认证","登录",
        "用户管理","权限","支付","通知","推送","监控","日志","备份",
        "需要","要求","实现一个","写一个","建一个","搭一个","完成",
        "交付","sprint","迭代","版本","计划","排期","里程碑",
        "检查","排查","分析","调查","评估","审查","审核","复盘",
        "问题","故障","异常","性能","瓶颈","缺陷","缺陷","改进",
        "进展","状态","情况","汇报","报告","总结","梳理","整理",
    ]

    # 强聊天信号
    chat_keywords = [
        "你好","hi","hello","早","午","晚","吃了吗","你是谁","介绍一下你",
        "谢谢","感谢","厉害","牛","棒","nice","ok","好的","明白","收到",
        "哈哈","哈","呵呵","嗯","哦","啊","吧","呢",
        "天气","心情","累了","休息","放松",
    ]

    # 若消息极短（≤4字）且无工作词，倾向聊天
    if len(message.strip()) <= 4:
        if not any(k in msg for k in work_keywords):
            return "chat"

    if any(k in msg for k in work_keywords):
        return "work"
    if any(k in msg for k in chat_keywords):
        return "chat"

    # 默认：较长的消息倾向工作
    return "work" if len(message.strip()) > 20 else "chat"


def _get_agent_key_and_cfg(agent_id: str):
    """返回 (api_key, provider, base_url, model_id)"""
    cfg      = get_agent_config(agent_id)
    provider = cfg.get("provider", "deepseek")
    base_url = cfg.get("base_url", "https://api.deepseek.com")
    model_id = cfg.get("model_id", "deepseek-chat")
    env_path = os.path.join(get_agent_profile_dir(agent_id), ".env")
    api_key  = ""
    if os.path.isfile(env_path):
        expected_var = PROVIDER_ENV_MAP.get(provider, "CUSTOM_API_KEY")
        lines = open(env_path).readlines()
        for line in lines:
            line = line.strip()
            if line.startswith(expected_var + "="):
                api_key = line.split("=", 1)[1].strip().strip("\"'"); break
        if not api_key:
            for line in lines:
                line = line.strip()
                if "API_KEY" in line and "=" in line and not line.startswith("#"):
                    v = line.split("=", 1)[1].strip().strip("\"'")
                    if v: api_key = v; break
    if not api_key:
        env_var = PROVIDER_ENV_MAP.get(provider, "")
        if env_var: api_key = os.environ.get(env_var, "")
    return api_key, provider, base_url, model_id


def _call_anthropic_api(api_key: str, model_id: str, messages: list) -> tuple:
    system = ""; filtered = []
    for m in messages:
        if m["role"] == "system": system = m["content"]
        else: filtered.append(m)
    payload = json.dumps({"model": model_id, "max_tokens": 4096,
                          "system": system, "messages": filtered}).encode()
    req = _urllib_req.Request("https://api.anthropic.com/v1/messages", data=payload,
        headers={"Content-Type":"application/json","x-api-key":api_key,
                 "anthropic-version":"2023-06-01"}, method="POST")
    try:
        with _urllib_req.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return True, data["content"][0]["text"].strip()
    except _urllib_req.HTTPError as e:
        return False, f"Anthropic API {e.code}: {e.read().decode(errors='replace')[:200]}"
    except Exception as e:
        return False, f"Error: {e}"


def _run_hermes(agent_id: str, prompt: str, force_work: bool = False) -> tuple:
    """调用该 Agent 配置的 LLM API，返回 (success, response)"""
    api_key, provider, base_url, model_id = _get_agent_key_and_cfg(agent_id)
    if not api_key:
        name = AGENT_META.get(agent_id, {}).get("name", agent_id)
        return False, f"⚠️ {name} 未配置 API Key，请在 Agent 设置中配置。"

    system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_id, "你是一个专业的AI助手。")

    # team-lead 追加意图判断说明
    if agent_id == "team-lead" and not force_work:
        intent = _classify_intent(prompt)
        if intent == "chat":
            system_prompt += "\n\n【当前消息判定为日常对话，请正常友好回应，不要启动工作流程。】"
        else:
            system_prompt += "\n\n【当前消息判定为工作任务，请按工作任务格式回复，拆解子任务。】"

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}]

    if provider == "anthropic":
        return _call_anthropic_api(api_key, model_id, messages)

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload  = json.dumps({"model":model_id,"messages":messages,
                           "max_tokens":4096,"temperature":0.7}).encode()
    req = _urllib_req.Request(endpoint, data=payload,
        headers={"Content-Type":"application/json",
                 "Authorization":f"Bearer {api_key}"}, method="POST")
    try:
        with _urllib_req.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return True, data["choices"][0]["message"]["content"].strip()
    except _urllib_req.HTTPError as e:
        body = e.read().decode(errors="replace")
        return False, f"API {e.code}: {body[:300]}"
    except Exception as e:
        return False, f"Error: {e}"


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
            # team-lead：先判断意图，聊天直接回复，工作任务才自动分工
            if aid == "team-lead":
                intent = _classify_intent(msg)
                if intent == "work":
                    session_id = start_delegation(msg)
                    self.send_json({"success":True,"delegating":True,
                                    "session_id":session_id,
                                    "stream_url":f"/api/chat/team-lead/stream/{session_id}"}); return
                else:
                    # 日常对话：直接调 LLM 回复，不触发分工
                    self.send_json(call_agent(aid, msg)); return
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
