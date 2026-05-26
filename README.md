# Hermes Kanban Dashboard

多 Agent 协作看板与聊天管理界面。

## 架构

```
┌──────────────────────────────────────────────┐
│             Kanban Dashboard                  │
│       (server.py + index.html)               │
├──────────────────────────────────────────────┤
│  API:  /api/chat/<agent>                     │
│        /api/chat/<agent>/forward             │
│        /api/chat/<agent>/history             │
│        /api/chat/<agent>/config              │
│        /api/agents                           │
│        /api/model-presets                    │
├──────────────────────────────────────────────┤
│  Agent Profiles: ~/.hermes/profiles/<agent>/ │
│    ├── config.yaml   (model, provider, url)  │
│    ├── .env          (API keys)              │
│    ├── SOUL.md       (role definition)       │
│    └── skills/       (task-specific skills)  │
└──────────────────────────────────────────────┘
```

## AGENT 列表

| ID | 角色 | 职责 |
|----|------|------|
| team-lead | 团队负责人 | 目标下达与验收 |
| project-director | 项目总监 | 全流程项目管理 |
| product-manager | 产品经理 | 需求分析与产品设计 |
| architect | 架构师 | 技术方案设计 |
| designer | 设计师 | UI/UX 设计 |
| frontend-engineer | 前端工程师 | 前端开发 |
| backend-engineer | 后端工程师 | 后端开发 |
| qa-tester | 测试工程师 | 测试与质量保障 |
| devops | 运维 | CI/CD 与部署 |
| researcher | 研究员 | 技术调研（工作节点） |
| backend-worker | 后端工作节点 | 后端任务执行 |
| frontend-worker | 前端工作节点 | 前端任务执行 |

## 启动

```bash
cd hermes-kanban-dashboard
python3 server.py
# 默认运行在 http://localhost:8000
```

## API 使用示例

```bash
# 向 Agent 发送消息
curl -X POST http://localhost:8000/api/chat/qa-tester \
  -H "Content-Type: application/json" \
  -d '{"message":"运行测试用例"}'

# 获取 Agent 配置
curl http://localhost:8000/api/chat/qa-tester/config

# 更新 Agent 配置（更换模型/API Key）
curl -X POST http://localhost:8000/api/chat/qa-tester/config \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.1","provider":"glm","base_url":"https://open.bigmodel.cn/api/paas/v4","api_key":"your_key"}'
```

## 配置

每个 Agent 的配置位于 `~/.hermes/profiles/<agent_id>/` 目录：
- `config.yaml` — 模型、Provider、Base URL 等
- `.env` — API Key 等环境变量

支持的 Provider 映射见 `server.py` 中 `PROVIDER_ENV_MAP`。

