## Context

pi-mono 是一个 TypeScript AI 代理框架（v0.55.1），包含 7 个模块。当前正在将其核心功能迁移到 Python，已完成约 45% 的工作：

- **已完成**：agent 框架核心（Agent、AgentLoop、事件系统）、AI 统一 API（types、stream、5 个 provider）、coding-agent 基础架构（会话管理、7 个内置工具、模型注册/解析、print mode）
- **未完成**：REST 服务端、.pi 扩展系统、settings.yaml 配置、web-ui REST 适配、测试体系
- **技术栈**：Python 3.11、uv、hatchling、pydantic、httpx、asyncio、pytest

原始 TS 项目中 web-ui 通过 RPC（JSON 行协议 stdin/stdout）与 coding-agent 子进程通信。Python 版本将改为 REST API + SSE，web-ui 直接通过 HTTP 调用 Python 后端。

## Goals / Non-Goals

**Goals:**

- 完成 Python 版本的 4 个核心模块：ai（仅 OpenAI 兼容）、agent、coding-agent、rest-server
- 提供 REST API 服务端，web-ui 通过 HTTP/SSE 与后端交互
- 实现 `.pi/` 扩展机制（Python 模块加载、提示词模板、settings.yaml）
- 在已有代码基础上继续迁移，保持现有架构和设计模式
- TDD 驱动，测试覆盖率 80%+

**Non-Goals:**

- 不迁移 mom（Slack 机器人）、pods（GPU 容器管理）、tui（终端 UI 库）
- 不支持 Anthropic、Google、Bedrock、Mistral 等非 OpenAI 兼容的 provider（已有的可保留但不作为范围）
- 不重写 web-ui 技术栈（保留 Lit Web Components）
- 不实现 OAuth 认证流程（使用 API key 直接认证）
- 不实现 TUI 交互式组件（保留 print mode + REST API 模式）
- 不实现会话分支/分叉（树状结构），仅支持线性会话

## Decisions

### D1: AI 层仅保留 OpenAI 兼容 Provider

**决策**：移除 Anthropic、Google 等独立 provider，仅保留 OpenAI Completions provider 作为统一接口。

**理由**：
- OpenAI Chat Completions API 已成为事实标准，大多数 LLM 服务商（包括 Anthropic 通过兼容层、本地模型如 Ollama/vLLM/LM Studio）都提供兼容接口
- 减少维护负担，5 个 provider 各 800-1100 行代码，维护成本高
- 用户可通过配置不同的 `base_url` 接入任意兼容服务

**替代方案**：
- 保留所有 provider → 维护成本过高，与缩小范围的目标冲突
- 使用 LiteLLM 作为统一层 → 引入重量级依赖，且控制力不足

**影响**：
- 保留 `openai_completions.py` 作为核心 provider，重命名为 `openai_compat.py`
- 保留 `LLMProvider` Protocol、`stream()`/`complete()` 公共 API 不变
- 移除 `anthropic.py`、`google.py`、`openai_responses.py` 及对应 SDK 依赖
- `ModelCompat` 机制保留，用于处理不同 OpenAI 兼容服务的差异

### D2: REST API 框架选择 FastAPI + SSE

**决策**：使用 FastAPI 作为 REST 框架，Server-Sent Events（SSE）用于流式事件推送。

**理由**：
- FastAPI 原生支持 async/await，与现有 asyncio 架构一致
- 自动生成 OpenAPI 文档，便于 web-ui 集成
- SSE 比 WebSocket 更简单，单向事件流足够满足需求（命令通过 REST POST，事件通过 SSE 推送）
- 原始 RPC 模式本质上也是「命令→响应 + 单向事件流」，REST + SSE 是最自然的 HTTP 映射

**替代方案**：
- WebSocket → 双向通信能力过剩，增加连接管理复杂度
- REST + Polling → 延迟高，体验差
- gRPC → 前端集成复杂，web-ui 需要额外的代理层

**API 设计**：

```
# 会话管理
POST   /api/sessions                    创建会话
GET    /api/sessions                    列出会话
GET    /api/sessions/{id}               获取会话详情
DELETE /api/sessions/{id}               删除会话

# 消息交互
POST   /api/sessions/{id}/messages      发送消息（prompt）
POST   /api/sessions/{id}/steer         插入 steering 消息
POST   /api/sessions/{id}/abort         中止当前处理
GET    /api/sessions/{id}/messages      获取消息历史

# 事件流
GET    /api/sessions/{id}/events        SSE 事件流（agent/message/tool 事件）

# 模型与配置
GET    /api/models                      列出可用模型
PUT    /api/sessions/{id}/model         切换模型
PUT    /api/sessions/{id}/thinking      设置思考级别

# 系统
GET    /api/settings                    获取全局设置
PUT    /api/settings                    更新全局设置
GET    /api/health                      健康检查
```

### D3: 会话管理架构 — 进程内 Agent 实例

**决策**：每个会话对应一个进程内的 `Agent` 实例，由 REST 服务端管理生命周期。

**理由**：
- 已有的 `Agent` 类和 `AgentLoop` 设计为进程内使用，无需子进程隔离
- 避免 RPC 序列化开销，直接调用 Python 对象
- 简化部署，单进程即可运行

**替代方案**：
- 每个会话一个子进程（原 TS 模式）→ 增加 IPC 复杂度，Python 场景下不必要
- 多进程 Worker Pool → 过度工程化，初期不需要

**实现**：
```
REST Server (FastAPI + uvicorn)
    ↓
SessionRegistry: dict[session_id, AgentSession]
    ↓
AgentSession → Agent → AgentLoop → LLM Provider
```

### D4: SSE 事件流设计

**决策**：每个会话维护一个 SSE 端点，客户端连接后接收所有 Agent 事件。

**事件格式**（遵循 SSE 标准）：
```
event: message_start
data: {"type": "message_start", "message": {...}}

event: message_update
data: {"type": "message_update", "content": {...}}

event: tool_execution_start
data: {"type": "tool_execution_start", "tool_name": "bash", ...}

event: agent_end
data: {"type": "agent_end"}
```

**理由**：
- 原始 Agent 事件类型（agent_start、message_update、tool_execution_end 等）直接映射为 SSE event type
- 前端通过 `EventSource` API 或 `fetch` + ReadableStream 消费
- SSE 自动重连机制提升稳定性

### D5: .pi 扩展系统 — Python 模块动态加载

**决策**：扩展为 Python 模块，放在项目目录的 `.pi/extensions/*.py` 中，通过 `importlib` 动态加载。

**扩展 API**：
```python
# .pi/extensions/my_extension.py
from pi_mono.coding_agent.core.extensions import ExtensionAPI

def create_extension(api: ExtensionAPI) -> None:
    # 注册自定义工具
    api.register_tool(name="my_tool", description="...", parameters={...}, execute=my_handler)

    # 注册提示词模板
    api.register_prompt("review", path=".pi/prompts/review.md")

    # 订阅事件
    api.on_message_end(my_callback)
```

**理由**：
- 与原 TS 扩展机制（`module.default(api)` 模式）一致
- Python 的 `importlib` 提供成熟的动态加载能力
- 扩展运行在同一进程中，直接访问 Agent API

### D6: 配置系统 — settings.yaml + 环境变量

**决策**：`.pi/settings.yaml` 作为项目级配置，环境变量用于敏感信息（API key），Pydantic 模型做验证。

**配置层级**（优先级从高到低）：
1. 环境变量（`PI_API_KEY`、`PI_BASE_URL` 等）
2. `.pi/settings.yaml`（项目级配置）
3. `~/.pi/settings.yaml`（全局用户配置）
4. 代码内默认值

**settings.yaml 结构**：
```yaml
# 模型配置
default_model: "gpt-4o"
base_url: "https://api.openai.com/v1"
api_key_env: "OPENAI_API_KEY"  # 引用环境变量名，不存储实际 key

# Agent 配置
thinking_level: "off"
max_turns: 50
auto_compact: true
compact_threshold: 100000

# 服务端配置
server:
  host: "127.0.0.1"
  port: 8080
  cors_origins: ["http://localhost:5173"]

# 自定义模型
models:
  - id: "local-llama"
    provider: "ollama"
    base_url: "http://localhost:11434/v1"

# 扩展
extensions:
  enabled: true
  paths: [".pi/extensions"]
```

**理由**：
- YAML 可读性好，适合手工编辑
- 环境变量隔离敏感数据，不会被提交到 git
- 多层级配置允许全局默认 + 项目覆盖
- Pydantic 验证确保配置正确性

### D7: Web-UI 适配方案

**决策**：web-ui 保留 Lit Web Components 技术栈，新增一个 REST API 客户端层替代原有的直接 Agent 调用。

**改动范围**：
- 新增 `RestAgentClient` 类，实现与 `Agent` 相同的接口（prompt、steer、abort、subscribe）
- `AgentInterface` 组件从直接使用 `Agent` 对象改为使用 `RestAgentClient`
- 会话存储从 IndexedDB 迁移到服务端（通过 REST API 读写）
- 保留前端的 UI 组件、artifacts 渲染、沙箱执行等功能

**理由**：
- 最小改动原则，只替换数据层，UI 层不变
- `RestAgentClient` 封装 HTTP 调用和 SSE 订阅，对上层组件透明

### D8: 项目结构

**决策**：保持现有 `src/pi_mono/` 单包结构，新增 `server` 子模块。

```
src/pi_mono/
├── ai/                      # LLM API 层
│   ├── types.py             # 核心类型（保留）
│   ├── stream.py            # 流式接口（保留）
│   ├── base.py              # LLMProvider Protocol（保留）
│   ├── api_registry.py      # Provider 注册（保留）
│   ├── openai_compat.py     # OpenAI 兼容 Provider（从 openai_completions.py 重构）
│   ├── models.py            # 模型定义（保留）
│   └── env_api_keys.py      # API key 管理（保留）
├── agent/                   # Agent 框架
│   ├── agent.py             # Agent 类（保留）
│   ├── agent_loop.py        # 循环逻辑（保留）
│   ├── types.py             # 事件和类型（保留）
│   └── events.py            # 事件总线（保留）
├── coding_agent/            # 编码代理
│   ├── core/
│   │   ├── agent_session.py # 会话管理（保留）
│   │   ├── session_manager.py
│   │   ├── settings_manager.py  # 改为支持 YAML
│   │   ├── model_registry.py
│   │   ├── model_resolver.py
│   │   ├── system_prompt.py
│   │   ├── tools/           # 内置工具（保留）
│   │   ├── extensions/      # 扩展系统（补全）
│   │   │   ├── types.py     # ExtensionAPI 定义
│   │   │   ├── loader.py    # 动态加载器
│   │   │   └── runner.py    # 扩展运行器
│   │   └── compaction/      # 上下文压缩（保留）
│   ├── modes/
│   │   └── print_mode.py    # 非交互模式（保留）
│   └── main.py              # CLI 入口（保留）
├── server/                  # 新增：REST 服务端
│   ├── app.py               # FastAPI 应用
│   ├── routes/
│   │   ├── sessions.py      # 会话路由
│   │   ├── messages.py      # 消息路由
│   │   ├── models.py        # 模型路由
│   │   └── settings.py      # 配置路由
│   ├── sse.py               # SSE 事件流
│   ├── session_registry.py  # 会话注册表
│   ├── schemas.py           # Pydantic 请求/响应模型
│   └── main.py              # 服务启动入口
└── config/                  # 新增：配置管理
    ├── settings.py          # Settings Pydantic 模型
    └── loader.py            # YAML 配置加载器
```

## Risks / Trade-offs

**[单进程 Agent 实例] → 内存和并发限制**
- 风险：多个活跃会话可能消耗大量内存（每个 Agent 持有完整消息历史）
- 缓解：实现会话超时自动清理；上下文压缩机制（已有 CompactionStrategy）限制消息列表增长

**[仅 OpenAI 兼容] → 功能受限**
- 风险：部分 LLM 特性（如 Anthropic 原生 thinking、Google 多模态）无法使用
- 缓解：OpenAI API 标准持续扩展（已支持 reasoning）；保留 `LLMProvider` Protocol，后续可按需添加 provider

**[SSE 单向流] → 扩展 UI 交互受限**
- 风险：原 TS 扩展支持双向 UI 交互（select、confirm、input 对话框），SSE 单向流无法直接实现
- 缓解：通过 SSE 发送 `extension_ui_request` 事件 + REST POST 回传响应，模拟双向交互

**[Python 动态扩展加载] → 安全风险**
- 风险：`importlib` 加载任意 Python 代码，可能执行恶意代码
- 缓解：仅加载项目 `.pi/extensions/` 目录下的文件（用户可控）；与原 TS 项目风险一致

**[已有代码重构] → 回归风险**
- 风险：重构 AI provider（合并为单一 openai_compat）可能破坏已有功能
- 缓解：TDD 驱动，先写测试覆盖现有行为，再进行重构
