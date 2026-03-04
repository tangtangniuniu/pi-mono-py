## Why

pi-mono 是一个基于 TypeScript 的 AI 代理框架，包含 7 个模块（ai、agent、coding-agent、tui、mom、pods、web-ui）。需要将其核心功能迁移到 Python 生态，以便：（1）利用 Python 丰富的 AI/ML 生态系统；（2）降低部署和集成门槛。当前已完成约 45% 的移植工作（agent 框架、AI 统一 API、coding-agent 基础架构），需要继续完成剩余部分并缩小范围，聚焦于 agent、coding-agent、ai（仅 OpenAI 兼容接口）和 web-ui 四个模块。

## What Changes

- **BREAKING** 从 TypeScript monorepo 迁移为 Python 单包项目（`pi_mono`），使用 `uv` + Python 3.11
- **BREAKING** AI 模块从支持 20+ 供应商缩减为仅支持 OpenAI 兼容接口（覆盖 OpenAI、兼容 API 的本地模型等）
- **BREAKING** 移除 mom（Slack 机器人）、pods（GPU 容器管理）、tui（终端 UI 库）模块
- 新增 Python REST 服务端模块，为 web-ui 提供 HTTP API 接口
- web-ui 保留原有 Lit Web Components 技术栈，但改为通过 REST API 访问 coding-agent（替代原有的直接 RPC/SDK 调用）
- 新增 `.pi/` 目录支持：扩展系统（Python 模块）、提示词模板、`settings.yaml` 配置文件
- 采用 TDD 驱动开发，目标测试覆盖率 80%+

## Capabilities

### New Capabilities

- `ai-openai-compat`：统一的 OpenAI 兼容 LLM API 层，支持流式传输、工具调用、模型配置和成本跟踪，仅实现 OpenAI 兼容接口协议
- `agent-core`：有状态 AI 代理框架核心，包括代理循环、工具执行、事件系统、消息管理和上下文压缩
- `coding-agent`：编码代理核心功能，包括会话管理、内置工具集（bash、read、write、edit、grep、find、ls）、系统提示词、模型注册和解析
- `rest-server`：Python REST 服务端模块，为 web-ui 提供 HTTP API，包括会话管理、消息流式传输（SSE）、工具执行状态推送
- `pi-extensions`：`.pi/` 目录扩展机制，支持 Python 扩展模块加载、提示词模板、`settings.yaml` 配置管理
- `web-ui-rest`：web-ui 改造，从直接调用改为通过 REST API 与 coding-agent 后端交互

### Modified Capabilities

（无已有 spec 需要修改）

## Impact

- **代码**：`src/pi_mono/` 下所有模块需要补全和重构；web-ui 前端需要修改 API 调用层
- **API**：新增 REST API 接口层（会话 CRUD、消息发送、SSE 流式推送、工具状态查询）
- **依赖**：Python 侧新增 FastAPI/Starlette（REST 框架）、uvicorn（ASGI 服务器）、httpx（HTTP 客户端）、openai（SDK）；移除 anthropic、google-genai 等非 OpenAI 兼容 SDK
- **构建**：从 npm workspaces 迁移为 uv + pyproject.toml；web-ui 保留原有构建工具链
- **配置**：新增 `.pi/settings.yaml` 作为统一配置入口
- **测试**：需从零建立完整测试体系（unit + integration + e2e），目标覆盖率 80%+
