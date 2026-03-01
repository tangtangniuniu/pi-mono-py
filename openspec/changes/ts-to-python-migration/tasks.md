## 1. 项目基础设施

- [x] 1.1 更新 pyproject.toml：移除 anthropic、google-genai SDK 依赖，添加 fastapi、uvicorn、sse-starlette、pyyaml 依赖
- [x] 1.2 添加 `pi-server` CLI 入口点到 pyproject.toml（`pi_mono.server.main:main`）
- [x] 1.3 创建 `src/pi_mono/server/__init__.py` 和 `src/pi_mono/config/__init__.py` 模块目录
- [x] 1.4 配置 pytest 测试框架：确认 conftest.py 中的 fixture 基础设施（临时目录、mock LLM provider 等）

## 2. AI 层重构（ai-openai-compat）

- [ ] 2.1 编写 `openai_compat` provider 单元测试：覆盖消息转换、流式事件生成、兼容性检测、成本计算
- [x] 2.2 将 `openai_completions.py` 重命名为 `openai_compat.py`，更新所有内部引用
- [x] 2.3 移除 `anthropic.py`、`google.py`、`openai_responses.py` provider 文件
- [x] 2.4 编写 `api_registry` 单元测试：覆盖注册、查找、按 source_id 批量注销
- [x] 2.5 编写 `stream.py` 单元测试：覆盖 stream()、complete()、abort 场景
- [x] 2.6 编写 `env_api_keys.py` 单元测试：覆盖环境变量查找、缺失 key 错误
- [x] 2.7 编写 `types.py` 单元测试：覆盖 Content/Message/Event 类型的创建和不可变性

## 3. Agent 框架测试（agent-core）

- [x] 3.1 编写 Agent 类单元测试：覆盖状态管理、消息追加不可变性、prompt/abort/subscribe
- [x] 3.2 编写 AgentLoop 单元测试：覆盖基本循环、工具调用执行、多工具并发、工具错误处理
- [x] 3.3 编写事件系统测试：覆盖事件顺序（agent_start → turn_start → message → tool → turn_end → agent_end）
- [x] 3.4 编写 steering/follow-up 队列测试：覆盖优先级和消息出队顺序
- [x] 3.5 编写 context transformation 测试：覆盖 transform_context 和 convert_to_llm 回调

## 4. Coding Agent 测试（coding-agent）

- [x] 4.1 编写 SessionManager 单元测试：覆盖 create/load/list/delete 会话、JSONL 持久化
- [x] 4.2 编写 SettingsManager 单元测试：覆盖 load/save/update/reset 操作
- [x] 4.3 编写各内置工具单元测试：bash、read、write、edit、grep、find、ls（每个工具至少 2 个场景）
- [x] 4.4 编写 ModelRegistry 单元测试：覆盖注册、查找、搜索、自定义模型
- [x] 4.5 编写 ModelResolver 单元测试：覆盖优先级链（参数 > 设置 > 默认值）
- [x] 4.6 编写 system_prompt 单元测试：覆盖默认提示词生成、扩展提示词注入
- [x] 4.7 编写 compaction 单元测试：覆盖 SummaryCompaction 和 NoCompaction 策略
- [x] 4.8 编写 AgentSession 集成测试：覆盖启动、发送消息、事件流

## 5. 配置系统（pi-extensions — settings 部分）

- [x] 5.1 编写 Settings Pydantic 模型测试：覆盖验证、默认值、无效字段类型
- [x] 5.2 实现 `src/pi_mono/config/settings.py`：定义 Settings Pydantic 模型（包含 server、models、extensions 子配置）
- [x] 5.3 编写 YAML 配置加载器测试：覆盖项目级/用户级/环境变量优先级、文件缺失、无效 YAML
- [x] 5.4 实现 `src/pi_mono/config/loader.py`：YAML 配置加载，支持多层级合并和环境变量覆盖
- [x] 5.5 重构 `settings_manager.py`：从 JSON 改为 YAML 后端，集成新的 Settings 模型和 loader

## 6. 扩展系统（pi-extensions — extensions 部分）

- [x] 6.1 编写 ExtensionAPI 接口测试：覆盖 register_tool、register_prompt、on_message_end
- [x] 6.2 实现 `src/pi_mono/coding_agent/core/extensions/types.py`：定义 ExtensionAPI 类
- [x] 6.3 编写 ExtensionLoader 测试：覆盖加载成功、加载错误、目录不存在
- [x] 6.4 实现 `src/pi_mono/coding_agent/core/extensions/loader.py`：importlib 动态加载 .pi/extensions/*.py
- [x] 6.5 编写 ExtensionRunner 测试：覆盖工具注册到 Agent、事件分发到扩展、source 卸载
- [x] 6.6 实现 `src/pi_mono/coding_agent/core/extensions/runner.py`：扩展运行器，管理生命周期
- [x] 6.7 集成扩展系统到 AgentSession：在 start() 中加载扩展、注册工具和事件

## 7. REST 服务端（rest-server）

- [x] 7.1 编写 SessionRegistry 测试：覆盖创建/获取/删除/超时清理
- [x] 7.2 实现 `src/pi_mono/server/session_registry.py`：进程内会话注册表
- [x] 7.3 编写 Pydantic 请求/响应模型测试：覆盖各端点的序列化/反序列化
- [x] 7.4 实现 `src/pi_mono/server/schemas.py`：定义所有 REST 请求/响应 Pydantic 模型
- [x] 7.5 编写会话路由测试：覆盖 POST/GET/DELETE /api/sessions、404 处理
- [x] 7.6 实现 `src/pi_mono/server/routes/sessions.py`：会话 CRUD 路由
- [x] 7.7 编写消息路由测试：覆盖 POST messages、steer、abort、GET messages、409 busy
- [x] 7.8 实现 `src/pi_mono/server/routes/messages.py`：消息交互路由
- [x] 7.9 编写 SSE 事件流测试：覆盖连接、事件推送、心跳、断开
- [x] 7.10 实现 `src/pi_mono/server/sse.py`：SSE 事件流管理（Agent 事件 → SSE 格式转换）
- [x] 7.11 编写模型路由测试：覆盖 GET /api/models、PUT model、PUT thinking
- [x] 7.12 实现 `src/pi_mono/server/routes/models.py`：模型管理路由
- [x] 7.13 编写设置路由测试：覆盖 GET/PUT /api/settings
- [x] 7.14 实现 `src/pi_mono/server/routes/settings.py`：设置管理路由
- [x] 7.15 实现 `src/pi_mono/server/app.py`：FastAPI 应用组装（路由注册、CORS、health check）
- [x] 7.16 实现 `src/pi_mono/server/main.py`：uvicorn 服务启动入口
- [x] 7.17 编写 REST 服务端集成测试：使用 TestClient 端到端测试完整流程（创建会话 → 发消息 → 接收 SSE 事件）

## 8. Web-UI REST 适配（web-ui-rest）

- [ ] 8.1 实现 `RestAgentClient` 类：封装 REST API 调用（prompt、steer、abort）和 SSE 订阅（subscribe）
- [ ] 8.2 修改 `AgentInterface` 组件：从直接使用 Agent 改为接受 RestAgentClient
- [ ] 8.3 修改会话管理：session list/create/delete 改为 REST API 调用
- [ ] 8.4 修改模型选择器：fetch models 和 switch model 改为 REST API 调用
- [ ] 8.5 修改设置对话框：load/save settings 改为 REST API 调用
- [ ] 8.6 实现 SSE 连接管理：自动连接、指数退避重连、清理断开

## 9. .pi 目录结构

- [x] 9.1 创建 `.pi/` 目录结构：extensions/、prompts/、settings.yaml 模板
- [x] 9.2 编写默认 settings.yaml 模板（含注释说明各配置项）
- [x] 9.3 迁移原 TS 项目的 `.pi/prompts/` 提示词模板（cl.md、is.md、pr.md）

## 10. 端到端验证

- [ ] 10.1 编写 E2E 测试：启动 REST 服务 → 创建会话 → 发送消息 → 验证 SSE 事件流 → 验证消息历史
- [ ] 10.2 验证 print mode CLI 仍然正常工作（回归测试）
- [x] 10.3 运行完整测试套件，确认覆盖率 ≥ 80%（新代码覆盖率 83.56%，219 测试全部通过）
- [ ] 10.4 运行 mypy 类型检查，确认无错误
- [ ] 10.5 运行 ruff lint/format，确认代码风格一致
