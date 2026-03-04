# pi-mono-py

[pi-mono](https://github.com/badlogic/pi-mono) 的 Python 实现。统一 LLM API、Agent 运行时和终端编程助手。

## 功能模块

| 模块 | 说明 |
|------|------|
| `pi_mono.ai` | 统一多供应商 LLM API（OpenAI、Anthropic、Google 等 20+ 供应商） |
| `pi_mono.agent` | Agent 运行时，支持工具调用和状态管理 |
| `pi_mono.coding_agent` | 交互式终端编程助手 CLI |
| `pi_mono.server` | FastAPI REST 服务，通过 HTTP/SSE 暴露 Agent 能力 |
| `pi_mono.config` | YAML 配置加载和管理 |

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone https://github.com/badlogic/pi-mono.git pi-mono-py
cd pi-mono-py
uv sync --dev
```

### 运行

```bash
# 启动终端编程助手
uv run pi

# 启动 REST API 服务
uv run pi-server
```

## 架构

四层架构，每层只依赖其下方的层：

```
coding_agent  ─┐
server        ─┤──▶  agent  ──▶  ai
               │
config ────────┘
```

- **ai** — 供应商无关的流式 LLM 接口。定义 `LLMProvider` 协议，所有核心数据类型为不可变 frozen dataclass。
- **agent** — 有状态的 Agent 循环：LLM 调用 → 解析工具调用 → 执行工具 → 循环。事件驱动，通过 `Literal` 类型字段实现区分联合。
- **coding_agent** — 基于 Rich 的交互式终端 UI，内置 7 个工具（bash、read、write、edit、grep、find、ls），支持通过 `.pi/extensions/` 加载扩展。
- **server** — FastAPI 应用，提供会话管理、SSE 消息流、模型列表、配置管理等 REST 端点。
- **config** — 从 `.pi/settings.yaml` 加载 Pydantic 配置模型。

## 开发

```bash
# 运行全部测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/ai/test_types.py

# 运行单个测试
uv run pytest tests/ai/test_types.py::test_name

# 带覆盖率报告
uv run pytest --cov=pi_mono --cov-report=term

# 代码检查
uv run ruff check src/ tests/

# 代码格式化
uv run ruff format src/ tests/

# 类型检查
uv run mypy src/
```

## 配置

项目级配置放在 `.pi/settings.yaml`，全局配置放在 `~/.pi/settings.yaml`。

```yaml
server:
  host: "127.0.0.1"
  port: 8080

models:
  default: "claude-sonnet-4-20250514"
  thinking_level: "off"    # off, minimal, low, medium, high, xhigh
  max_turns: 100

extensions:
  enabled: true
```

## TypeScript 参考实现

`pi-mono/` 目录包含原始 TypeScript 单体仓库，模块对应关系：

| TypeScript | Python |
|------------|--------|
| `packages/ai/` | `src/pi_mono/ai/` |
| `packages/agent/` | `src/pi_mono/agent/` |
| `packages/coding-agent/` | `src/pi_mono/coding_agent/` |
| `packages/web-ui/` | 保持 TypeScript，通过 REST API 与 `pi-server` 交互 |

## 许可证

MIT
