# Auto-SDET

> Autonomous Unit Test Generation Agent powered by LangGraph FSM + E2B Sandbox + DeepSeek-V4
>
> 基于 LangGraph 有限状态机 + E2B 沙箱 + DeepSeek-V4 的自主单元测试生成 Agent

```
auto-sdet test src/calculator.py
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Auto-SDET  │  target: calculator.py         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

✨ [Generator]  Reading file context (MCP)...
✨ [Generator]  Context loaded: 2 dependency file(s)
✨ [Generator]  Calling DeepSeek-V4...
✓  [Generator]  Generated test_calculator.py (42 lines)

⚡ [Executor]   Creating E2B sandbox...
⚡ [Executor]   Running pytest in sandbox...
✗  [Executor]   Tests failed: ImportError: No module named 'numpy'

🧠 [Reflector]  Analyzing error (attempt 1/3)...
🧠 [Reflector]  Changes made:
@@ -3,7 +3,7 @@
-import numpy as np
+# numpy removed - not needed
✓  [Reflector]  Patch applied, re-executing...

⚡ [Executor]   Running pytest in sandbox...
✓  [Executor]   All tests passed!  Coverage: 88%

✨ Test file saved to: tests/test_calculator.py  |  Coverage: 88%
```

---

## How It Works / 工作原理

Auto-SDET implements a **deterministic FSM** (Finite State Machine) using LangGraph. Unlike a simple LLM chain, the FSM makes every execution path predictable and traceable.

Auto-SDET 使用 LangGraph 实现**确定性有限状态机（FSM）**。与简单的 LLM 链不同，FSM 使每条执行路径都可预测、可追溯。

```
[CLI] → [Generator] → [Executor] ──→ END (success)
                           │
                           ├──→ [Reflector] → [Executor] (self-healing loop / 自愈循环)
                           │
                           └──→ END (max retries exceeded)
```

| Node / 节点 | Responsibility / 职责 |
|-------------|----------------------|
| **Generator** | LLM uses `read_file` / `list_directory` tools (function calling) to gather context, then emits pytest file<br>LLM 通过 `read_file` / `list_directory` 工具调用主动收集上下文，再生成 pytest 文件 |
| **Executor** | Spins up E2B micro-VM → uploads files → runs pytest → captures results<br>启动 E2B 微型虚拟机 → 上传文件 → 运行 pytest → 捕获结果 |
| **Reflector** | Uses Chain-of-Thought over full error history to patch test code<br>基于完整错误历史进行 CoT 推理，修复测试代码 |
| **Router** | Routes to next state based on exit_code and retry count<br>根据 exit_code 和重试次数决定下一个状态 |

---

## Quick Start / 快速开始

### Prerequisites / 前置要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended / 推荐) or pip
- [DeepSeek API Key](https://platform.deepseek.com/)
- [E2B API Key](https://e2b.dev/)

### 1. Clone & Install / 克隆并安装

```bash
git clone https://github.com/yourname/auto-sdet.git
cd auto-sdet

# Option A: uv (recommended / 推荐)
uv sync

# Option B: pip
pip install -e .
```

### 2. Configure Environment / 配置环境变量

```bash
cp .env.example .env
# Edit .env and fill in your API keys / 编辑 .env 填入你的 API Key
```

```env
# Required / 必填
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-fast
E2B_API_KEY=e2b_xxx
E2B_SANDBOX_TIMEOUT=60
MAX_RETRIES=3

# Optional — LangSmith observability / 可选 — LangSmith 可观测性
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxx
LANGCHAIN_PROJECT=auto-sdet
```

### 3. Run / 运行

```bash
# Try the included example / 运行内置示例
auto-sdet test examples/calculator.py

# Point it at your own module / 指向你自己的模块
auto-sdet test src/mymodule.py --output-dir ./tests/

# Verbose output to see each node / 显示每个节点的详细日志
auto-sdet test src/mymodule.py --verbose
```

---

## CLI Reference / 命令参考

```
Usage: auto-sdet test [OPTIONS] TARGET

  Generate unit tests for TARGET source file.
  为目标源文件生成单元测试。

Options:
  --max-retries INTEGER   Maximum self-healing retry attempts  [default: 3]
                          最大自愈重试次数
  --model TEXT            DeepSeek model name  [default: deepseek-v4-fast]
                          DeepSeek 模型名称
  --output-dir PATH       Output directory for generated tests
                          生成测试文件的输出目录
  --verbose               Show detailed node logs / 显示详细节点日志
  --help                  Show this message and exit.
```

---

## Architecture / 架构详解

### Agent State / 状态定义

All nodes communicate through a shared `AgentState` TypedDict. Each node only returns the fields it modifies — LangGraph merges them automatically.

所有节点通过共享的 `AgentState` TypedDict 通信。每个节点只返回它修改的字段，LangGraph 自动合并。

```python
class AgentState(TypedDict, total=False):
    source_code: str
    source_path: str
    context_files: dict[str, str]
    test_code: str
    execution_result: ExecutionResult | None
    retry_count: int
    max_retries: int
    error_history: list[str]       # Accumulated across retries / 跨重试累积
    status: Literal["generating", "executing", "reflecting", "done", "failed"]
```

### Self-Healing Loop / 自愈循环

The Reflector uses **Chain-of-Thought prompting** to avoid superficial fixes. The full `error_history` is injected into each call to prevent the A→B→A oscillation pattern.

Reflector 使用 **CoT 提示**避免表面修复。完整的 `error_history` 注入每次调用，防止 A→B→A 振荡。

```
Step 1 — Error Classification  错误分类  (ImportError / AssertionError / TypeError ...)
Step 2 — Root Cause Analysis   根因分析  (定位具体行号，交叉验证源码)
Step 3 — Minimal Fix Strategy  最小修复  (禁止删除测试用例来"修复"失败)
Step 4 — Output complete fixed file  输出完整修复文件
```

### Sandbox Isolation / 沙箱隔离

Every test run happens inside a **fresh E2B micro-VM**, destroyed after the run regardless of outcome.

每次测试运行都在全新的 **E2B 微型虚拟机**中执行，无论结果如何都会被销毁。

| Approach / 方案 | Isolation / 隔离性 | Startup / 启动速度 | Complexity / 复杂度 |
|-----------------|-------------------|--------------------|---------------------|
| **E2B micro-VM** | Highest / 最高 | ~300ms | Low / 低 |
| Docker | High / 高 | 1–5s | Medium / 中 |
| subprocess | None / 无 | Instant / 即时 | Low / 低 |
| AWS Lambda | High / 高 | 1–10s | High / 高 |

### AST-based Dependency Resolution / AST 依赖解析

The Executor uses Python's `ast` module to parse the source file and automatically locate internal project dependencies (e.g., `from auto_sdet.models.schemas import AgentState`), upload them flat into `/workspace/`, and rewrite import statements so tests run correctly in the sandbox.

Executor 使用 Python `ast` 模块解析源文件，自动定位项目内部依赖（如 `from auto_sdet.models.schemas import AgentState`），扁平化上传到沙箱 `/workspace/` 并重写 import 语句，保证测试在沙箱中正确运行。

### Coverage Reporting / 覆盖率报告

`pytest-cov` is installed inside the sandbox and invoked with `--cov=<module> --cov-report=term-missing`. The coverage percentage is parsed from stdout and surfaced in `ExecutionResult.coverage_pct`, displayed in both the Executor log and the final CLI summary.

沙箱内集成 `pytest-cov`，通过 `--cov=<module> --cov-report=term-missing` 调用。覆盖率百分比从 stdout 解析并存入 `ExecutionResult.coverage_pct`，在 Executor 日志和最终 CLI 输出中展示。

### Reflector Diff Display / Reflector 差异显示

When the Reflector patches the test code, a unified diff between the previous and patched versions is printed to the terminal — green `+` lines for additions, red `-` for removals, cyan `@@` for hunk locations — making the self-healing process transparent and debuggable.

Reflector 修复测试代码时，会在终端输出修改前后的统一 diff —— 绿色 `+` 表示新增、红色 `-` 表示删除、青色 `@@` 表示位置 —— 使自愈过程透明可调试。

### LLM Tool Use (Function Calling) / LLM 工具调用

The Generator node implements true Agentic behaviour: file-reading capabilities are exposed as `@tool`-decorated LangChain tools (`read_file`, `list_directory`) and bound to the LLM via `llm.bind_tools(...)`. Inside a manual tool-calling loop, the LLM autonomously decides which files to inspect (target source + internal dependencies) before emitting the final test code. A hard iteration cap prevents runaway tool calls.

Generator 节点实现真正的 Agentic 行为：文件读取能力封装为 `@tool` 装饰的 LangChain 工具（`read_file`、`list_directory`），通过 `llm.bind_tools(...)` 绑定到 LLM。在手写的工具调用循环中，LLM 自主决定读取哪些文件（目标源码+内部依赖）后再生成最终测试代码。硬性迭代上限防止失控调用。

```
LLM ─→ tool_call(read_file, target.py) ─→ ToolMessage(content)
    ─→ tool_call(list_directory, parent_dir) ─→ ToolMessage(content)
    ─→ tool_call(read_file, dep.py) ─→ ToolMessage(content)
    ─→ FINAL: ```python ... ```  (no more tool_calls)
```

### Multi-Provider LLM Abstraction / 多模型 Provider 抽象

A single `get_llm()` factory in `tools/llm_factory.py` returns a `BaseChatModel` based on the `LLM_PROVIDER` env var. Both Generator and Reflector are provider-agnostic and need zero code changes when switching backends.

`tools/llm_factory.py` 中的统一 `get_llm()` 工厂函数根据 `LLM_PROVIDER` 环境变量返回 `BaseChatModel` 实例。Generator 和 Reflector 完全 provider 无关，切换后端零代码改动。

| Provider | Backend / 后端 | API Key Required / 需 API Key |
|----------|----------------|------------------------------|
| `deepseek` | DeepSeek API (default) | ✓ |
| `openai` | OpenAI API | ✓ |
| `anthropic` | Anthropic Claude API | ✓ (and `pip install langchain-anthropic`) |
| `ollama` | Local Ollama (OpenAI-compatible) | ✗ — fully offline |

Switch providers by setting `LLM_PROVIDER` and the matching `<PROVIDER>_*` variables in `.env` — no code changes needed.

只需在 `.env` 中设置 `LLM_PROVIDER` 和对应的 `<PROVIDER>_*` 变量即可切换后端，无需改动代码。

### Observability with LangSmith / 基于 LangSmith 的可观测性

When `LANGCHAIN_TRACING_V2=true` is set, every Agent run is automatically traced to LangSmith, capturing:
- Full DAG of node execution (Generator → Executor → Reflector → Executor ...)
- Per-LLM-call input/output, token usage, and latency
- Tool calls and their results
- Total run duration and token consumption

启用 `LANGCHAIN_TRACING_V2=true` 后，每次 Agent 运行会自动上报到 LangSmith，记录：节点执行 DAG、每次 LLM 调用的输入输出/Token/耗时、工具调用结果、总运行时长与 Token 消耗。

---

## Benchmark Results / 基准测试结果

Auto-SDET was benchmarked against **19 open-source Python algorithm modules** from [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python), spanning sorting, searching, math, string manipulation, and data structures.

Auto-SDET 在 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) 的 **19 个开源 Python 算法模块**上进行基准测试，涵盖排序、搜索、数学、字符串处理和数据结构。

| Metric / 指标 | Value / 数值 |
|---------------|--------------|
| One-shot pass rate / 一次通过率 | **68.4%** |
| Final pass rate (after self-healing) / 自愈后最终通过率 | **94.7%** |
| Average retries / 平均自愈次数 | 0.53 |
| Average code coverage / 平均代码覆盖率 | **72.2%** |
| Average wall time per module / 单模块平均耗时 | ~200s |
| Average token consumption per module / 单模块平均 Token 消耗 | ~14K |

Reproduce locally / 本地复现：

```bash
# Clone benchmark dataset / 克隆基准数据集
git clone https://github.com/TheAlgorithms/Python ../algorithms-bench

# Run full benchmark / 运行全量基准
python scripts/benchmark.py

# Quick smoke test (first 3 files) / 快速冒烟测试（前 3 个文件）
python scripts/benchmark.py --limit 3
```

Results are saved to `benchmark_results/results.json` with per-file metrics + aggregate summary.

---

## Dogfooding — Unit Tests / 单元测试（吃自己的狗粮）

Auto-SDET tests its own source code using a **two-layer testing pyramid**:

Auto-SDET 用**双层测试金字塔**测试自己的源码：

1. **Unit tests** (`tests/`) — pure functions and helpers, mostly auto-generated by Auto-SDET itself
2. **End-to-end benchmark** (`scripts/benchmark.py`) — exercises the full Agent loop on 19 real modules

| Module / 模块 | Coverage / 覆盖率 | Source / 测试来源 |
|---------------|-------------------|---------------------|
| `graph/router.py` | **100%** | auto-generated |
| `models/schemas.py` | **100%** | auto-generated |
| `prompts/generator.py` | **100%** | auto-generated |
| `prompts/reflector.py` | **100%** | auto-generated |
| `config.py` | 95% | indirect |
| `tools/e2b_sandbox.py` (helpers) | 50% | hand-written |
| `graph/nodes/generator.py` (helpers) | 36% | hand-written |
| **Project total / 项目整体** | **43%** | — |

The remaining 57% — LLM-calling nodes, sandbox executor, CLI — are validated by the e2e benchmark, where mocking would be more brittle than running the real flow.

剩余 57%（LLM 调用节点、沙箱执行器、CLI）由端到端基准测试验证 —— 在这些场景下，mock 测试比直接跑真实流程更脆弱。

```bash
# Run unit tests / 运行单元测试
pip install -e ".[dev]"
pytest tests/ --cov=src/auto_sdet --cov-report=term-missing
```

---

## Project Structure / 项目结构

```
auto-sdet/
├── pyproject.toml              # Project config & dependencies / 项目配置与依赖
├── .env.example                # Environment variable template / 环境变量模板
├── examples/
│   └── calculator.py           # Demo target file / 示例目标文件
├── scripts/
│   └── benchmark.py            # Benchmark runner / 基准测试脚本
└── src/auto_sdet/
    ├── cli.py                  # Click CLI entry point / CLI 入口
    ├── config.py               # pydantic-settings config loader / 配置加载
    ├── graph/
    │   ├── graph.py            # LangGraph FSM assembly / FSM 组装
    │   ├── router.py           # Conditional edge routing / 条件边路由
    │   └── nodes/
    │       ├── generator.py    # LLM test generation node / 测试生成节点
    │       ├── executor.py     # E2B sandbox execution node / 沙箱执行节点
    │       └── reflector.py    # CoT diff & patching / CoT 差异修复节点
    ├── tools/
    │   ├── llm_factory.py      # Multi-provider LLM factory / 多模型 LLM 工厂
    │   ├── mcp_context.py      # MCP filesystem context / MCP 文件上下文
    │   ├── mcp_tools.py        # @tool wrappers for LLM tool_call / LLM 工具调用封装
    │   └── e2b_sandbox.py      # E2B sandbox + AST dep resolution / 沙箱+AST依赖解析
    ├── models/
    │   └── schemas.py          # Pydantic models + AgentState / 数据模型
    └── prompts/
        ├── generator.py        # Generator prompt templates / 生成阶段 Prompt
        └── reflector.py        # Reflector CoT prompt templates / 修复阶段 Prompt
```

---

## Tech Stack / 技术栈

| Component / 组件 | Technology / 技术 | Why / 选型原因 |
|------------------|-------------------|----------------|
| FSM Orchestration | LangGraph 0.2+ | Native StateGraph + conditional edges / 原生状态机+条件边 |
| LLM (default) | DeepSeek-V4 | OpenAI-compatible, cost-effective / 兼容 OpenAI 接口，成本优化 |
| LLM (alternatives) | OpenAI / Claude / Ollama | Pluggable via `LLM_PROVIDER` env / 通过环境变量切换 |
| Sandbox | E2B micro-VM | Process/filesystem/network isolation / 进程+文件系统+网络全隔离 |
| Context | MCP filesystem | Standardized LLM file access / 标准化 LLM 文件访问协议 |
| Coverage | pytest-cov | In-sandbox line coverage reporting / 沙箱内行覆盖率报告 |
| Observability | LangSmith | End-to-end agent tracing / 端到端 Agent 链路追踪 |
| Validation | Pydantic v2 | Type-safe schemas + SecretStr / 类型安全+密钥保护 |
| CLI | Click + Rich | Colored real-time status output / 彩色实时状态输出 |
| Config | pydantic-settings + python-dotenv | Auto-load .env with type safety / 类型安全的环境变量加载 |
| Package Manager | uv | 10–100x faster than pip |
