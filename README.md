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
| **Generator** | Reads source via MCP → calls DeepSeek-V4 → produces pytest file<br>通过 MCP 读取源码 → 调用 DeepSeek-V4 → 生成 pytest 测试文件 |
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
DEEPSEEK_MODEL=deepseek-chat
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
  --model TEXT            DeepSeek model name  [default: deepseek-chat]
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
    │   ├── mcp_context.py      # MCP filesystem context / MCP 文件上下文
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
| LLM | DeepSeek-V4 | OpenAI-compatible, cost-effective / 兼容 OpenAI 接口，成本优化 |
| Sandbox | E2B micro-VM | Process/filesystem/network isolation / 进程+文件系统+网络全隔离 |
| Context | MCP filesystem | Standardized LLM file access / 标准化 LLM 文件访问协议 |
| Coverage | pytest-cov | In-sandbox line coverage reporting / 沙箱内行覆盖率报告 |
| Observability | LangSmith | End-to-end agent tracing / 端到端 Agent 链路追踪 |
| Validation | Pydantic v2 | Type-safe schemas + SecretStr / 类型安全+密钥保护 |
| CLI | Click + Rich | Colored real-time status output / 彩色实时状态输出 |
| Config | pydantic-settings + python-dotenv | Auto-load .env with type safety / 类型安全的环境变量加载 |
| Package Manager | uv | 10–100x faster than pip |
