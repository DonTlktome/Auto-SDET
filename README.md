# Auto-SDET

> Autonomous Unit Test Generation Agent powered by LangGraph FSM + E2B Sandbox + DeepSeek-V3
>
> 基于 LangGraph 有限状态机 + E2B 沙箱 + DeepSeek-V3 的自主单元测试生成 Agent

```
auto-sdet test src/calculator.py
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Auto-SDET  │  target: calculator.py         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

✨ [Generator]  Reading file context (MCP)...
✨ [Generator]  Context loaded: 2 dependency file(s)
✨ [Generator]  Calling DeepSeek-V3...
✓  [Generator]  Generated test_calculator.py (42 lines)

⚡ [Executor]   Creating E2B sandbox...
⚡ [Executor]   Running pytest in sandbox...
✗  [Executor]   Tests failed: ImportError: No module named 'numpy'

🧠 [Reflector]  Analyzing error (attempt 1/3)...
✓  [Reflector]  Patch applied, re-executing...

⚡ [Executor]   Running pytest in sandbox...
✓  [Executor]   All tests passed!

✨ Test file saved to: tests/test_calculator.py
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
| **Generator** | Reads source via MCP → calls DeepSeek-V3 → produces pytest file<br>通过 MCP 读取源码 → 调用 DeepSeek-V3 → 生成 pytest 测试文件 |
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
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
E2B_API_KEY=e2b_xxx
E2B_SANDBOX_TIMEOUT=60
MAX_RETRIES=3
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

---

## Project Structure / 项目结构

```
auto-sdet/
├── pyproject.toml              # Project config & dependencies / 项目配置与依赖
├── .env.example                # Environment variable template / 环境变量模板
├── examples/
│   └── calculator.py           # Demo target file / 示例目标文件
└── src/auto_sdet/
    ├── cli.py                  # Click CLI entry point / CLI 入口
    ├── config.py               # pydantic-settings config loader / 配置加载
    ├── graph/
    │   ├── graph.py            # LangGraph FSM assembly / FSM 组装
    │   ├── router.py           # Conditional edge routing / 条件边路由
    │   └── nodes/
    │       ├── generator.py    # LLM test generation node / 测试生成节点
    │       ├── executor.py     # E2B sandbox execution node / 沙箱执行节点
    │       └── reflector.py    # CoT error analysis & patching / CoT 修复节点
    ├── tools/
    │   ├── mcp_context.py      # MCP filesystem context / MCP 文件上下文
    │   └── e2b_sandbox.py      # E2B sandbox wrapper / E2B 沙箱封装
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
| LLM | DeepSeek-V3 | OpenAI-compatible, cost-effective / 兼容 OpenAI 接口，成本优化 |
| Sandbox | E2B micro-VM | Process/filesystem/network isolation / 进程+文件系统+网络全隔离 |
| Context | MCP filesystem | Standardized LLM file access / 标准化 LLM 文件访问协议 |
| Validation | Pydantic v2 | Type-safe schemas + SecretStr / 类型安全+密钥保护 |
| CLI | Click + Rich | Colored real-time status output / 彩色实时状态输出 |
| Config | pydantic-settings | Auto-load .env with type safety / 类型安全的环境变量加载 |
| Package Manager | uv | 10–100x faster than pip |
