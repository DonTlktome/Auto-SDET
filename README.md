# Auto-SDET

> **Autonomous Unit Test Generation Agent** powered by LangGraph FSM · LLM-as-Judge · Memory-R1 · E2B Sandbox
>
> **自主单元测试生成 Agent** · LangGraph 状态机 · LLM-as-Judge 质量门禁 · Memory-R1 情景记忆 · E2B 沙箱

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-5--node%20FSM-FF6F00)
![Memory-R1](https://img.shields.io/badge/Memory--R1-CRUD-7B1FA2)
![Tests](https://img.shields.io/badge/tests-182%20passing-43A047)
![Final Pass Rate](https://img.shields.io/badge/final%20pass%20rate-100%25-43A047)
![Coverage](https://img.shields.io/badge/avg%20coverage-72.9%25-43A047)

---

## TL;DR

**Give it a Python source file → it generates pytest code and self-heals until the tests pass.** 19 modules in the open-source benchmark suite, **100% final pass rate** (vs baseline 94.7%), 72.9% average coverage.

**给一个 Python 源文件，生成 pytest 测试代码 + 自愈直到通过。** 19 模块开源基准实测最终通过率 **100%**（超 baseline 94.7%）、平均覆盖率 72.9%。

| What's interesting | 中文 |
|---|---|
| 🧠 **5-node LangGraph FSM** (Generator → Evaluator → Executor → Reflector → MemoryManager) with dual circuit breakers for bounded termination | 五节点闭环状态机 + 双重熔断保证有界终止 |
| 🛡️ **Dual-source quality gate** — LLM-as-Judge (semantic) + ruff static analysis (deterministic), compensating for the same-model self-evaluation blind spot | 双源 quality gate — LLM 评模糊语义 + ruff 评客观静态，补完大模型同源自审盲区 |
| 💾 **Memory-R1 (Lu 2025) episodic memory** — LLM-driven CRUD operations (ADD/UPDATE/DELETE/NOOP) over ChromaDB, with multi-layer safety guards preventing LLM mis-deletion | Memory-R1 跨任务情景记忆 — LLM 控制器做 CRUD 操作 + 多层防误删护栏 |
| 🔒 **E2B Micro-VM sandbox** with AST-based dependency resolution — generated code runs in a fresh isolated VM, ~300ms startup | E2B 微型虚拟机沙箱 + AST 自动依赖解析 |
| 🔬 **7 benchmark rounds + 4 engineering case studies** documenting the full architecture evolution (json_mode protocol attribution / LLM-as-Judge limits / Memory-R1 upgrade / ROI-driven optimization) | 7 轮 benchmark + 4 个独立工程 case study 完整归档架构演进 |

📖 **Engineering deep-dive:** [`RESUME_IMPROVEMENT.md`](RESUME_IMPROVEMENT.md) · **Code walkthrough:** [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)

---

## Demo / 运行示例

```
auto-sdet test src/calculator.py
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Auto-SDET  │  target: calculator.py                                       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

✨ [Generator]    Calling LLM with read_file / list_directory tools...
✨ [Generator]    Generated test_calculator.py (42 lines)

⚖  [Evaluator]    Scoring 4 dimensions via LLM-as-Judge...
⚖  [Evaluator]    Score 0.83 (testable_cov=0.85 / assertion=0.80 / mock=0.90 / consistency=0.78) → PASS

⚡ [Executor]     Running pytest in E2B sandbox...
✗  [Executor]     Tests failed: ImportError: No module named 'numpy'

🧠 [Reflector]    Retrieved 1 similar past case from episodic memory
🧠 [Reflector]    classification=ImportError  root_cause=...  strategy=...
🧠 [Reflector]    Patch applied (3 lines changed)

🗂 [MemoryManager] Decision: UPDATE (replaces stale neighbor with sharper root_cause)

⚖  [Evaluator]    Score 0.86 → PASS
⚡ [Executor]     All tests passed!  Coverage: 88%

✨ Test file saved to: tests/test_calculator.py  |  Coverage: 88%
```

---

## How It Works / 工作原理

Auto-SDET 是一个 **5 节点闭环 LangGraph 状态机**。每条 LLM 产物（无论来自首生成还是修复）都必须经过质量门禁；每次失败修复都会沉淀进可被未来运行复用的记忆库。

```
                   ┌────────────┐
                   │ Generator  │  LLM 主动调用 read_file/list_directory 收集上下文
                   └──────┬─────┘
                          ↓
                   ┌────────────┐
        ┌─────────▶│ Evaluator  │  LLM-as-Judge 四维度评分 (≥0.5 通过)
        │          └──────┬─────┘
        │                 │ pass
        │                 ↓
        │          ┌────────────┐  E2B 沙箱跑 pytest + 覆盖率
        │          │  Executor  │
        │          └──┬──────┬──┘
        │             │ pass │ fail
        │             ↓      ↓
        │            END   ┌────────────┐
        │                  │ Reflector  │  CoT 结构化输出 + 检索相似历史失败
        │                  └──────┬─────┘
        │                         ↓
        │                  ┌──────────────────┐
        └──────────────────│ MemoryManager    │  Memory-R1 CRUD：增/改/删/跳过
                           └──────────────────┘
```

| 节点 | 职责 |
|------|------|
| **Generator** | LLM 通过 function calling 工具自主读取目标源码与内部依赖，再产出 pytest 文件 |
| **Evaluator** | LLM-as-Judge 对 4 个维度（可测代码覆盖度 / 断言质量 / Mock 合理性 / 代码一致性）打分；低分代码不进沙箱直接打回 |
| **Executor** | 启动 E2B 微型虚拟机运行 pytest，收集 stdout/stderr/exit_code 与覆盖率 |
| **Reflector** | 基于结构化 CoT（错误分类 / 根因 / 修复策略 / 受影响行）产出修复版测试代码，并检索相似历史失败做为提示 |
| **MemoryManager** | 对 Reflector 产出的轨迹做 ADD / UPDATE / DELETE / NOOP 决策，维持记忆库不膨胀且不腐化 |

**终止性保证：** `retry_count ≤ max_retries`（执行失败循环上限）+ `evaluator_reject_count ≤ 2`（评估打回循环上限），两个计数器单调递增，FSM 一定有界。

---

## Highlights / 项目亮点

### 1. Dual-source quality gate / 双源 quality gate

首生成的测试代码强制经过 Evaluator，**两条信号源并行评估**：

- **LLM-as-Judge（模糊语义维度）** — 针对 4 个维度打分：`testable_coverage` / `assertion_quality` / `mock_correctness` / `code_consistency`
- **ruff 静态分析（客观确定维度）** — 在 < 100ms 内捕获语法错误、未定义符号、未使用 import、明显 lint 问题，**fatal 级别（语法 / undefined name）直接短路 LLM 调用打回 Reflector**

为什么两条信号源都要？LLM-as-Judge 有"**自审同质性**"盲区 —— Generator 和 Evaluator 是同一个 LLM，对自身常犯的 subtle bug 存在共同盲区（详见 `binary_search.py` case study）。ruff 是独立确定性工具，能客观捕获 LLM judge 看不出的硬错误。**职责分离：LLM 评模糊维度，ruff 评静态维度**。

**反 coverage gaming 设计：**
- 显式承认"主动跳过不可测代码"是可接受的（`skipped_intentionally` 字段），不强迫 LLM 用假断言去刷覆盖率
- 弱点描述强制 `actionable=True`，禁止"建议增加更多测试"这类无法执行的空话
- Evaluator 自身打回次数也设上限，避免与 Reflector 形成无限震荡

**非对称门禁：** 首次过 Evaluator，retry 路径直达 Executor（pytest 才是 ground truth，软评在 retry 路径上 ROI 不正 — 基于 5 轮 benchmark 实测做的对称性放松）。

### 2. Memory-R1 episodic memory / Memory-R1 情景记忆

测试失败修复后，Reflector 把"错误分类 + 根因 + 修复策略 + 修复摘要"打包成一条轨迹（trajectory），交给 MemoryManager 决策入库方式：

| 操作 | 何时使用 |
|------|---------|
| **ADD** | 新轨迹与库内邻居均不重复，纳入记忆 |
| **UPDATE** | 邻居方向相同但根因 / 修复摘要更模糊，用新轨迹替换 |
| **DELETE** | 新证据明确证伪邻居（软删除，仅置 `deprecated` 标记不真删） |
| **NOOP** | 记忆库已有等价轨迹，跳过避免污染检索 |

向量检索复用 ChromaDB 持久化客户端 + sentence-transformers 嵌入。多重安全护栏防止 LLM 误删：低置信度 DELETE 自动降级为 NOOP；目标 ID 不在邻居集合中（疑似幻觉）也降级为 NOOP；Manager 自身失败默认走 ADD（保守保留数据）。

下次同类失败再次发生时，Reflector 检索 top-k 相似历史失败注入 prompt，避免重蹈覆辙。

### 3. Structured CoT reflection / 结构化思维链反思

Reflector 不再依赖自由文本输出 + 正则提取代码块。改用 Pydantic 强制约束：

```
ReflectionResult {
    error_classification: Literal[ImportError, AssertionError, TypeError, ...]
    root_cause: str          # 一句话定位具体代码位置
    fix_strategy: str        # 一句话陈述修复方法
    affected_lines: str      # 涉及的行号或区间
    fixed_code: str          # 完整可运行的修复后测试文件
}
```

诊断字段升级为一等公民数据，CLI 实时展示，LangSmith trace 完整可见。结构化输出失败直接饱和 retry_count 终止重试（同样输入会同样失败，没必要无意义循环烧 token）。

### 4. Multi-provider LLM abstraction / 多模型抽象

`get_llm()` 根据环境变量返回 `BaseChatModel`，业务代码 provider 无关。支持 DeepSeek / OpenAI / Claude / Ollama。**节点级协议精细化配置**：

- **Generator** 是 multi-turn `bind_tools`，DeepSeek V4 thinking 在多轮场景下需要 `reasoning_content` 跨轮回传（LangChain ChatDeepSeek 目前不支持）→ 关 thinking
- **Reflector / Evaluator / MemoryManager** 是 single-turn `with_structured_output(method="json_mode")`，走 `response_format` 协议绕开 `tool_choice` 限制 → **保留 thinking 同时拿结构化输出**

`get_llm(multi_turn_tool_calling=...)` 参数封装这个区分。这是基于读 DeepSeek 官方协议文档 + `probe_structured_thinking.py` 最小验证脚本归因得出的节点级配置策略。

### 5. End-to-end observability / 端到端可观测性

接入 LangSmith Tracing 后，单次运行的节点 DAG、每次 LLM 输入输出、token 消耗、工具调用结果、Reflector diff 全部可回放。CLI 同步展示 Evaluator 评分细分、Reflector 诊断字段、MemoryManager CRUD 决策与原因。

---

## Quick Start / 快速开始

### Prerequisites / 前置要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- [DeepSeek API Key](https://platform.deepseek.com/)（默认 provider）
- [E2B API Key](https://e2b.dev/)

### 1. Install / 安装

```bash
git clone https://github.com/yourname/auto-sdet.git
cd auto-sdet

uv sync
# 或
pip install -e .
```

### 2. Configure / 配置

```bash
cp .env.example .env
# 编辑 .env 填入 API Keys
```

```env
# 必填
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4-flash
E2B_API_KEY=e2b_xxx
MAX_RETRIES=3

# 可选 — LangSmith 可观测性
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxx
LANGCHAIN_PROJECT=auto-sdet
```

### 3. Run / 运行

```bash
# 内置示例
auto-sdet test examples/calculator.py

# 自己的模块
auto-sdet test src/mymodule.py --output-dir ./tests/

# 显示节点详细日志
auto-sdet test src/mymodule.py --verbose
```

---

## CLI Reference / 命令参考

```
Usage: auto-sdet test [OPTIONS] TARGET

Options:
  --max-retries INTEGER   最大自愈重试次数  [默认: 3]
  --model TEXT            LLM 模型名称
  --output-dir PATH       生成测试文件的输出目录
  --verbose               显示详细节点日志
  --help                  显示帮助
```

---

## Architecture / 架构详解

### Agent State / 状态定义

5 个节点通过共享 `AgentState`（TypedDict）通信，每个节点只返回它修改的字段，LangGraph 自动合并。

```python
class AgentState(TypedDict, total=False):
    source_code: str
    source_path: str
    context_files: dict[str, str]

    test_code: str

    # Evaluator 产物
    evaluation_result: EvaluationResult | None
    evaluator_reject_count: int

    # Reflector → MemoryManager 单向交接
    pending_trajectory: MemoryTrajectory | None

    # Executor 产物
    execution_result: ExecutionResult | None
    retry_count: int
    max_retries: int
    error_history: list[str]

    status: Literal["generating", "evaluating", "executing", "reflecting", "done", "failed"]
```

### Closed-loop quality gating / 闭环质量门禁

Generator 与 Reflector 都不能跳过 Evaluator 直达 Executor。这强制实现了"无受信任代码路径"的对称设计：Reflector 改完代码同样要被审视，避免修复反而引入新 bug 而被白白送去跑沙箱。

低于 0.5 分时，Evaluator 合成一段类 stderr 的弱点摘要塞进 `error_history`，路由直接打回 Reflector — 整套 self-healing 机器人复用，无需为评估失败单独造一条修复路径。

### Memory-R1 CRUD lifecycle / Memory-R1 生命周期

```
Reflector 产出 trajectory ──┐
                            ↓
                    MemoryManager 检索 top-3 邻居（含已 deprecated 的）
                            ↓
                    LLM 输出 MemoryOperation { ADD/UPDATE/DELETE/NOOP, target_id?, confidence }
                            ↓
                    安全护栏:
                      • DELETE 置信度 < high → 降级 NOOP
                      • UPDATE/DELETE 目标 ID 不在邻居集合 → 降级 NOOP（防幻觉）
                      • Manager 调用失败 → 默认 ADD（保守保留数据）
                            ↓
                    应用到持久化向量存储
```

下次类似失败时，Reflector 检索 `top_k=3` 邻居（仅 live 不含 deprecated）作为提示注入。

### Sandbox isolation + AST dep resolution / 沙箱隔离 + AST 依赖解析

Executor 用 Python 的 `ast` 模块静态解析源文件的 import 语句，自动定位项目内部依赖，扁平化上传至 E2B 微型虚拟机的 `/workspace/`，并重写 import 适配扁平结构。每次 pytest 运行后销毁沙箱，无状态泄漏。`pytest-cov` 内置启用，覆盖率从 stdout 解析后写回 `ExecutionResult`。

| 隔离方案 | 隔离强度 | 启动速度 | 复杂度 |
|---------|---------|---------|--------|
| **E2B micro-VM** | 最高 | ~300ms | 低 |
| Docker | 高 | 1–5s | 中 |
| subprocess | 无 | 即时 | 低 |
| AWS Lambda | 高 | 1–10s | 高 |

### LLM Tool Use (function calling) / 真实工具调用

Generator 不是固定 workflow。`read_file` / `list_directory` 用 `@tool` 装饰、通过 `bind_tools` 绑定到 LLM。LLM 在多轮 tool_call 循环里**自主决定**先看主文件还是先查目录、要不要递归读依赖。硬性迭代上限 + 缓存命中复用，防止失控调用与重复读文件。

```
LLM ─→ tool_call(read_file, target.py) ─→ ToolMessage(content)
    ─→ tool_call(list_directory, parent_dir) ─→ ToolMessage(content)
    ─→ tool_call(read_file, dep.py)         ─→ ToolMessage(content)
    ─→ FINAL: ```python ... ```             (no more tool_calls)
```

### Multi-provider LLM abstraction / 多模型 Provider 抽象

| Provider | 后端 | 备注 |
|---------|------|------|
| `deepseek` | DeepSeek API（默认） | thinking 兼容协议精细化：multi-turn bind_tools 关 thinking；single-turn json_mode 保留 thinking。由 `get_llm(multi_turn_tool_calling=...)` 区分 |
| `openai` | OpenAI API | — |
| `anthropic` | Anthropic Claude | 需 `pip install langchain-anthropic` |
| `ollama` | 本地 Ollama | 完全离线 |

切换 provider 只需改 `.env`，业务代码零改动。

---

## Benchmark Results / 基准测试结果

在 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) 抽取的 **19 个 Python 算法模块**上跑全量基准（涵盖排序、搜索、数学、字符串、数据结构）。架构演进过程跑了 7 次基准量化对比，最终（5 节点 + 4 项 ROI 优化 + ruff 双源信号）：

| 指标 | baseline (3 节点) | 5 节点 cold | 5 节点 + thinking | 优化迭代 | **5 节点 + 双源（最终）** |
|------|-------|---|----|----|-----------------|
| 一次通过率 / one-shot | 68.4% | 31.6% | 36.8% | 47.4% | **26.3%** |
| **最终通过率 / final** | 94.7% | 89.5% | 89.5% | 94.7% | **🌟 100%** |
| 平均自愈次数 / avg retries | 0.53 | 1.00 | 1.05 | 0.79 | **0.95** |
| 平均代码覆盖率 / avg coverage | 72.2% | 68.4% | 74.6% | 72.9% | **72.9%** ✓ |
| 单模块平均耗时 / avg wall time | 201.9s | 251.2s | 506.4s | 337.5s | **367.5s** |
| **失败文件数 / failed files** | 1 | 2 | 2 | 1 | **0** 🎉 |
| Memory 累积 / trajectories | n/a | 0→11 | 18→25 | 25→29 | **35→42** |
| Evaluator rejects | n/a | n/a | 1 | 0 | **1** (palindrome 触发) |

**主要结论：**
- 🌟 **最终通过率第一次达到 100%**（超过 baseline 94.7%，binary_search.py 等历来失败 case 全部通过）
- ✅ 覆盖率超 baseline +0.7pp，质量未牺牲
- ✅ ruff 双源信号验证有效（Evaluator 第一次出现非零 reject）
- ⚠️ 单模块耗时 +82%（5 节点 + thinking + Memory + ruff 的本质成本）
- 📊 一次通过率单 run 噪声大（5 次 run 范围 26-68%，二项分布 ±15pp 噪声），不应作为单 run 主指标

详细架构演进 case study 见 [`RESUME_IMPROVEMENT.md`](RESUME_IMPROVEMENT.md)。

本地复现：

```bash
git clone https://github.com/TheAlgorithms/Python ../algorithms-bench

# 全量
python scripts/benchmark.py

# 快速冒烟（前 3 个）
python scripts/benchmark.py --limit 3
```

每跑完一个模块就增量写 `benchmark_results/results.json`，中途崩溃可断点续跑。

---

## Dogfooding — Unit Tests / 自举测试

Auto-SDET 用**双层测试金字塔**测自己的源码：

1. **单元测试**（`tests/`）— 主要由 Auto-SDET 自身生成（router / schemas / prompts 等纯函数模块覆盖率 100%）
2. **端到端基准**（`scripts/benchmark.py`）— 跑 19 个真实算法模块的完整 Agent 链路

| 模块 | 覆盖率 | 来源 |
|------|--------|------|
| `graph/router.py` | 100% | 自动生成 |
| `models/schemas.py` | 100% | 自动生成 |
| `prompts/generator.py` | 100% | 自动生成 |
| `prompts/reflector.py` | 100% | 自动生成 |
| `tools/e2b_sandbox.py` (helpers) | 50% | 手写 |
| `graph/nodes/generator.py` (helpers) | 36% | 手写 |

剩余 LLM 节点 / 沙箱执行器 / CLI 由端到端基准验证 — mock 化反而比跑真实流程更脆弱。

```bash
pip install -e ".[dev]"
pytest tests/ --cov=src/auto_sdet --cov-report=term-missing
```

---

## Project Structure / 项目结构

```
auto-sdet/
├── pyproject.toml
├── .env.example
├── examples/
│   └── calculator.py
├── scripts/
│   └── benchmark.py
└── src/auto_sdet/
    ├── cli.py                       # Click CLI 入口
    ├── config.py                    # pydantic-settings 配置加载
    ├── graph/
    │   ├── graph.py                 # 5 节点 FSM 组装
    │   ├── router.py                # 条件边路由（after_executor + after_evaluator）
    │   └── nodes/
    │       ├── generator.py         # LLM Tool Use + 测试生成
    │       ├── evaluator.py         # LLM-as-Judge 4 维度质量评分
    │       ├── executor.py          # E2B 沙箱执行 + 覆盖率解析
    │       ├── reflector.py         # 结构化 CoT 反思 + 记忆检索
    │       └── memory_manager.py    # Memory-R1 CRUD 控制器
    ├── tools/
    │   ├── llm_factory.py           # 多 provider 抽象
    │   ├── mcp_context.py           # 文件上下文初始化
    │   ├── mcp_tools.py             # @tool 工具封装
    │   ├── e2b_sandbox.py           # 沙箱 + AST 依赖解析
    │   └── memory_store.py          # ChromaDB 向量记忆存储 + CRUD 接口
    ├── models/
    │   └── schemas.py               # Pydantic 模型 + AgentState
    └── prompts/
        ├── generator.py
        ├── evaluator.py
        ├── reflector.py
        └── memory_manager.py
```

---

## Tech Stack / 技术栈

| 组件 | 技术 | 选型原因 |
|------|------|---------|
| FSM 编排 | LangGraph | 原生 StateGraph + 条件边 + 状态合并 |
| LLM（默认） | DeepSeek V4 | 成本低、OpenAI 兼容、推理模式可选 |
| LLM（备选） | OpenAI / Claude / Ollama | 通过 `LLM_PROVIDER` 切换 |
| 质量门禁 | LLM-as-Judge + **ruff 双源信号** | LLM 评模糊语义 + ruff 评客观静态，互补 LLM 自审同质性盲区 |
| 情景记忆 | Memory-R1 CRUD + ChromaDB | 记忆库不再单调膨胀，跨任务复用修复经验 |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) | 本地、轻量、零外部 API 调用 |
| 沙箱 | E2B micro-VM | 进程/文件系统/网络全隔离，~300ms 启动 |
| 上下文 | MCP filesystem | 标准化 LLM 文件访问协议 |
| 覆盖率 | pytest-cov | 沙箱内行覆盖率 |
| 可观测性 | LangSmith | 节点级 DAG + token + 工具调用 trace |
| 类型/验证 | Pydantic v2 + TypedDict | BaseModel 工具 I/O，TypedDict LangGraph 状态 |
| 结构化输出 | `with_structured_output(method="json_mode")` | 强约束 schema + 兼容 DeepSeek thinking（response_format 协议路径） |
| CLI | Click + Rich | 装饰器风格 + 彩色输出 |
| 配置 | pydantic-settings + python-dotenv | 类型安全的 .env 加载 |
| 包管理 | uv | 比 pip 快 10–100x |

---

## Notes / 设计取舍备忘

部分尝试性架构（如 LangGraph `Send` 函数级并行生成）在权衡后被回退，详细的失败案例分析归档在 [`_deprecated/parallel_attempt/README.md`](_deprecated/parallel_attempt/README.md)，作为工程取舍的真实样本。