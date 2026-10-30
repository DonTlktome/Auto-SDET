# Auto-SDET

> **Autonomous Unit Test Generation Agent** powered by LangGraph FSM · LLM-as-Judge · Memory-R1 · E2B Sandbox
>
> **自主单元测试生成 Agent** · LangGraph 状态机 · LLM-as-Judge 质量门禁 · Memory-R1 情景记忆 · E2B 沙箱

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-5--node%20FSM-FF6F00)
![Memory-R1](https://img.shields.io/badge/Memory--R1-CRUD-7B1FA2)
![Tests](https://img.shields.io/badge/tests-249%20passing-43A047)
![Final Pass Rate](https://img.shields.io/badge/final%20pass%20rate-98.2%25%20(3%20runs)-43A047)
![Coverage](https://img.shields.io/badge/avg%20coverage-78.2%25-43A047)

---

## TL;DR

**Give it a Python source file → it generates pytest code and self-heals until the tests pass.** On 19 TheAlgorithms modules, 3 runs with `deepseek-v4-flash`: **98.2% final pass rate** (94.7–100%), 33.4% one-shot, 78.2% average coverage, ~61 s per module. Every number below links to a result file in [`benchmark_results/flash/`](benchmark_results/flash/).

**给一个 Python 源文件，生成 pytest 测试代码 + 自愈直到通过。** 19 个 TheAlgorithms 模块、`deepseek-v4-flash`、3 轮实测：最终通过率 **98.2%**（94.7–100%）、一次通过率 33.4%、平均覆盖率 78.2%、单模块约 61 秒。下文每个数字都对应 [`benchmark_results/flash/`](benchmark_results/flash/) 中的结果文件。情景记忆开 / 关的对照实验**没有测到显著收益**（见[记忆开 / 关对比](#记忆开--关对比)）。

| What's interesting | 中文 |
|---|---|
| 🧠 **5-node LangGraph FSM** (Generator → Evaluator → Executor → Reflector → MemoryManager) with dual circuit breakers for bounded termination | 五节点闭环状态机 + 双重熔断保证有界终止 |
| 🛡️ **Dual-source quality gate** — LLM-as-Judge (semantic) + ruff static analysis (deterministic), compensating for the same-model self-evaluation blind spot | 双源 quality gate — LLM 评模糊语义 + ruff 评客观静态，补完大模型同源自审盲区 |
| 💾 **Memory-R1 (Lu 2025) episodic memory** — LLM-driven CRUD operations (ADD/UPDATE/DELETE/NOOP) over ChromaDB, with multi-layer safety guards preventing LLM mis-deletion | Memory-R1 跨任务情景记忆 — LLM 控制器做 CRUD 操作 + 多层防误删护栏 |
| 🔒 **E2B Micro-VM sandbox** with AST-based dependency resolution — fresh isolated VM per run (measured create ≈ 0.7 s); sandbox infra failures retried and kept out of the agent's retry budget | E2B 微型虚拟机沙箱 + AST 自动依赖解析；基础设施故障自动重试、不占 Agent 重试次数 |
| 🔬 **Controlled experiments with self-describing result files** — memory on / off / fixed (3 runs each, paired per-module tests), time breakdown, held-out threshold calibration, plus 4 earlier engineering case studies | 带元数据结果文件的对照实验：记忆开 / 关 / 修复版各 3 轮 + 按模块配对检验 + 耗时拆账 + 留出集阈值标定 |

📖 **Engineering deep-dive:** [`RESUME_IMPROVEMENT.md`](RESUME_IMPROVEMENT.md) · **Code walkthrough:** [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md)

---

## Demo / 运行示例

```
auto-sdet test src/calculator.py
```

示意输出（节选、简化；真实运行日志见 `benchmark_results/flash/runs/*/console.log`）：

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

⚡ [Executor]     All tests passed!  Coverage: 88%      ← 重试路径不再经过 Evaluator（非对称门禁）

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

> **实测结论（如实）：** 在 19 模块规模上，记忆开 / 关对最终通过率、重试次数、覆盖率都**没有可测到的差异**（3 轮配对检验 p ≥ 0.29）。原检索（配置 A，默认）用 pytest 输出前 300 字符作 query、却用"分类 + 根因"作入库文档，二者不在同一表示空间 —— 在留出标注集上其检索精度（0.106）等于随机基础率（0.114）。修复版（配置 C，`MEMORY_VARIANT=signature`：两侧统一为归一化错误签名 + 余弦阈值标定 + 验证后写入）精度提升到 0.276，并检出了真实的跨模块命中，但同类失败在无记忆时同样一次修好，最终指标仍无差异。详见 [记忆开 / 关对比](#记忆开--关对比)。

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

诊断字段升级为一等公民数据，CLI 实时展示，LangSmith trace 完整可见。

**结构化输出失败的处理：** thinking 可能把 `max_tokens` 预算耗尽，导致 JSON 被截断。此时用同一 prompt、关闭 thinking 再请求一次；仍失败则记为一次失败的修复尝试，按剩余次数继续重试（早期版本会直接把 `retry_count` 打满、放弃剩余重试 —— 修复前的 pro 单轮数据中 3 个失败模块都只尝试了 1 次）。flash 3 组 × 3 轮实测：Reflector 截断 9 次、其他解析失败 2 次，11 次全部被关闭 thinking 的重新请求恢复，未产出修复的尝试为 0（[`2026-10-30_summary.md`](benchmark_results/flash/2026-10-30_summary.md)）。

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

### Asymmetric quality gating / 非对称质量门禁

Generator 的首次产出必须经过 Evaluator；Reflector 的修复则经 MemoryManager 直达 Executor，不再过 Evaluator（`graph.py`）。理由：重试路径上 pytest 本身就是真值，软评分在此处的打回率很低却每次都要付出一次 LLM 调用。早期版本曾是对称设计（两条路径都过 Evaluator），后基于 benchmark 改为非对称。

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

**沙箱内只装测试真正需要的包：** `pytest`、`pytest-cov`，加上同一次 AST 解析出的被测模块第三方依赖（排除标准库、相对导入和项目内模块，`cv2`→`opencv-python` 这类导入名与包名不一致的情况做映射）。Agent 自身的依赖（langgraph / langchain 等）不进沙箱 —— 早期版本每次都装这一串，实测 pip 步骤 7.9s → 2.8s（[`2026-10-30_pip_install_old_vs_new.json`](benchmark_results/2026-10-30_pip_install_old_vs_new.json)）。

> 生产部署应使用预装 pytest / pytest-cov 的自定义 E2B 沙箱模板，把 pip 步骤彻底去掉；本仓库的基准实验仍按需安装，保证用默认 code-interpreter 模板即可复现。

**基础设施故障与测试失败分离：** 沙箱创建失败、上传失败、pip 失败、pytest 运行中沙箱死亡等都归为基础设施故障，在 Executor 内换新沙箱自动重试（最多 3 次），**不计入 `retry_count`，也不交给 Reflector**；三次都失败则模块以 `infra_failed` 结束，与"测试没修好"分开统计。测试本身挂起超时（死循环、无限递归）仍算测试失败，交给 Reflector。

实测（flash 9 轮，单模块平均，含重试的全部沙箱执行）：沙箱创建 0.73 s、pip 4.1–4.2 s、pytest 2.8–3.6 s —— 沙箱侧合计不到单模块耗时的 15%，主要时间在 LLM 调用（见[耗时拆账](#耗时拆账)）。

| 隔离方案 | 隔离强度 | 复杂度 | 备注 |
|---------|---------|--------|------|
| **E2B micro-VM** | 最高 | 低 | 本项目实测创建 ≈ 0.7 s / 次 |
| Docker | 高 | 中 | 需要本机 / 集群容器运行时 |
| subprocess | 无 | 低 | 生成的代码直接跑在宿主机上 |

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

**设置：** [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)（commit `791deb40`）中的 19 个模块（列表见 `scripts/benchmark.py` 的 `BENCH_MODULES`，顺序固定）；模型 **`deepseek-v4-flash`**（Generator 关 thinking，Evaluator / Reflector / MemoryManager 开 thinking）；`max_retries=3`；每组配置 3 轮，每轮记忆库从空目录开始；同一轮的 A_i / B_i / C_i 并发运行（同样的 API 负载下配对）。每个结果文件都带 git commit、模型、各开关、模块顺序、起止时间等元数据；每轮验收前用 `scripts/check_api_errors.py` 扫描 API 错误（9 轮全部 CLEAN）。

| 配置 | 说明 | commit |
|---|---|---|
| **A** | 当前默认架构：5 节点 + ruff 双源门禁 + 非对称路由 + Memory-R1 记忆（原检索） | `275ff17` |
| **B** | 同 A，关闭记忆（不检索、不写入） | `275ff17` |
| **C** | A + 记忆修复（统一错误签名、余弦阈值 0.40、pytest 重跑后写入 outcome） | `80f9cd9` |

均值（最小–最大），n = 3 轮 × 19 模块：

| 指标 | A | B | C |
|---|---|---|---|
| 一次通过率 | 33.4% (21.1–47.4) | 26.3% (15.8–36.8) | 26.3% (15.8–31.6) |
| **最终通过率** | **98.2%** (94.7–100) | 100% (100–100) | 100% (100–100) |
| 平均重试次数 | 0.74 (0.58–0.84) | 0.84 (0.79–0.95) | 0.88 (0.74–1.11) |
| 平均覆盖率¹ | 78.2% (77.8–78.9) | 78.3% (76.8–80.2) | 78.3% (76.8–79.2) |
| 单模块耗时 | 60.8 s (57.4–63.9) | 66.3 s (63.8–69.6) | 70.6 s (61.9–76.1) |
| Reflector 修复尝试 / 轮 | 14 (11–16) | 16 (15–18) | 16.7 (14–21) |
| Reflector 截断 + 解析失败 / 轮（均被恢复） | 1.3 (0–3) | 1.0 (0–2) | 1.3 (0–2) |
| 基础设施故障 / 轮（已自动重试） | 0 | 0 | 0.3 (0–1) |
| LLM 费用估算 / 轮² | $0.08 | $0.09 | $0.09 |

¹ 只统计通过且测得覆盖率的模块：170 个通过的模块中有 18 个覆盖率为空 —— 生成的测试用 importlib 以别名加载被测模块，`--cov=<模块名>` 统计不到。
² token 数实测；单价为第三方报价假设（$0.14 / $0.28 每百万 token），未经账单核对。

9 轮中唯一的最终失败：A_r1 的 `searches/binary_search.py`，3 次真实修复尝试后仍失败。

来源：[`2026-10-30_summary.md`](benchmark_results/flash/2026-10-30_summary.md)（由 `scripts/summarize_benchmarks.py` 从 9 个 `benchmark_results/flash/2026-10-30_{A,B,C}_r{1,2,3}.json` 生成）。

### 耗时拆账

单模块平均（秒），包含该模块所有重试中的全部沙箱执行与 LLM 调用：

| 段 | A | B | C |
|---|---|---|---|
| 沙箱创建 | 0.73 | 0.73 | 0.73 |
| pip 安装 | 4.1 | 4.2 | 4.2 |
| pytest 执行 | 3.6 | 2.8 | 2.8 |
| LLM 调用（关键路径） | 48.6 | 55.2 | 58.1 |
| 模块结束时等待记忆写入 | 0.0 | 0.0 | 2.2 |
| 其他（图调度 / 嵌入 / 上传） | 3.7 | 3.5 | 2.5 |
| **合计（单模块耗时）** | **60.8** | **66.3** | **70.6** |
| 参考：LLM 调用含后台 MemoryManager（与其他步骤并行，不计入合计） | 50.2 | 55.2 | 58.1 |

LLM 调用占单模块耗时的 80–83%，沙箱侧（创建 + pip + pytest）合计 7.7–8.4 s，约 11–14%。沙箱只装 pytest / pytest-cov 和被测模块的第三方依赖后，pip 单次耗时由 7.9 s 降到 2.8 s（[`2026-10-30_pip_install_old_vs_new.json`](benchmark_results/2026-10-30_pip_install_old_vs_new.json)，各 3 次独立测量）；上表的 pip 时间是每个模块 1.6–2.1 次沙箱执行的累计。

### 记忆开 / 关对比

按 (模块, 轮次) 配对，57 对（`summarize_benchmarks.py`，精确检验）：

| 对比 | 最终通过（McNemar） | 一次通过（McNemar，噪声基线³） | 每模块平均重试（符号检验） |
|---|---|---|---|
| A vs B | 仅 A 通过 0 / 仅 B 通过 1，p = 1.0 | 9 / 5，p = 0.42 | A 更多 2 / B 更多 6，p = 0.29 |
| C vs A | 1 / 0，p = 1.0 | 5 / 9，p = 0.42 | C 更多 6 / A 更多 2，p = 0.29 |
| C vs B | 0 / 0，p = 1.0 | 6 / 6，p = 1.0 | 4 / 4，p = 1.0 |

³ Generator 不读记忆，一次通过率不可能受记忆影响 —— 它的组间差异就是纯运行噪声，幅度与受记忆影响的指标相当。

**结论：在 19 个模块的规模上，记忆没有可测到的收益。** 原因可以拆开看：

1. **原检索（A）近乎随机。** query 是 pytest 输出前 300 字符（基本全是会话表头）加文件名，入库文档却是"分类：根因首句"。在留出标注集上，它的检索精度 0.106 等于随机基础率 0.114。
2. **修复版检索（C）更准但仍有限。** 两侧统一为归一化错误签名后，精度 0.276（召回 0.871）；但把数字归一化为 `<n>` 也带来误检 —— 任何"列表不相等"断言的签名几乎相同（出现过相似度 0.94 的不相关命中）。
3. **有真实的跨模块命中，但没有改变结果。** 人工逐案核对"跨模块命中且随后修好"的候选：A 34 个中 4 个相关 + 2 个部分相关，C 32 个中 8 个相关 + 1 个部分相关，且全部属于同一类失败 —— 生成的测试对模块位置做了错误假设（写死宿主机 Windows 路径、用 `sorts.xxx` 包路径做 import / patch / runpy）。同类失败在不开记忆的 B 中下一次运行修好的比例是 73%（11/15），与 A 73%、C 75% 相同。最佳案例（C_r1 `singly_linked_list.py` 命中 `stack.py` 的"写死 Windows 路径"记忆，余弦 0.811，修复照着记忆的做法改）的完整证据见 [`2026-10-30_memory_case_review.md`](benchmark_results/flash/2026-10-30_memory_case_review.md)。
4. **天花板效应。** 无记忆的 B 最终通过率已是 100%，记忆只可能影响重试次数，而 57 对样本不足以检出这么小的差异。

### 阈值标定（配置 C）

在 12 个与基准**不重叠**的留出模块上用配置 B 跑 3 轮，收集 25 个带诊断的失败样本，按事先写定的规则人工标注失败机制（同标签的跨模块样本对为"相关"，共 273 对、31 对相关），扫描余弦阈值并取 F1 最大值：**阈值 0.40，精度 0.276、召回 0.871、F1 0.419**（F1 曲线平坦：0.55 处 0.414）。同一批样本按原检索方式评估：精度 0.106、召回 0.871。过程、标注规则与数据见 [`benchmark_results/flash/calibration/`](benchmark_results/flash/calibration/README.md)。

### 其他数据点（不与上表合并）

- **修复前的 pro 单轮**（`deepseek-v4-pro`，commit `d165eb7`，记忆开 / 关各 1 轮）：[`benchmark_results/2026-10-30_A_r1.json`](benchmark_results/2026-10-30_A_r1.json) / [`B_r1`](benchmark_results/2026-10-30_B_r1.json)。最终通过 89.5% / 94.7%；3 个失败模块都是 Reflector 的 thinking 耗尽 `max_tokens` 后 `retry_count` 被直接打满、实际只尝试了 1 次修复 —— 这正是之后修复的问题。
- **作废的一轮**（DeepSeek 402 余额不足）：[`benchmark_results/flash/invalid/`](benchmark_results/flash/invalid/2026-10-30_round1_402/README.md)，留作证据，不参与统计。

### 历史记录（早期单轮结果，仅作演进参考）

早期架构演进中的单轮结果，文件已补交入库（内容未改动），但**不能与上表直接比较**：每列只有 1 轮；未记录模型与 commit；除 baseline / cold 外记忆库都带着之前几轮在同样 19 个模块上积累的轨迹运行（"最终"列开始时库中已有 35 条），存在同模块泄漏；`results_5node_coldstart.json` 与 `results_5node_memoryR1.json` 逐字节相同；`results.json` 是一次 5 模块运行。

| 指标 | baseline (3 节点) | 5 节点 cold | 5 节点 + thinking | 优化迭代 | 5 节点 + 双源 |
|------|-------|---|----|----|-----|
| 一次通过率 | 68.4% | 31.6% | 36.8% | 47.4% | 26.3% |
| 最终通过率 | 94.7% | 89.5% | 89.5% | 94.7% | 100% |
| 平均自愈次数 | 0.53 | 1.00 | 1.05 | 0.79 | 0.95 |
| 平均覆盖率 | 72.2% | 68.4% | 74.6% | 72.9% | 72.9% |
| 单模块平均耗时 | 201.9s | 251.2s | 506.4s | 337.5s | 367.5s |
| 结果文件 | `results_baseline.json` | `results_5node_coldstart.json` | `results_5node_thinking_warmstart.json` | `results_5node_optimized.json` | `results_5node_dualsource.json` |

详细架构演进 case study 见 [`RESUME_IMPROVEMENT.md`](RESUME_IMPROVEMENT.md)。

### 本地复现

```bash
git clone https://github.com/TheAlgorithms/Python ../algorithms-bench   # 实验用 commit 791deb40

# 一轮（结果写 benchmark_results/<日期>_<配置>_r<n>.json，不覆盖已有文件；--resume 断点续跑）
DEEPSEEK_MODEL=deepseek-v4-flash python scripts/benchmark.py --config A --run-index 1
DEEPSEEK_MODEL=deepseek-v4-flash python scripts/benchmark.py --config B --run-index 1
DEEPSEEK_MODEL=deepseek-v4-flash python scripts/benchmark.py --config C --run-index 1 \
    --cosine-threshold 0.40 --calibration-file benchmark_results/flash/calibration/threshold.json

# 验收与汇总
python scripts/check_api_errors.py benchmark_results/runs/<run>        # 扫描 402/429/5xx 等 API 错误
python scripts/summarize_benchmarks.py --results-dir benchmark_results  # 均值 / 最小 / 最大 + 配对检验
python scripts/analyze_memory_trace.py benchmark_results/runs/<run>     # 跨模块记忆命中
```

正式实验的 A / B 运行在 commit `275ff17`，C 运行在 `80f9cd9`（`git diff 275ff17 80f9cd9` 只含记忆相关代码与标定数据）。

---

## Dogfooding — Unit Tests / 自举测试

Auto-SDET 用**双层测试金字塔**测自己的源码：

1. **单元测试**（`tests/`，249 个）— 纯函数模块的测试最初由 Auto-SDET 自身生成，之后的改动配套手写测试
2. **端到端基准**（`scripts/benchmark.py`）— 跑 19 个真实算法模块的完整 Agent 链路

当前覆盖率（`pytest tests/ --cov=src/auto_sdet` 实测，总计 62%）：

| 模块 | 覆盖率 | 来源 |
|------|--------|------|
| `models/schemas.py` | 100% | 自动生成 + 手写补充 |
| `prompts/generator.py` | 100% | 自动生成 |
| `tools/error_signature.py` | 97% | 手写 |
| `tools/run_trace.py` | 94% | 手写 |
| `tools/llm_usage.py` | 93% | 手写 |
| `graph/router.py` | 85% | 自动生成 + 手写补充 |
| `prompts/reflector.py` | 82% | 自动生成 |
| `tools/memory_store.py` | 61% | 手写 |
| `tools/e2b_sandbox.py` | 54% | 手写（依赖解析 / infra 重试；真实沙箱路径由基准覆盖） |
| `graph/nodes/generator.py` | 43% | 手写（代码提取 / DSML 解包） |

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
│   ├── benchmark.py                 # 一轮基准：--config A|B|C，自描述结果文件
│   ├── summarize_benchmarks.py      # 多轮汇总 + 按模块配对检验
│   ├── analyze_memory_trace.py      # 跨模块记忆命中分析
│   ├── memory_counterfactual.py     # 同类失败有 / 无记忆的修复率对比
│   ├── calibrate_memory_threshold.py# 留出集标定余弦阈值
│   ├── check_api_errors.py          # 每轮验收前扫描 LLM API 错误
│   └── measure_pip_install.py       # 沙箱 pip 耗时对比
├── benchmark_results/
│   ├── flash/                       # 正式实验（deepseek-v4-flash）结果、trace、标定
│   └── *.json                       # 修复前 pro 单轮 + 早期历史结果
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
    │   ├── e2b_sandbox.py           # 沙箱 + AST 依赖解析 + infra 故障重试 + 耗时拆账
    │   ├── ruff_checker.py          # Evaluator 的 ruff 静态分析信号
    │   ├── memory_store.py          # ChromaDB 向量记忆存储 + CRUD 接口（l2 / cosine）
    │   ├── error_signature.py       # 归一化错误签名（signature 记忆变体）
    │   ├── llm_usage.py             # LLM 耗时 / token 统计 callback
    │   └── run_trace.py             # 实验事件 trace（JSONL）
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
| 沙箱 | E2B micro-VM | 进程/文件系统/网络全隔离，实测创建 ≈ 0.7 s |
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