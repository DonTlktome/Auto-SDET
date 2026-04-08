# Auto-SDET
Autonomous Unit Test Generation Agent powered by LangGraph FSM + E2B Sandbox + DeepSeek-V3

## 1. 项目概述

**Auto-SDET**（Autonomous Software Development Engineer for Testing）是一个**自主测试用例生成 Agent**，能够接收 Python 源代码文件，自动生成高质量的 pytest 测试套件，并在隔离沙箱中运行验证，失败时通过 Chain-of-Thought 推理自我修复，直至测试通过或达到最大重试次数。

### 核心亮点

| 特性 | 描述 |
|------|------|
| **有限状态机架构** | 基于 LangGraph 的确定性 FSM，每条路径可预测、可追溯 |
| **自愈循环** | 测试失败后，Reflector 节点使用 CoT + 错误历史自动修复 |
| **安全沙箱执行** | E2B 微型虚拟机提供进程/文件系统/网络全隔离 |
| **MCP 文件上下文** | 使用 Model Context Protocol 标准化文件读取 |
| **类型安全** | Pydantic v2 + TypedDict 双轨类型系统 |

---

## 2. 项目结构

```
Auto-SDET/
├── pyproject.toml                    # 项目配置（uv/pip、依赖、入口点）
├── .env.example                      # 环境变量模板
├── README.md                         # 项目说明
├── auto-sdet-architecture.docx       # 架构图（Word 文档）
├── auto-sdet-interview-guide.docx    # 面试准备指南
│
├── examples/
│   └── calculator.py                 # 示例目标文件（数学运算函数 + 类）
│
└── src/auto_sdet/
    ├── __init__.py                   # 包根，版本 0.1.0
    ├── cli.py                        # Click CLI 入口（auto-sdet test ...）
    ├── config.py                     # Pydantic-settings 配置加载（.env）
    │
    ├── models/
    │   └── schemas.py                # Pydantic 模型 + AgentState TypedDict
    │
    ├── graph/
    │   ├── graph.py                  # LangGraph StateGraph 组装 & run_agent()
    │   ├── router.py                 # 条件边路由逻辑
    │   └── nodes/
    │       ├── generator.py          # LLM 测试代码生成节点
    │       ├── executor.py           # E2B 沙箱测试执行节点
    │       └── reflector.py          # CoT 错误分析与修复节点
    │
    ├── tools/
    │   ├── mcp_context.py            # MCP 文件系统上下文（读取文件）
    │   └── e2b_sandbox.py            # E2B 沙箱包装器（执行测试）
    │
    └── prompts/
        ├── generator.py              # 生成阶段 System + User Prompt 模板
        └── reflector.py              # 修复阶段 CoT Prompt 模板
```

---

## 3. 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| **FSM 编排** | LangGraph 0.2+ | 带条件边路由的状态机 |
| **LLM** | DeepSeek-V3（OpenAI 兼容 API） | 代码生成（成本优化选型） |
| **沙箱** | E2B（微型虚拟机） | 隔离测试执行（~300ms 启动） |
| **上下文协议** | MCP（Model Context Protocol） | 标准化 LLM 文件访问 |
| **类型安全** | Pydantic v2 | 模型验证 + Schema 定义 |
| **CLI** | Click 8.1+ | 命令行接口 |
| **终端 UI** | Rich 13+ | 彩色格式化状态输出 |
| **配置管理** | pydantic-settings | 带类型安全的 .env 加载 |
| **包管理** | uv（推荐） | 比 pip 快 10-100x |

---

## 4. 核心架构：有限状态机（FSM）

### 4.1 工作流程图

```
 ┌─────────────────────────────────────────┐
 │              用户输入                    │
 │   auto-sdet test calculator.py          │
 └──────────────────┬──────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Generator     │  读取源码 → 调用 DeepSeek → 生成测试代码
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Executor      │  创建 E2B 沙箱 → 运行 pytest → 捕获结果
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
       通过？            失败？
     exit_code=0       exit_code≠0
          │                 │
          │                 ▼
          │        retry < max_retries?
          │           是 ↓      否 ↓
          │       ┌──────────┐  ┌────────┐
          │       │ Reflector│  │  失败  │
          │       │ CoT 修复 │  │  终止  │
          │       └────┬─────┘  └────────┘
          │            │ 回到 Executor
          ▼            ▼
      ┌──────────────────┐
      │   保存测试文件    │
      │   输出结果报告   │
      └──────────────────┘
```

### 4.2 FSM 状态定义

```python
# AgentState TypedDict（LangGraph 共享状态）
{
    # 输入上下文
    "source_path": str,
    "source_code": str,
    "context_files": dict[str, str],

    # 生成的测试
    "test_code": str,

    # 执行反馈
    "execution_result": ExecutionResult | None,

    # 循环控制
    "retry_count": int,
    "max_retries": int,
    "error_history": list[str],

    # 状态追踪
    "status": str,  # "generating" | "executing" | "done" | "failed"
}
```

### 4.3 路由决策逻辑

```
route_after_executor(state) →
  IF exit_code == 0      → "end_success"
  IF retry < max_retries → "reflect"
  ELSE                   → "end_failed"
```

---

## 5. 核心组件详解

### 5.1 Generator 节点（`graph/nodes/generator.py`）

**职责：** 将源代码转化为 pytest 测试文件

**执行流程：**
1. 调用 `MCPContextManager.gather_context()` 读取目标文件及同级依赖（最多 5 个）
2. 使用 XML 标签构建结构化 Prompt（`<source_file>`、`<context_files>`）
3. 调用 DeepSeek API（`ChatOpenAI` 兼容接口）
4. 正则提取 markdown 代码块中的 Python 代码（多重 fallback 策略）
5. 返回更新后的状态

**Prompt 规则强制执行：**
- 每个公共函数必须有正例 + 边界条件测试
- 使用 `@pytest.mark.parametrize` 处理多输入
- 使用 `unittest.mock.patch` 处理外部依赖
- 禁止网络访问、禁止非标准库 import

### 5.2 Executor 节点（`graph/nodes/executor.py`）

**职责：** 在隔离环境中执行测试，收集结果

**执行流程：**
1. 创建 `SandboxExecutor` 实例
2. 调用 `execute_test(source_path, source_code, test_code, context_files)`
3. 将 stderr 追加到 `error_history`（累积而非覆盖，防止遗忘）
4. 返回 `ExecutionResult` 及更新的错误历史

> **设计原则：** Executor 只负责执行，不做路由决策（单一职责原则）

### 5.3 Reflector 节点（`graph/nodes/reflector.py`）

**职责：** 使用 CoT 推理分析失败原因并修复测试代码

**CoT 四步推理：**
```
Step 1: 错误分类（ImportError / AssertionError / SyntaxError）
Step 2: 根因分析（定位到具体行号）
Step 3: 最小修复策略（只修 bug，不删测试）
Step 4: 输出完整修复文件
```

**防振荡机制：** 将完整 `error_history` 注入 Prompt，避免 LLM 在 A→B→A 路径上循环

### 5.4 E2B 沙箱工具（`tools/e2b_sandbox.py`）

**执行步骤：**
1. 创建 E2B Sandbox（新实例，~300ms）
2. 写入源码 + 测试文件 + 依赖到 `/workspace/`
3. 执行 `pip install pytest`
4. 执行 `pytest test_*.py -v`（带超时控制）
5. 捕获 stdout/stderr/exit_code/duration
6. 销毁沙箱（`finally` 块保证资源释放）

**为什么选 E2B：**

| 方案 | 隔离性 | 启动速度 | 安全性 | 复杂度 |
|------|--------|---------|--------|--------|
| E2B 微型 VM | 最高 | ~300ms | 高 | 低 |
| Docker | 高 | 1-5s | 中 | 中 |
| subprocess | 无 | 即时 | 低 | 低 |
| AWS Lambda | 高 | 1-10s | 高 | 高 |

### 5.5 MCP 上下文工具（`tools/mcp_context.py`）

- `read_file(path)` → `FileContent`
- `list_directory(path, pattern)` → 文件路径列表
- `gather_context(target_path)` → 源码 + 上下文文件字典
- **Fallback 策略：** MCP 服务不可用时，直接使用 `Path.read_text()` 降级
- **过滤规则：** 跳过 `__init__.py`、`test_*.py`、目标文件本身
- **Token 预算控制：** 限制最多 5 个同级文件

---

## 6. 数据流转示例

以 `examples/calculator.py` 为例：

```
初始状态: source_path="calculator.py", retry_count=0

↓ Generator 节点
  source_code: "class Calculator: ..."
  test_code: "def test_add(): assert add(1,2)==3 ..."

↓ Executor 节点
  exit_code=1, stderr="ImportError: cannot import..."
  error_history: ["ImportError: cannot import..."]

↓ Router → reflect（retry_count=0 < max_retries=3）

↓ Reflector 节点
  test_code（修复后）: "from calculator import add\ndef test_add()..."
  retry_count: 1

↓ Executor 节点
  exit_code=0, stdout="5 passed in 0.42s"

↓ Router → end_success

最终状态: status="done", test_calculator.py 已保存
```

---

## 7. 安装与使用

### 环境变量（`.env`）

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
E2B_API_KEY=e2b_xxx
E2B_SANDBOX_TIMEOUT=60
MAX_RETRIES=3
```

### 安装

```bash
# 推荐：使用 uv
uv sync

# 或使用 pip
pip install -e .
```

### 使用

```bash
# 基本用法
auto-sdet test examples/calculator.py

# 高级选项
auto-sdet test src/mymodule.py --max-retries 5 --model deepseek-chat --output-dir ./tests/
```
