# OpBench v0.7 Dataset Card

Release：`opbench-v0.7.0`

状态：Released（50/50 fresh replay；122/122 valid Agent Attempts）

日期：2026-08-11

## 1. 发布概览

OpBench v0.7 是面向真实 PyTorch 算子修复的 verified Dataset release。正式发布
由 **50 verified Tasks** 组成：

```text
14 retained historical
+ 21 new
+ 15 replacement
= 50 Tasks
```

所有任务均通过真实 Runtime Admission、Prompt 泄漏审查、复杂度审查以及
Task/Source/Environment/Gold/Hidden Test 内容身份闭合。最终集合没有 easy Task：
46 条为 hard，4 条为经 blind pilot 与第二 reviewer 确认的 medium。

Boundary、Precision、Device 不再是互斥顶层分类，而是从统一四轴 taxonomy
派生的可重叠分析切片：

| Dataset / Slice | Task 数 | Dataset hash |
| --- | ---: | --- |
| `pytorch_v0.7` | 50 | `sha256:3695622dd2619a760d510ef49e0a9dbff637c98790ad3263c521bae8e99c9518` |
| `pytorch_v0.7_boundary` | 31 | `sha256:2890f5937a5b2c7f5a12c870fc9cc550f0f16ff065467245ecf65223b5976a01` |
| `pytorch_v0.7_precision` | 5 | `sha256:508ec6928d94c159499ae84bf4f37e594b2bdafdef89b04369f481deeddb2c8d` |
| `pytorch_v0.7_device` | 15 | `sha256:b598fdfe94af9921132b147ab693477de8fb360dabe7e5f611792e5f38c0f138` |

内容寻址 release manifest 位于
[`factory/v0.7/p9/release_manifest.json`](../../factory/v0.7/p9/release_manifest.json)。
2026-07-28 发布的 25/6/8 历史冻结保持原字节和原 Hash，存放于
[`archives/v0.7-pre-quality/`](../../archives/v0.7-pre-quality/)，但不再代表当前
50-task 完成条件。

## 2. 任务来源与准入

任务来自真实 PyTorch issue/PR 修复，不是合成 bug。旧 25 条任务经过重新准入后，
14 条保留、10 条淘汰、1 条 deferred；36 条新增或替换任务由当前质量漏斗重新制作，
不是为了凑数恢复旧成员。

每条正式任务至少满足：

1. Base Commit、Source Snapshot、Runtime、Task Manifest 和 Patch Scope 可解析；
2. Baseline 的 F2P 真实失败且 P2P 通过；
3. Gold 后 F2P/P2P 均通过，selector 被实际 collected/executed；
4. Hidden Test 使用行为断言，而非只搜索实现文本或私有符号；
5. AgentTaskView 只描述可观测问题，不包含答案路径、内部符号、修复步骤、PR 或 Commit；
6. Prompt 自动扫描、盲审与私有语义复审均绑定最终内容 Hash；
7. complexity、duplicate、Admission 和 replay evidence 全部闭合。

静态 `git apply --check`、语法检查和上游测试只作为草案门禁，不能替代真实
Runtime Admission。正式 50 条均为 `admission_status=verified`。

## 3. Taxonomy

v0.7 使用四个主要轴，避免把精度、边界和设备混成相互排斥的问题类型：

- `contract_family`：被破坏的行为合同；
- `failure_type`：外部可见的失败形式；
- `execution_context`：device、mode、phase 与 distributed；
- `trigger_tags`：触发条件，仅在有事实证据时填写。

### Contract family

| Family | Task 数 |
| --- | ---: |
| API behavior | 27 |
| Efficiency / safety | 6 |
| Result | 6 |
| Mutation / state | 5 |
| Tensor metadata | 4 |
| Gradient | 2 |

### Failure type

| Failure type | Task 数 |
| --- | ---: |
| Unexpected error | 31 |
| Wrong result | 12 |
| Missing error | 5 |
| Performance regression | 2 |

### Execution context

| 轴 | 分布 |
| --- | --- |
| Device | CPU 35；CUDA 15 |
| Mode | compile 35；eager 21 |
| Phase | forward 49；backward 4 |
| Runtime tier | CPU overlay 34；CPU source 1；CUDA overlay 13；CUDA kernel build 2 |

同一任务可同时覆盖 eager/compile 或 forward/backward，因此这些计数不要求加总为
50。gradient family 为 2 条；backward phase 与 gradient family 的并集为 5 条，
低于早期 6 条搜索目标。当前集合也没有 distributed Task。这两项都是明确发布的
覆盖缺口，不通过错误标注、简单任务或降低 Admission 标准填表。

## 4. 派生切片

- Boundary 31 条：从 empty/scalar/extreme/endpoint/layout/dynamic-shape 等触发条件
  推导，覆盖 CPU 与 CUDA，也覆盖结果、API、元数据、梯度和安全合同；
- Precision 5 条：由 mixed dtype、numerical stability/overflow 等证据推导；
- Device 15 条：所有需要 CUDA 行为或 CUDA 特有执行路径的任务。

切片允许重叠，也允许任务只属于 cumulative。它们用于检索和分层分析，不是三个
独立采样总体；仅在样本数足够时才报告独立评分，避免小桶产生虚假精确度。

## 5. Runtime 与 Source

50 条任务都绑定精确 Environment 与 Source Registry 身份。Python overlay 任务
使用与目标源码兼容的冻结 wheel/image；source-build 任务从 authoritative
workspace 构建；CUDA kernel 任务冻结 toolchain、架构、build flags 和超时。

Evaluator 每次从 Agent 最终 patch 创建 fresh workspace，再运行 evaluator-only
selector。Runtime 准备失败、远端 transport 失败或资源清理失败归为
infrastructure-invalid，不伪装成 Agent 的逻辑失败，也不进入有效结果分母。

## 6. Agent 验证合同

最终质量验证冻结为：

```text
36 new/replacement × 3 repeats
+ 14 retained historical × 1 repeat
= 122 fresh logical Attempts
```

合同绑定同一 Dataset release、AgentTaskView、Prompt renderer、Agent/Model、Runtime
Profile、Capability、Budget、Evaluator、Retry、Termination 与 Scoring 身份。正式
结果由 append-only ledger 选择有效 logical Attempt，并要求 Trace、Integrity 与
Attempt-owned resource cleanup 全部通过。

最终运行状态和聚合结果见
[`validation_report.md`](validation_report.md)。完整 Agent 自然语言输出、隐藏思考
过程以及用 reasoning trace 参与评分不属于 v0.7 合同。

## 7. 泄漏与污染风险

这些任务来自公开 PyTorch 历史，相关 issue、PR、代码和测试可能已进入模型训练
语料，因此本数据集不声称训练数据无污染。平台降低评测内泄漏的措施包括：

- Agent 只接收经扫描的 AgentTaskView；
- 不公开 Gold、evaluator-only selector、答案路径和修复指引；
- Agent workspace 与 fresh evaluator 分离，只通过 Frozen Patch 交接；
- Prompt 与私有答案索引执行自动重叠扫描，并由独立 review 复核；
- 公开产物拒绝凭据、主机、账号、控制器路径和原始无界日志。

这些措施约束评测过程，不消除公开上游历史导致的模型记忆风险。

## 8. 限制与适用范围

- 仅覆盖 PyTorch，不能代表其他算子框架；
- 50 条仍是研究型规模，细分类别的置信区间有限；
- CUDA 只在冻结的目标硬件/Runtime 上验证；
- distributed 当前为空；
- source/kernel build 成本高，wall-clock 不能直接作为模型性能分数；
- 单 Agent/模型配置的结果是描述性、**non-leaderboard** 证据，不是跨 Agent 排名；
- 未执行反馈可见性对照，不能推断 feedback causality。

适合用途包括 OpBench 平台回归、受控 Agent patch 评测、失败归因和按统一 taxonomy
分层分析。发布结果时应同时记录 Dataset、Agent、Prompt、Runtime、Budget、Scoring
和 repeat identity。

## 9. Reproducibility

验证四份 Dataset：

```bash
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7/dataset.json --require-verified
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7_boundary/dataset.json --require-verified
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7_precision/dataset.json --require-verified
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7_device/dataset.json --require-verified
```

确定性验证正式发布：

```bash
PYTHONPATH=src python scripts/build_v07_quality_release.py \
  --verify-existing --created-at 2026-08-10T16:45:00Z
```

重建 Agent 验证报告：

```bash
PYTHONPATH=src python scripts/summarize_mcp_experiment.py \
  --contract factory/v0.7/p9/validation_contract.json \
  --input-root runs/v0.7_quality_validation \
  --output-root runs/v0.7_quality_validation_report
```

## 10. Version boundary

`opbench-v0.7.0` 的最终引用至少同时记录累计 Dataset hash、p9 release manifest
identity 和 validation contract identity。历史 25-task 产物仅用于可追溯性；不能与
当前 50-task 结果拼接成同一个分母，也不能用旧 18-attempt Boundary cohort 代替
122-attempt 质量验证。
