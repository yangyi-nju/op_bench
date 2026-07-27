# OpBench v0.7 Dataset Card

Release：`opbench-v0.7.0`

状态：Completed

日期：2026-07-28

## 1. 概览

OpBench v0.7 是一个面向真实 PyTorch 算子修复的 verified Dataset release：

- cumulative Dataset：**25 verified** Tasks；
- Boundary Slice：**6 verified Boundary** Tasks，覆盖 B1–B5；
- Precision Slice：**8 verified Precision** Tasks，覆盖 P1–P5。

三份 Dataset 都只包含 `admission_status=verified`、可解析 Source/Environment
Registry、有效 Admission evidence 和可重建 replay identity 的 Task。

| Dataset | Task 数 | Generated Dataset hash |
| --- | ---: | --- |
| `pytorch_v0.7` | 25 | `sha256:4d7bde25e747bcc041aa5105ce5ce881a3f1e9fe2a7545667cdbc2c14d85064a` |
| `pytorch_v0.7_boundary` | 6 | `sha256:810a9cc85c576f44edd2672197ab83b7dfee7f674e597c76c78050bd119d606a` |
| `pytorch_v0.7_precision` | 8 | `sha256:65818466a02e99466386cb8e038dc4da59d91dcb3bea7b83c8901d31a96aa8eb` |

内容寻址 release manifest 位于
[`factory/v0.7/p4/release_manifest.json`](../../factory/v0.7/p4/release_manifest.json)。

## 2. 数据来源与成员关系

Task 来自真实 PyTorch issue/PR 修复，不是合成 bug。Boundary Candidate Search 的
首选 author-date window 为 `2024-01-01..2025-04-30`；每条正式 Task 都绑定
Base Commit、Source Snapshot、上游修复、测试资产、Runtime 与人工 review。

25-task cumulative 的集合关系是：

```text
17 inherited v0.5 cumulative
+ 2 restored Precision P4
+ 6 new Boundary
= 25 verified Tasks
```

Precision Slice 从 v0.5 的 6 条任务继承，并恢复 #129154 与 #144073 两条真实
P4 数值稳定性 Task，合计 8 条。Boundary Slice 的 6 条任务全部由 v0.7 Factory
生产：#117065、#118762、#126461、#139751、#143792、#147352。

继承任务不伪造 Factory provenance。Release Composition 分别保留
`inherited_cumulative`、`inherited_precision`、`restored_precision` 与
`factory_boundary` origin，并对当前 Task、Admission evidence、Registry 和
输入 Dataset hash 做闭合校验。

## 3. Taxonomy coverage

### Boundary Slice

| 子类 | 定义 | Task 数 | 代表问题 |
| --- | --- | ---: | --- |
| B1 | Empty / zero-size | 1 | empty matrix decomposition |
| B2 | Scalar / degenerate shape | 2 | zero-dimensional index、rank-zero cumulative op |
| B3 | Integer / size overflow | 1 | checked storage-offset arithmetic |
| B4 | Parameter endpoints | 1 | default `dim` endpoint |
| B5 | Kernel launch / grid bounds | 1 | non-divisible Y-grid mask |

Boundary 分布为 B1=1、B2=2、B3=1、B4=1、B5=1。B3 使用小 tensor 与极端整数
命中同一 checked-arithmetic 修复路径；B5 使用经人工确认的低内存 surrogate
调用生产 predicate，不依赖真实超大 GPU allocation。

### Precision Slice

| 子类 | 定义 | Task 数 |
| --- | --- | ---: |
| P1 | dtype / promotion | 1 |
| P2 | autocast / mixed precision | 1 |
| P3 | decomposition accuracy | 2 |
| P4 | numerical stability / overflow | 2 |
| P5 | CUDA numerical boundary | 2 |

Precision 分布为 P1=1、P2=1、P3=2、P4=2、P5=2。恢复的两条 P4 Task 使用
matched runtime 解决历史 API/ABI 代差，没有修改目标 bug 或测试语义。

## 4. Runtime 与 Source 策略

25-task cumulative Runtime 分布：

| Runtime tier | Task 数 |
| --- | ---: |
| `cpu_python_overlay` | 17 |
| `cpu_source_snapshot_fuller` | 2 |
| `cuda_python_overlay` | 4 |
| `cuda_kernel_build` | 2 |

Boundary Slice 中 3 条使用 CPU matched-wheel overlay，1 条使用 CUDA
matched-wheel overlay，#143792 与 #147352 使用 exact Base Commit 的 CPU
full-source build。后两者不是“把新源码覆盖到任意 wheel”；evaluator 在每个
registered test 前从当前 authoritative workspace 做 source preparation，并在
隔离 Runtime 中构建后执行 selector。

Matched-wheel 资产冻结官方 wheel digest、实测 image identity、Python/CUDA
需求与 Source 兼容性。Exact-source 资产额外冻结 compiler、build flags、CPU/CUDA
命令分离和 source-build timeout。Runtime 不满足时产生环境或基础设施无效证据，
不会计为 Agent 逻辑失败。

## 5. Admission 与证据链

每条新 Boundary Task 必须完成：

1. Task/Source/Environment/Test/Gold/Patch Scope 内容身份闭合；
2. 6/6 Compatibility checks；
3. Baseline F2P 失败、P2P 保留，且 selector 实际 collected/executed；
4. 相同身份下 Gold F2P/P2P 全部通过；
5. replay hash 与当前 Task bundle 一致；
6. 人工复核根因、泄漏、测试真实性、surrogate 与资源成本；
7. 八阶段不可变 Factory chain：

```text
discovered → screened → bundled → preflight_passed
→ baseline_reproduced → gold_resolved → reviewed → verified
```

六条 Boundary Task 合计 36/36 Compatibility checks 通过；逐条 Baseline 为
F2P 0/1、P2P 1/1，Gold 为 F2P 1/1、P2P 1/1。两条 restored Precision P4
Task 也分别完成 6/6 Compatibility、Baseline/Gold 和 replay gate。

## 6. Candidate funnel

自动 screening 固定 **10** 个真实候选：

- **6 accepted**；
- **2 deferred**；
- **2 rejected**（自动规则）。

自动 accepted 只有资格进入人工 Admission，不等于 verified。后续人工复核确认
6 条 accepted 全部可以制作成 verified Boundary Task；#127448 因改动面较大继续
deferred；#147433 的上游修复随后被 revert，因此人工最终判为 rejected。没有为达到
6 条目标降低 Admission threshold，也没有从 deferred/rejected 中补数。

候选、Decision、reason code 与 deterministic screening index 见
[`factory/v0.7/p3/screening/`](../../factory/v0.7/p3/screening/)。

## 7. 真实 Codex validation

冻结后的 6 条 Boundary Task 使用 `codex_mcp_canonical`、`gpt-5.6-sol` 和
`codex-cli 0.146.0-alpha.3.1` 各运行 3 repeat：

- 18/18 valid Attempts；
- **14 resolved、3 f2p_failed、1 no_patch**；
- Agent terminal：17 finished、1 timeout；
- 18/18 trace complete，MCP protocol error 0；
- 五个正式 root 的 fresh Integrity 均为 14/14 checks passed；
- accepted cohort 的 18 个选择均来自 retry index 1，公开 retry 数为 0。

五条 Task 为 3/3 resolved；rank-zero `cummin` 为 2/3 resolved 加一个 no patch；
Triton Y-grid Task 为 0/3 resolved，人工审计确认是 Agent patch 回归而不是 Runtime
或 transport 故障。

预验收 source cohort 暴露过 SSH pre-execution 断连归因问题。平台加入有界
transport retry 与 remote-start sentinel 后，从统一代码快照重跑完整 source
cohort；最终 6/6 resolved 且无 transport 痕迹。被污染的预验收结果不在正式分母。

完整聚合与限制见
[`docs/v0.7/validation_report.md`](validation_report.md)。该结果只描述当前
Task/platform/Agent 组合，是明确的 **non-leaderboard** evidence，不是正式
cross-Agent ranking，也不支持 feedback-causality 结论。

## 8. 污染与泄漏风险

这些 Task 来自公开 PyTorch 历史，相关 issue、PR、代码与测试可能已出现在模型训练
语料中。因此本 Dataset 不能声称“训练数据无污染”，真实 Codex 结果也不能单独证明
全新问题求解能力。

平台采取的泄漏控制包括：

- AgentTaskView 只公开问题陈述、允许修改范围、Runtime hint 与受控能力；
- Gold Patch、evaluator-only selector 内容、Admission 原始日志与控制器配置不进入
  Agent 视图或公开报告；
- Public artifact scanner 拒绝凭据、控制器路径、主机/目标配置与直接答案线索；
- Candidate 人工 review 检查陈述是否复述修复、测试是否暴露答案；
- Agent workspace 与 fresh evaluator 分离，只通过 Frozen Patch 交接。

这些措施减少评测过程泄漏，但不能消除公开上游历史带来的训练污染风险。

## 9. 限制与适用范围

- 25 条任务仍是小规模、PyTorch-only Dataset；
- 11 条历史 inherited Task 尚未补齐 Boundary/Precision taxonomy；
- Boundary 每个子类只有 1–2 条 Task，不能代表完整算子分布；
- GPU Task 只在冻结的单一硬件类别和 Runtime 资产上验证；
- source-build Task 成本高，Validation 的 wall-clock 不适合作为性能指标；
- 单 Agent、单模型、3 repeat 的 validation 不是正式多 Agent 排名；
- v0.7 没有执行反馈可见性对照，不能推断反馈因果效应。

适合用途：OpBench 平台回归、Task/Runtime/Admission 验证、受控 Agent patch
评测、Boundary/Precision 子集分析。发布者应保留 Dataset、Agent、Prompt、
Runtime、Budget、Scoring 与 repeat identity，不能拼接不可比较的结果。

## 10. Reproducibility

验证三份 Dataset：

```bash
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7/dataset.json --require-verified
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7_boundary/dataset.json --require-verified
PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.7_precision/dataset.json --require-verified
```

重建请求和内容寻址 release：

```bash
PYTHONPATH=src python scripts/build_v07_dataset_requests.py \
  boundary-freeze --repo-root . --output <freeze-request>
PYTHONPATH=src python scripts/freeze_dataset.py \
  --input <freeze-request> --output-dir <freeze-output>
PYTHONPATH=src python scripts/build_v07_dataset_requests.py \
  release --repo-root . --output <release-request>
PYTHONPATH=src python scripts/compose_dataset_release.py \
  --input <release-request> --repo-root . --output-root <empty-output-root>
```

重建公开 validation report 需要五个完成 Integrity gate 的正式 cohort root、
固定的 model/CLI identity 和
[`validation_contract.json`](../../factory/v0.7/p4/validation_contract.json)。
具体命令见
[`validation_report.md`](validation_report.md#7-可复现性与公开边界)。

## 11. 引用与版本边界

发布身份：`opbench-v0.7.0`。使用结果时至少同时记录：

- Dataset ID 与 generated Dataset hash；
- OpBench platform version；
- Agent/Adapter、model、CLI、prompt/feedback policy；
- Runtime Profile 与硬件类别；
- scoring protocol、budget 与 repeat。

v0.5/v0.6 的历史成绩不因 v0.7 Dataset 发布而改写。v0.7 验证只证明 Factory、
Dataset、Runtime、Evaluation、Trace 与 Integrity 能在冻结身份下共同工作。
