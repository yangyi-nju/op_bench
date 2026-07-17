# OpBench 全局项目方案

> 状态：已批准的项目方向，本地文档重写稿
>
> 更新日期：2026-07-17
>
> 当前工程基线：v0.5，17 条 verified task，51 次真实 Codex attempt
>
> 适用范围：v0.6 至 v1.0，并为 v2.0 预留演进边界

## 1. 项目使命

OpBench 的使命是：

> 建立一个面向深度学习算子缺陷修复的、环境绑定、交互受控、可复现、可审计的 Coding Agent 评测基准与实验平台。

项目借鉴 SWE-bench 从真实 Issue/PR 构建修复任务、以补丁和测试判定结果的思想，但进一步把算子任务中的框架版本、源码加载方式、CPU/CUDA 能力、编译路径、数值行为和交互轨迹纳入评测真值。

平台需要稳定回答：

1. 真实算子缺陷能否在声明的源码、运行时和硬件条件下复现；
2. Coding Agent 能否修复目标缺陷且不破坏已有行为；
3. Agent 如何使用可见反馈进行定位、修改、测试和自我修正；
4. 失败来自任务、Agent、基础设施还是 Evaluator；
5. 结果能否由冻结输入和原始产物重新生成。

## 2. 当前基础与核心问题

v0.1–v0.5 已经形成真实 Codex 评测闭环：

- v0.1：真实 Codex、Action Interface、隔离评分；
- v0.2：Source/Environment Registry、Admission、Gold 闭环；
- v0.3：10 条 verified task、3-repeat、Public/Hidden 分层；
- v0.4：CUDA Overlay、CUDA Kernel Build、RemoteDocker；
- v0.5：17 条 verified task、51 次真实 Codex attempt、37/51 resolved、八维指标和完整性硬校验。

这些能力证明 OpBench 已经是可运行的评测集 Demo，但平台能力仍分散在 Agent Bridge、Runner、Workspace、Evaluator、RemoteDocker、Resume 和 Reporter 中。不同模块之间缺少统一的版本化协议、Attempt 生命周期、能力语义、轨迹合同和可重建产物边界。

因此当前最重要的工作不是继续增加 Task 数，而是先把已有真实评测闭环升级为规范平台，再在稳定协议上扩充数据和执行正式研究实验。

## 3. 目标地平线

### 3.1 v1.0 Research Release

答辩前交付一套论文级、可复现的研究发布：

- OpBench Evaluation Protocol v1；
- 高质量 PyTorch 算子缺陷数据集；
- Controlled Agent 主赛道；
- 统一 CLI/MCP Action 能力语义；
- 可见反馈、自我修正、预算和轨迹记录；
- Fresh Evaluation 与失败归因；
- 至少两个冻结 Agent 配置的正式实验；
- 至少一项反馈机制消融；
- 可重新生成的原始结果、汇总和分析；
- 数据集卡、复现文档、论文和离线演示；
- 至少一次 clean-room reproduction，条件允许时增加外部人工复现。

### 3.2 v2.0 Evaluation Platform

答辩后逐步扩展：

- 多框架持续数据发布；
- Controlled 与 Native Agent 双赛道；
- CPU/GPU 分布式 Worker；
- 在线提交、自动 Admission 和人工审核；
- 排行榜、趋势分析和结果签名；
- correctness、precision、boundary、compatibility、performance 等多类问题；
- 面向外部 Agent 的标准接入和持续评测服务。

v1.0 需要为这些方向保留协议和 Adapter 扩展点，但不提前建设 Web 平台、数据库和在线服务。

## 4. 核心原则

### GP-01. 评测可信度优先于功能数量

任务、环境、Agent 能力和评分输入无法证明一致时，更多任务和更多模型没有研究价值。

### GP-02. 环境是任务定义的一部分

Task 真值包括 repository/base commit、source snapshot、runtime、Python ABI、设备能力、source loading mode、构建参数和内容摘要。

### GP-03. Agent 可以使用反馈，但不能获得答案侧信息

Controlled Agent 可以搜索、编辑、运行注册测试和查看 diff。Hidden Test、Gold Patch、PR/Gold provenance、Admission 证据和 Evaluator 控制信息不属于 AgentTaskView。

### GP-04. 能力语义与接入方式分离

Canonical Action Service 定义能力、预算、错误和审计语义。CLI 与 MCP 只是 Adapter，不得各自实现不同的评测规则。

### GP-05. 真实 Agent 是正式评测对象

OpBench 从 v0.1 起就运行真实 Codex。Fake/Scripted Agent 只用于确定性单元测试、故障注入和回归，不能替代真实 Agent 验收或实验。

### GP-06. Agent 工作区与评分环境分离

Agent 最终只交付冻结 Patch。Evaluator 在干净源码中应用精确 Patch，再注入 Hidden Test 并计算 F2P/P2P。

### GP-07. 数据质量优先于数据数量

任务不足时减少数量，不降低 Baseline、Gold、F2P、P2P、环境或来源证据标准。

### GP-08. 版本、数据、协议和实验分离

平台版本、Action Protocol、Evaluation Protocol、Scoring Specification、Dataset 和 Experiment Cohort 分别版本化。

### GP-09. 冻结后变更必须使旧结果显式失效

Evaluator、Task、预算、重试或评分口径实质变化时创建新 Cohort，不把新旧结果拼接计分。

### GP-10. 证据优先于说明

公开结论必须能定位到 Manifest、Schema、原始 Attempt、Patch、Evaluation 和 Summary。聊天总结不能替代仓库和 Artifact 证据。

## 5. 研究贡献与问题

### 5.1 预期贡献

1. **环境绑定的算子修复任务模型**：把源码、运行时、设备、编译和测试资产纳入 Task 真值。
2. **受控交互式 Agent 评测协议**：统一 Action、Feedback、Budget、Termination、Patch Freeze 和 Fresh Evaluation。
3. **算子缺陷 Taxonomy 与质量准入方法**：区分 problem dimension、implementation layer、runtime tier 和 failure contract。
4. **反馈利用与失败模式研究**：分析定位、修改、测试、回退、预算和基础设施阶段。

### 5.2 研究问题

- RQ1：真实 Coding Agent 在不同算子问题类型和运行层级上的修复能力如何？
- RQ2：Agent 在一次 Attempt 内如何使用公开反馈修正假设与 Patch？
- RQ3：提供注册测试反馈相较受限反馈是否改变成功率、成本和失败模式？
- RQ4：环境、Agent、Task 和 Evaluator 故障如何被稳定区分并审计？

## 6. 平台工作流

### 6.1 数据构建

```text
Issue/PR
→ Candidate Screening
→ Task Bundle
→ Source/Environment Binding
→ Baseline Failure
→ Gold Success
→ F2P/P2P Review
→ Admission Evidence
→ Dataset Freeze
```

### 6.2 Agent 评测

```text
Frozen Dataset + Run Manifest
→ Materialize Authoritative Workspace
→ Build Sanitized AgentTaskView
→ Launch Real Agent
→ Canonical Actions + Visible Feedback
→ Budget/Trajectory/Checkpoint
→ Finish and Freeze Final Patch
→ Fresh Evaluation
→ Result Attribution
→ Integrity and Summary Rebuild
```

### 6.3 发布

```text
Design
→ Acceptance Freeze
→ Implementation
→ Focused Tests
→ Full Regression
→ End-to-End Validation
→ Replay/Canary
→ Documentation and Artifact Review
→ Release Decision
```

## 7. 六个版本轴

1. Platform Release，例如 `opbench-v0.6.0`；
2. Action/Capability Protocol；
3. Evaluation Protocol；
4. Scoring Specification；
5. Dataset Release；
6. Experiment Cohort。

可直接比较的 Attempt 至少需要一致或被显式控制：平台提交、协议和 Schema、Dataset Hash、Task/Source/Image Digest、Agent/Model/Adapter、Prompt、Feedback Policy、Budget、Retry/Termination、Hardware 和 Evaluator/Scoring 版本。

v0.5 的 51 次 Attempt 永久标记为 Legacy Baseline，不重命名为 v0.6 Protocol 结果。

## 8. 全局路线图

| 版本 | 目标窗口 | 核心目标 | 退出条件 |
| --- | --- | --- | --- |
| v0.5 | 已完成 | 17 task、51 次真实 Codex Attempt | Legacy Evidence 冻结 |
| v0.6 | 2026-07-17 至 09-15 | Demo → 规范 Agent 评测平台 | 协议、Runtime、MCP、轨迹、Fresh Eval、Replay、真实 Codex 通过 |
| v0.7 | 2026-09-16 至 10-31 | Dataset Factory 与 Boundary Slice | 4–6 条高质量 Boundary Task、完整 Admission |
| v0.8 | 2026-11-01 至 12-20 | Compatibility 与 Evaluation/Scoring RC | Operator Core 分层、Compatibility Slice、Spec RC |
| v0.9 | 2026-12-21 至 2027-02-14 | 正式多 Agent 实验与反馈消融 | Cohort 冻结、重复完整、统计和轨迹分析完成 |
| Contingency | 2027-02-15 至 02-28 | P0/P1 修复与必要重跑 | 不新增研究范围 |
| v1.0 | 2027-03-01 至 04-15 | 论文级研究发布 | 稳定协议、文档、复现、演示和发布 Artifact 完整 |

### 8.1 v0.6：规范评测平台

v0.6 不扩数据集。它把 v0.5 的真实 Codex Action Bridge Demo 标准化为统一 Agent Runtime：

- Versioned Contracts 和 Run Manifest；
- AgentTaskView 与 Capability Policy；
- Authoritative Workspace 和 Patch Freeze；
- Canonical Action Service 与 CLI/MCP Adapter；
- AttemptSession、Budget、Termination、Resume；
- 多轮 Visible Feedback 和 Trajectory；
- Fresh Evaluator、Artifact、Integrity 和 Summary Rebuild；
- Local/Remote Conformance；
- v0.5 Baseline/Gold/Patch Replay；
- 真实 Codex 端到端与批量验证。

七月底可以形成用于简历和演示的中间纵向闭环；它属于 v0.6 的内部工程进度，版本完成标准以 `docs/v0.6/acceptance_matrix.md` 为准。

### 8.2 v0.7：Dataset Factory 与 Boundary

在 v0.6 冻结协议上：

- 建立可复用 Candidate→Admission→Freeze 流程；
- 增加 4–6 条 verified Boundary Task；
- 尝试恢复两条 matched-runtime Precision 候选；
- 审计现有 Operator Core 构念；
- 标记可用于 v0.9 Feedback Ablation 的 Task。

### 8.3 v0.8：Compatibility 与 Spec RC

- 增加 4–6 条 Device/API Compatibility Task；
- 冻结 Evaluation/Scoring Specification RC；
- 明确 Operator Core 与 Legacy Regression；
- 固定正式实验的 Budget、Retry、Aggregation 和排除规则。

### 8.4 v0.9：正式实验

- 数据、平台、协议、Agent、Prompt、预算和硬件全量冻结；
- 至少两个真实 Agent 配置；
- 每条件每任务至少 3 Repeat；
- 执行 Feedback/No-Feedback 配对消融；
- 报告 Resolved、Regression、Cost、Stability 和 Trajectory Failure。

### 8.5 v1.0：研究发布

- 论文、数据集卡、协议、评分规范；
- Quickstart、离线演示、原始 Artifact 和可重建报告；
- Clean-room reproduction；
- 明确样本量、污染、环境和外部有效性限制。

## 9. v0.6 内部里程碑

- M1：协议、Schema、Run Manifest 与兼容策略；
- M2：AgentTaskView、Workspace、Patch Freeze；
- M3：Action Service、MCP、真实 Codex 多轮交互；
- M4：AttemptSession、Budget、Termination、Resume、Trajectory；
- M5：Fresh Evaluator、Artifact、Integrity、Summary Rebuild；
- M6：Local/Remote Conformance、v0.5 Replay、真实批量验证；
- M7：文档、演示、发布检查。

这些是同一 v0.6 Release 的工程检查点，不是独立版本。

## 10. 工程方法

工作直接在当前项目上下文顺序推进：

```text
确认范围
→ 写清设计和验收条件
→ 实现与 TDD
→ Focused Tests
→ Full Regression
→ Diff/Artifact 自审
→ 端到端验证
→ 更新状态与文档
```

原则：

- 默认在当前本地分支开发；
- 一次只推进一个会修改同一模块的里程碑，减少分支和接口漂移；
- 验收以命令、测试、Hash 和 Artifact 为依据，不以过程文案数量为依据；
- 验证只覆盖 OpBench 合同和当前 Attempt 持有的资源，不扩展为环境安全研究；
- Controller 正常连接模型 Provider；Controlled Agent 对任务数据面的访问由 Capability Policy 和 Runtime Profile 约束；
- 项目内部自验收不表述为第三方独立验证；正式发布前另做 clean-room reproduction。

## 11. 全局 Release Gates

- **G0 Design Ready**：目标、非目标、接口、兼容和 Acceptance 已写清；
- **G1 Component Ready**：Schema、单元测试、错误路径和 Legacy Regression 通过；
- **G2 Integrated Runtime Ready**：真实 Agent、Workspace、Action、Session、Evaluator、Artifact 端到端通过；
- **G3 Evidence Ready**：Replay、真实 Codex Canary、Summary Rebuild 和完整性通过；
- **G4 Release Candidate Ready**：配置/数据冻结，无开放 P0/P1，文档与实际一致；
- **G5 Released**：发布决定、Tag、Artifact 和回顾完成。

未执行的项目必须写 `Not Executed`；环境不满足写 `Blocked`；不得用文档意图推断 `Passed`。

## 12. 主要风险与止损

| 风险 | 控制 |
| --- | --- |
| 平台范围过大 | v0.6 不扩数据、不做 UI/排行榜/多框架；Should 可后移 |
| 协议过度设计 | 先覆盖现有真实 Codex 路径，再抽象第二 Adapter |
| Runtime 与 Evaluator 状态分裂 | 一个 Authoritative Workspace、一个 Frozen Patch、一个 Fresh Evaluation 输入 |
| Agent/Infra 失败混淆 | 三轴结果和稳定 failure taxonomy |
| RemoteDocker 清理风险 | 只管理本 Attempt 创建并持有标识的容器/进程，不做宿主广域扫描或 kill |
| 结果不可比较 | Comparability Key 和 Cohort Freeze |
| 数据扩充返工 | v0.7 等待 v0.6 Protocol 稳定后 Admission |
| 样本量过小 | 报告置信区间、per-task 结果和限制，不夸大总体结论 |
| Benchmark 污染 | 冻结来源、时间、Agent 可见输入和 Known Contamination Notes |

## 13. 文档与事实层级

1. `docs/project_plan.md`：全局使命、路线和 Gate；
2. `docs/project_state.md`：当前状态、阻塞和下一动作；
3. `docs/vX.Y/design.md`：版本范围与完成定义；
4. `docs/vX.Y/implementation_plan.md`：实施依赖与验证命令；
5. `docs/vX.Y/acceptance_matrix.md`：可观察验收项；
6. Dataset/Task/Environment/Run Manifest 与内容 Hash；
7. 原始 Attempt、Patch、Evaluation、Integrity 和 Summary；
8. README、CHANGELOG、论文和简历表述。

下层文档不能静默改变上层范围。历史实验报告保持历史事实，不按新路线改写。

## 14. 当前执行顺序

1. 完成本全局方案、v0.6 设计、实施计划和验收矩阵；
2. 将 Boundary 设计作为 v0.7 当前设计；
3. 更新项目状态、README、文档索引和 CHANGELOG；
4. 审阅本地未提交文档；
5. 从 v0.6 M1 开始产品实现；
6. 每个里程碑通过测试和 Artifact 验证后更新状态；
7. v0.6 完成后再开始 v0.7 正式 Admission。
