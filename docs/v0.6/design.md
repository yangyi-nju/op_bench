# OpBench v0.6 规范 Agent 评测平台设计

日期：2026-07-17

状态：设计方向已批准，待实现

## 1. 版本定位

`opbench-v0.6.0` 的目标是完成 OpBench 从“可运行的真实 Agent 评测集 Demo”到“规范 Agent 评测平台”的转变。

v0.5 已经完成真实 Codex Action Bridge、17 条 verified task、CPU/CUDA/Kernel Runtime、Fresh Evaluation、Resume 和 51 次正式 Attempt。v0.6 不否定这些能力，也不重新证明“真实 Agent 能运行”；它要把已有能力收敛为统一、版本化、可扩展、可审计的运行时和评测协议。

本版本只建设平台，不扩 Boundary/Compatibility 数据集，不执行正式 Agent 排名或反馈因果研究。

## 2. Demo 与平台的差距

| v0.5 Demo 现状 | v0.6 平台要求 |
| --- | --- |
| Codex Action Bridge 是参考路径 | Agent Adapter 接入统一 Runtime |
| Action 语义分散在 Bridge/Workspace/Executor | Canonical Action Service 是唯一能力真值 |
| Runner 隐含 Attempt 生命周期 | AttemptSession 显式管理状态、预算和终止 |
| 日志主要服务调试 | Versioned Event/Trajectory 可重建 |
| Result 与 Summary 能聚合 | Manifest→Session→Patch→Evaluation→Summary 全链路 Hash 绑定 |
| Local/Remote 路径存在行为差异风险 | Runtime Profile 和 Conformance 固定语义 |
| Resume 以结果记录为中心 | Attempt Identity、Checkpoint 和终态幂等 |
| Agent 可见输入靠 Adapter 约定 | AgentTaskView 明确白名单和私有边界 |
| Fresh Evaluation 已存在 | Frozen Patch 是唯一评分输入，Evaluator 合同版本化 |
| 真实 Codex 已运行 | Codex 通过标准 Adapter/MCP 进入统一平台 |

## 3. Release Contract

### 3.1 Must

v0.6 必须交付：

1. 版本化 Task、AgentTaskView、Action、Session、Evaluation、Artifact 和 Run Manifest 合同；
2. 一个 Authoritative Workspace，所有 Action、Test、Diff 和 Freeze 指向同一状态；
3. Canonical Action Service，CLI/MCP 共享相同业务实现；
4. 真实 Codex Adapter 和多轮工具/反馈闭环；
5. AttemptSession、Budget、Timeout、Cancel、Finish、Resume 和唯一终态；
6. Action/Observation/Test/Patch/Terminal Trajectory；
7. 唯一 Final Patch Freeze 和三处一致的 Patch Hash；
8. Fresh Evaluator，只接收 Frozen Patch 和 EvaluationSpec；
9. F2P、P2P、Regression、Agent Failure、Task Failure、Infrastructure Failure 和 Evaluator Failure 归因；
10. Run Manifest、Session Result、Evaluation Result、Integrity 和 Summary Rebuild；
11. Local CPU、Remote CPU/CUDA、CUDA Kernel 等既有 Runtime 的统一 Profile 与 Conformance；
12. v0.5 的 17 Baseline、17 Gold 和 51 Legacy Final Patch Replay；
13. Fake/Scripted 确定性回归和真实 Codex 端到端 Canary；
14. 统一 CLI、配置示例、开发文档、复现说明和可展示 Demo；
15. 默认 Legacy 路径兼容，显式选择 v0.6 Runtime，直到迁移完成。

### 3.2 Should

- Patch Checkpoint 和增量 Diff 统计；
- 细分 Token/Action/Test/Wall-clock Budget；
- CLI/MCP Byte-equivalent Conformance；
- 结构化 Trace Card；
- 代表性 Runtime 的小规模批量 Codex 验证；
- 从旧 results.jsonl 导入只读 Legacy Trajectory 摘要。

### 3.3 非目标

- 新增 Boundary 或 Compatibility Task；
- Web UI、数据库、排行榜和在线提交；
- 多框架与分布式 Worker；
- 正式 Agent 排名、Feedback Ablation 或论文主实验；
- Native/Network-enabled 独立赛道；
- 任意宿主 Shell 作为 Controlled Track 默认能力；
- 与 OpBench 评测合同无关的环境安全研究。

## 4. 核心不变量

### V06-I01. One Attempt, One Authority

同一 Attempt 的编辑、测试、Diff、Freeze 和最终 Patch 来自同一 Authoritative Workspace。

### V06-I02. Patch-only Evaluation Handoff

Agent Workspace 的缓存、构建物、测试修改和未跟踪状态不能直接进入评分。Evaluator 只接收 Frozen Patch 字节和版本化 EvaluationSpec。

### V06-I03. Agent-visible Whitelist

Agent 只看到 AgentTaskView、Capability Policy、Action Schema 和 Action Observation。Gold、Hidden、Admission、来源答案信息和 Evaluator 私有配置不可见。

### V06-I04. Server-enforced Capability

Adapter 或 Prompt 中的工具说明不是授权真值。Action Service 对每次请求重新验证 Session、Action、Path、Selector、Budget 和状态。

### V06-I05. One Terminal

每个 Attempt 恰好一个终态。Finish、Timeout、Cancel、Agent Exit 和 Infrastructure Error 竞争时按冻结优先级收敛。

### V06-I06. Immutable Evaluation Input

Session Result、Run Manifest 和 Evaluator 使用相同 Patch Hash；不允许评分前后重新读取可变 Workspace 生成不同 Patch。

### V06-I07. Rebuildable Evidence

Public Summary 能从冻结 Manifest、Session、Trajectory 和 Evaluation 重新生成；重建结果与存储结果不一致时 Release 失败。

### V06-I08. Legacy Facts Stay Historical

v0.5 的 51 次 Attempt 不重新标记为 v0.6 结果。Replay 是兼容性证据，不改变历史分数。

## 5. 逻辑架构

```text
Dataset + Task/Source/Environment Registry
                    │
                    ▼
              Run Manifest
                    │
                    ▼
          Agent Evaluation Runtime
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 AgentTaskView  AttemptSession  Runtime Profile
       │            │            │
       ▼            ▼            ▼
 Agent Adapter → Canonical Action Service
       │            │
       │            ▼
       │     Authoritative Workspace
       │            │
       └──── Events/Observations
                    │
                    ▼
              Frozen Patch
                    │
                    ▼
              Fresh Evaluator
                    │
                    ▼
    Session/Evaluation/Integrity/Summary
```

模块边界：

- **Control Plane**：Task、Manifest、Session、Budget、Capability、Artifact；
- **Agent Plane**：AgentTaskView、Adapter、MCP/CLI Transport；
- **Workspace Plane**：Source Materialization、Canonical Actions、Registered Tests、Patch Freeze；
- **Evaluation Plane**：Fresh Source、Hidden Injection、F2P/P2P、Result；
- **Evidence Plane**：Events、Hashes、Integrity、Summary。

Agent Adapter 不接收 Full Task、Gold、Hidden 或 Evaluator 对象。Evaluator 不读取正在运行的 Agent Workspace。

## 6. Versioned Contracts

### 6.1 RunManifest

至少包含：

- platform/action/evaluation/scoring schema versions；
- dataset id/content hash；
- task/source/environment/image identity；
- agent/model/adapter identity；
- system/task prompt hash；
- feedback/capability/budget/termination/retry policy；
- runtime profile 与 hardware capability；
- expected task/agent/repeat matrix；
- creation time 与 cohort id。

### 6.2 FullTaskSpec 与 AgentTaskView

`FullTaskSpec` 属于 Control/Evaluation Plane，包含完整 Task、Hidden、Gold、Admission 和环境资产引用。

`AgentTaskView` 是白名单投影，只包含：

- task id 和规范化 Issue；
- 允许公开的 Framework/Operator/Runtime 提示；
- Public/Registered Test 名称和说明；
- Capability Policy 摘要；
- Budget 和终止说明；
- 禁止暴露答案来源的稳定附件。

生成后进行 Schema Validation 和敏感字段扫描。不得把 FullTaskSpec 作为方便参数继续传给 Adapter。

### 6.3 SessionSpec

包含 Attempt Identity、Workspace Identity、Capability Policy、Budget、Deadline、Adapter Config、Runtime Profile、Artifact Root 和 Resume Policy，不包含 Hidden/Gold。

### 6.4 EvaluationSpec

包含 Base Source Identity、Frozen Patch Hash、Hidden/Public Test、F2P/P2P、Runtime Profile、Timeout 和 Scoring Version，只在 Agent 终止后交给 Evaluator。

## 7. Canonical Action Service

v0.6 标准 Action：

| Action | 作用 | 关键约束 |
| --- | --- | --- |
| `workspace_list` | 列目录 | Path Policy、数量/深度限制 |
| `workspace_search` | 搜索源码 | Workspace 内、结果和字节限制 |
| `workspace_read` | 读文件 | Regular file、范围和大小限制 |
| `workspace_write` | 写文件 | Write Policy、原子替换、Freeze 后拒绝 |
| `workspace_apply_patch` | 应用 Patch | Scope、解析、原子性、Freeze 后拒绝 |
| `command_run` | 运行允许命令 | Policy Allowlist、cwd、timeout、输出限制 |
| `test_run` | 运行注册测试 | Selector Registry、预算、结构化结果 |
| `vcs_diff` | 查看当前 Patch | Canonical Git Diff、大小限制 |
| `session_finish` | 请求结束 | 幂等、触发收敛和 Freeze |

Request 至少包含 `schema_version`、`session_id`、`action_id`、`action_name`、`arguments`、`client_sequence` 和 `deadline_ms`。

Observation 至少包含 `ok`、稳定 `error_code`、公开 `message`、结构化 `data`、`started_at`、`ended_at`、`budget_delta` 和可选 `mutation_state`。

同一 `action_id` 重试返回相同 Observation，不重复执行有副作用的操作。

## 8. Adapter 与 MCP

### 8.1 Adapter Contract

Adapter 只负责：

- 把 AgentTaskView 和 LaunchContext 转换为 Agent 输入；
- 建立 CLI/MCP Transport；
- 把模型工具调用转换为 Canonical Action Request；
- 返回 Agent Exit/Finish 和 Provider 使用信息。

Adapter 不负责 Workspace、预算判定、Patch、Evaluator 或 Scoring。

### 8.2 真实 Codex

Codex 是 v0.6 必须支持的真实 Agent：

- Controller 正常连接 Codex Provider；
- Codex 通过 attempt-scoped MCP 或 CLI Compatibility Adapter 调用工具；
- 真实交互必须包含读取、至少一次修改、至少一次注册测试和 Finish；
- Provider/Rate-limit 记录为运行事实，不与 Agent 修复失败混淆；
- 现有 `codex_action_bridge` 保持 Legacy 可用，直到新 Adapter 通过兼容验证。

### 8.3 Transport 等价

CLI 与 MCP 调用同一 Service，不复制业务逻辑。相同 Scripted Action Sequence 应得到相同 Patch、Error Code、Budget Delta 和 Event 语义。

## 9. Authoritative Workspace 与 Patch Freeze

Workspace 创建后记录：

- base source digest；
- materialization mode；
- runtime profile；
- writable scope；
- Git status baseline；
- workspace id。

所有修改必须通过 Canonical Action Service。`test_run` 与 `vcs_diff` 使用同一目录。

Freeze 顺序：

1. 停止接收新 Action；
2. 等待或收敛 in-flight Action；
3. 记录最终 Git status；
4. 生成 Canonical Patch；
5. 校验 Path/Mode/Symlink/Size Policy；
6. 写入 Patch Artifact 和 SHA-256；
7. 关闭 Workspace 写能力；
8. 生成 Session Terminal。

Patch 必须覆盖 add/modify/delete，禁止依赖未进入 Patch 的缓存或工作树状态。

## 10. AttemptSession

### 10.1 状态机

```text
created
→ preparing
→ ready
→ running
→ stopping
→ freezing
→ terminal
```

异常可以从任一非终态进入 `stopping`，但只能生成一个终态。

### 10.2 终止原因

- `agent_finished`；
- `agent_exited`；
- `budget_exhausted`；
- `timeout`；
- `cancelled`；
- `workspace_error`；
- `runtime_error`；
- `provider_error`；
- `platform_error`。

终止原因不直接等于评分结果。例如 Agent 正常 Finish 可能 unresolved；Provider Error 属于 Infrastructure Failure，不计为 Agent 修复失败。

### 10.3 Budget

至少支持：

- wall-clock；
- action count；
- test count；
- command count；
- output bytes；
- 可选 provider token/cost。

Budget 在 Service 端扣减。Adapter 自报只用于对账。

### 10.4 Resume

Attempt Identity 由 Cohort、Task、Agent、Repeat 和有效配置 Hash 组成。Resume 不重复已完成有效 Attempt；配置或 Task 内容变化会生成新 Identity。原始 JSONL append-only，聚合按稳定 Identity 去重并保留 Retry Audit。

## 11. Trajectory 与 Artifact

### 11.1 必需事件

- session created/prepared/started；
- agent launched/exited；
- action requested/observed；
- test started/completed；
- budget updated/exhausted；
- finish/timeout/cancel requested；
- patch freeze started/completed/failed；
- evaluation started/completed；
- terminal emitted。

每个 Event 包含 Schema Version、Session ID、Sequence、Timestamp、Event Type、Public Payload 和前一 Event Hash。大输出单独存 Artifact，只在 Event 中保存 Hash、Size 和 Media Type。

### 11.2 Artifact Layout

```text
runs/<cohort>/
  run_manifest.json
  attempts/<attempt-id>/
    agent_task_view.json
    session_result.json
    events.jsonl
    final.patch
    public_evaluation.json
    private_evaluation.json
    integrity.json
  results.jsonl
  summary.json
```

Public Artifact 不包含 Hidden 源码、Gold、私有 Test 输出、Credential 或宿主路径。

### 11.3 Integrity

`integrity.json` 至少验证：

- Manifest、Task、Source、Environment、Policy Hash；
- Event Sequence/Pairing/Terminal；
- Session/Patch/Evaluator 三方 Patch Hash；
- Evaluation 与 Summary 重建；
- Expected/Observed Attempt Matrix；
- 缺失、重复、Unexpected Attempt 和 Retry Audit。

## 12. Fresh Evaluator 与结果模型

Evaluator 在新的 Source Copy/Container 中：

1. 校验 Base Source Identity；
2. 严格应用 Frozen Patch；
3. 注入 Evaluation-only Test Asset；
4. 执行 F2P/P2P；
5. 记录 collected/executed/skipped/failed；
6. 生成私有证据和公开结果；
7. 清理 Evaluator-owned Resource。

不允许 Patch Fuzz Apply、从 Agent Workspace 读取额外文件、把 Hidden 失败反馈给已结束 Agent，或用 exit 0 掩盖未执行 Test。

结果至少分三轴：

- `attempt_validity`：valid / infrastructure_invalid；
- `agent_terminal`：finished / exited / timeout / budget / cancelled；
- `evaluation_outcome`：resolved / f2p_failed / p2p_regression / invalid_patch / no_patch / evaluation_error。

Infrastructure Invalid 另有稳定 `invalid_reason`，不计入 Agent resolved denominator，但保留原始记录。

## 13. Runtime Profile 与 Conformance

支持的 Profile 继承现有能力：

- Local/Remote CPU Python Overlay；
- Remote CUDA Python Overlay；
- Remote CUDA Kernel Build；
- 必要的 CPU Compile/Source Build Profile。

Profile 冻结 Executor、Image、Source Loading、Mount、Network、Timeout、Resource 和 Cleanup Policy。

安全边界采用普通工程约束：

- Container/Process 必须带 Attempt-owned 标识；
- 只清理当前 Attempt 创建并记录的资源；
- 不使用广域 `pkill` 或不带 Attempt/Container Identity 的清理；
- 本地配置、Credential 和 Remote Host 信息不进入仓库或 Public Artifact；
- Controlled Runtime 在可行时关闭 Agent Task Data-plane 网络；Controller 到 Provider 的连接不受此限制；
- 通过参数断言、Mock、Fixture、Attempt 资源清单和正常集成测试验证。

Conformance 对相同 Canonical Sequence 检查 Action、Patch、Result 和 Cleanup 语义，而不是要求不同硬件耗时一致。

## 14. Compatibility 与 Replay

### 14.1 Legacy Compatibility

- `scripts/run_experiment.py` 保持主入口；
- 默认行为在迁移前保持 v0.5 Legacy；
- 新 Runtime 由显式 Profile/Protocol Version 选择；
- 旧 Dataset/Task Manifest 可读取，新增字段提供确定默认值或显式 Migration；
- 旧 results.jsonl 和 summary 保持可读。

### 14.2 v0.5 Replay Matrix

v0.6 完成前执行：

- 17/17 Baseline Failure Replay；
- 17/17 Gold Success Replay；
- 51/51 Legacy Final Patch Replay；
- 每个差异给出 Task/Environment/Protocol 归因；
- Legacy Summary 与 v0.6 Summary 分离。

Replay 证明新 Evaluator/Runtime 没有静默改变历史语义，不生成新的 Agent 成绩。

## 15. CLI 与开发体验

统一入口至少支持：

- `--runtime-profile`；
- `--action-protocol`；
- `--evaluation-protocol`；
- `--feedback-policy`；
- `--budget-profile`；
- `--agent` / `--agent-repeat`；
- `--run-manifest` / `--resume`；
- `--verify-artifacts` / `--rebuild-summary`；
- `--verified-only` 和完整性硬校验。

错误信息必须给出稳定 Error Code、相关 Artifact 路径和下一步，不只输出 traceback。

## 16. 内部里程碑

| 里程碑 | 内容 | 退出证据 |
| --- | --- | --- |
| M1 | Contract、Schema、Manifest、Compatibility | Schema/round-trip/legacy tests |
| M2 | AgentTaskView、Workspace、Patch Freeze | denylist、mutation→test、add/delete/hash tests |
| M3 | Action Service、MCP、真实 Codex | CLI/MCP conformance、真实 CPU interaction |
| M4 | Session、Budget、Termination、Resume、Trajectory | race/idempotency/resume/event rebuild tests |
| M5 | Fresh Evaluator、Artifact、Integrity、Summary | bad/gold controls、patch identity、rebuild |
| M6 | Runtime Conformance、v0.5 Replay、真实批量验证 | Local/Remote matrix、17+17+51、Codex canaries |
| M7 | 文档、演示、Release Review | quickstart、clean run、无 P0/P1 |

七月底的纵向闭环是 M3/M4 的中间演示，不是独立版本。只有 M1–M7 全部满足才标记 v0.6 Completed。

## 17. 失败严重级别

- **P0**：Hidden/Gold 泄漏、宿主破坏、结果伪造、Patch/Evaluation 身份失效；
- **P1**：核心评测错误、错误终态、不可重建、Legacy 语义静默变化；
- **P2**：重要覆盖、诊断、兼容或可维护性缺陷；
- **P3**：文档、体验和非阻塞改进。

v0.6 Release 不允许 Open P0/P1。P2/P3 可以有明确 Backlog，但不得改变公开结论。

## 18. 完成定义

当且仅当：

1. M1–M7 退出条件全部满足；
2. `acceptance_matrix.md` 所有 Must 为 Passed；
3. 全量单元/集成测试和 Dataset Validation 通过；
4. CLI/MCP 使用同一 Canonical Action Service；
5. 至少一个真实 Codex CPU Attempt 完成多轮读取、修改、注册测试和 Finish；
6. CPU、CUDA Overlay、CUDA Kernel Build 各有代表性 Runtime/Replay 证据，真实 Canary 范围按可用资源记录；
7. Final Patch 在 Session、Artifact、Evaluator 中 Hash 一致；
8. Fresh Evaluator 的 Bad/Gold/Agent Controls 结果正确；
9. 17 Baseline、17 Gold、51 Legacy Patch Replay 完整或每个差异有批准解释；
10. Summary 可从原始 Artifact 精确重建；
11. README、Quickstart、设计、实现和实际 CLI 一致；
12. 无 Open P0/P1；
13. 发布说明不声称 v0.6 已证明 Feedback 因果、Agent 排名或对总体算子任务的泛化。

v0.6 完成后的准确表述是：

> OpBench 已从特定 Agent Bridge 驱动的评测集 Demo，升级为具有版本化任务视图、统一 Action/MCP 接入、Attempt 生命周期、多轮反馈轨迹、唯一 Patch Freeze、Fresh Evaluation、失败归因、兼容回放和可重建 Artifact 的规范 Coding Agent 评测平台，并继续支持真实 Codex 端到端运行。
