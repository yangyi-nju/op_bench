# OpBench v0.6 实施计划

日期：2026-07-17

状态：实施中（M1 已完成，M2 待开始）

目标版本：`opbench-v0.6.0`

## 1. 目标与执行约束

本计划把 [v0.6 设计](design.md) 转换为可顺序实施、逐项验证的工程工作。完成后的系统应当把 v0.5 已经可运行的真实 Codex 评测闭环，收敛为具有版本化合同、统一 Action 能力、显式 Attempt 生命周期、Fresh Evaluation 和可重建证据的规范平台。

执行约束：

- v0.6 是一个统一发布，M1–M7 只是内部工程检查点；
- 在当前本地分支按里程碑顺序推进，一次只修改一个连贯边界；
- 功能和缺陷修复采用测试先行，先构造失败证据，再实现最小闭环；
- Fake/Scripted Agent 只承担确定性测试，真实 Codex 承担端到端验收；
- 保留 v0.5 默认行为，v0.6 Runtime 在迁移期通过显式协议或 Profile 开启；
- 产品验证只覆盖 OpBench 合同和当前 Attempt 持有的资源；
- 每个里程碑都必须通过 focused tests、全量回归和变更范围自审；
- 未执行的验证标记为 `Not Executed`，环境不具备则标记为 `Blocked`，不能推断为通过。

## 2. 目标模块布局

优先复用现有模块，只有当合同边界确实独立时才增加新包。目标布局如下：

```text
src/op_bench/
  runtime/
    contracts.py       # 版本化数据合同与严格校验
    manifest.py        # RunManifest、comparability key
    task_view.py       # FullTaskSpec → AgentTaskView 白名单投影
    workspace.py       # Authoritative Workspace、freeze
    actions.py         # Canonical Action Service
    session.py         # AttemptSession、budget、termination、resume
    events.py          # append-only trajectory 与 hash chain
    artifacts.py       # artifact layout、public/private 边界
    evaluation.py      # EvaluationSpec 与 fresh evaluator 编排
    adapters.py        # Agent Adapter 合同和 Codex 接入
    mcp.py             # MCP transport，调用同一 Action Service
    conformance.py     # CLI/MCP、Local/Remote 一致性测试入口
```

预计需要适配但不复制的现有模块：

- `agents.py`：保留既有 Agent，实现标准 Adapter；
- `action_bridge.py`、`actions.py`：迁移为 Canonical Service 的兼容入口；
- `environment.py`、`executor.py`、`remote.py`：实现 Runtime Profile 和 attempt-owned resource contract；
- `evaluator.py`：接收冻结 `EvaluationSpec`，不读取 Agent Workspace；
- `resume.py`：迁移到稳定 Attempt Identity；
- `integrity.py`、`reporter.py`：读取版本化 Artifact 并支持重建；
- `scripts/run_experiment.py`：生成 RunManifest，编排 Runtime；
- `scripts/aggregate_experiments.py`：按冻结 Cohort 和 Attempt Identity 聚合。

Schema 存放在 `schemas/`，至少覆盖 RunManifest、AgentTaskView、Action Request/Observation、Event、Session Result、Evaluation Result 和 Integrity。示例配置放在 `configs/examples/`，不得包含本机地址、凭据或私有路径。

## 3. 依赖顺序

```text
M1 Contracts/Manifest
        │
        ▼
M2 TaskView/Workspace/Freeze
        │
        ▼
M3 Action Service/Adapters/MCP
        │
        ▼
M4 Session/Budget/Trajectory/Resume
        │
        ▼
M5 Fresh Evaluator/Artifacts/Integrity
        │
        ▼
M6 Conformance/Legacy Replay/Real Codex
        │
        ▼
M7 Documentation/Demo/Release Review
```

M1–M5 不得为了提前跑通真实 Agent 而绕过合同。M6 发现的语义差异回到对应模块修复，不在 Adapter 中增加特殊分支掩盖。

## 4. M1：Contracts、Schema 与 RunManifest

状态：已完成（2026-07-17）。C-01～C-08、60 项 M1 focused tests、229 项全量测试、17-task v0.5 migration 和离线示例重建均通过；代码审查发现的 Schema 语义分派、Legacy 类型与路径安全、payload 不可变性、结果轴和公共身份 API 问题已修复并回归。本里程碑未启动 Agent、Docker、SSH 或远程 Runtime。

### 4.1 交付

1. 定义 `schema_version` 和向后兼容规则；
2. 定义 `FullTaskSpec`、`AgentTaskView`、`CapabilityPolicy`、`BudgetPolicy`、`RuntimeProfile`、`SessionSpec`、`EvaluationSpec`；
3. 定义 `RunManifest`、Cohort ID、Attempt Identity 和 Comparability Key；
4. 为所有 wire-level 对象提供 JSON Schema、canonical JSON 和 SHA-256；
5. 从 v0.5 Dataset/Task/Environment 生成确定性默认合同；
6. 未知版本、未知字段、非法枚举、非规范整数和不完整身份 fail closed。

### 4.2 测试

- 每个合同 valid/invalid/round-trip/canonical hash 测试；
- 字段顺序变化不改变 canonical hash；
- 内容变化必须改变对应 identity；
- v0.5 的 17 条 Task 可迁移读取；
- Manifest 记录 expected task × agent × repeat matrix；
- 非法或不完整配置在启动 Agent 前失败。

### 4.3 退出条件

- Acceptance `C-*` 全部通过；
- 现有 Dataset、Task 和 Environment 测试无回归；
- 示例 Manifest 可由 Schema 独立验证并精确重建 hash。

## 5. M2：AgentTaskView、Workspace 与 Patch Freeze

### 5.1 交付

1. 从 `FullTaskSpec` 生成白名单 `AgentTaskView`；
2. 敏感字段和答案来源信息在结构层面不可到达 Adapter；
3. 建立唯一 Authoritative Workspace Identity；
4. 所有读写、测试和 diff 都绑定同一 Workspace；
5. 实现 path、file type、symlink、mode、size 和 writable-scope policy；
6. 实现停止接收 Action、收敛 in-flight mutation、生成 canonical patch、写入 hash 和关闭写能力的 Freeze；
7. 正确覆盖 add/modify/delete/no-patch，拒绝未授权路径和非法文件类型。

### 5.2 测试

- Gold、Hidden、Admission、PR 答案线索和私有环境字段不可见；
- Path traversal、symlink escape、freeze-after-write 被拒绝；
- mutation→test→diff→freeze 使用相同 Workspace ID；
- add/modify/delete patch 在干净副本严格可应用；
- Session、patch artifact 和后续 Evaluation 使用同一 hash；
- 并发 Action 与 Finish 只能形成一个冻结结果。

### 5.3 退出条件

- Acceptance `T-*`、`W-*` 全部通过；
- Bad/Gold/Empty patch fixtures 结果确定；
- v0.5 Action Bridge 的合法补丁仍可导出。

## 6. M3：Canonical Action Service、Adapter 与 MCP

### 6.1 交付

1. 实现标准 Action Request/Observation 和稳定 Error Code；
2. 实现 `workspace_list/search/read/write/apply_patch`；
3. 实现 policy-bound `command_run`、registry-bound `test_run`、`vcs_diff` 和 `session_finish`；
4. 在 Service 端重新校验 session、action、path、selector、budget 和 state；
5. 以 `action_id` 提供幂等行为；
6. CLI 与 MCP 只做传输和序列化，调用同一 Service；
7. 将 `codex_action_bridge` 接入标准 Adapter，保留 Legacy 兼容入口；
8. 完成真实 Codex 的 read→edit→test→finish 纵向闭环。

### 6.2 测试

- 每个 Action 的成功、拒绝、超时、输出截断和预算变化；
- 重复 `action_id` 不重复执行 mutation 或 test；
- 非注册 selector、非法 cwd、Freeze 后请求稳定拒绝；
- 同一 Scripted Sequence 经 CLI/MCP 得到等价 Observation、Patch 和 Event；
- Adapter 无法获得 FullTaskSpec、Evaluator 或 Workspace 内部对象；
- 真实 Codex canary 使用标准接入完成多轮交互。

### 6.3 退出条件

- Acceptance `A-*` 全部通过；
- 真实 Codex Canary 的 Manifest、Trajectory、Patch 和终态完整；
- Legacy Bridge 回归通过且没有复制一套 Action 规则。

## 7. M4：AttemptSession、Budget、Termination、Trajectory 与 Resume

### 7.1 交付

1. 实现 `created→preparing→ready→running→stopping→freezing→terminal` 状态机；
2. 冻结终止优先级和唯一 Terminal 规则；
3. 在 Service 端计算 wall-clock/action/test/command/output budgets；
4. 实现 timeout、cancel、agent exit、provider error、runtime error 和 finish 收敛；
5. 实现 append-only Event、全局 sequence、request/observation pairing 和前序 hash；
6. 大输出独立存 Artifact，Event 仅记录引用；
7. 实现稳定 Attempt Identity、有效终态判定、resume 去重和 retry audit；
8. 完成终态后重复 resume/finish 不改变结果。

### 7.2 测试

- Finish/Timeout/Cancel/Exit 竞争仍只有一个 Terminal；
- Budget 边界前一项允许、后一项拒绝且记账一致；
- Event sequence 无缺口、hash chain 可重算；
- Action Request 必须有且只有一个公开 Observation；
- 配置、Task 或协议变化产生新 Attempt Identity；
- 已完成有效 Attempt 跳过，Infrastructure Invalid 按 policy 重试并保留历史；
- 中断恢复不重复计分或覆盖原始 JSONL。

### 7.3 退出条件

- Acceptance `S-*`、`E-*` 中 Session/Trajectory 项通过；
- 故障注入测试无孤儿终态、重复计分或不可解释预算差异。

## 8. M5：Fresh Evaluator、Artifact、Integrity 与 Summary

### 8.1 交付

1. 在干净 Source/Container 中校验 Base Identity 并严格应用 Frozen Patch；
2. 评分阶段才注入 Evaluation-only Test；
3. 记录 F2P/P2P collected/executed/skipped/failed；
4. 输出 attempt validity、agent terminal、evaluation outcome 三轴结果；
5. 建立 public/private artifact 分层；
6. 校验 Manifest→Session→Patch→Evaluation→Result→Summary 的 hash 和引用；
7. 从原始 Artifact 重建 `results.jsonl` 和 `summary.json`；
8. expected/observed matrix、duplicate、missing、unexpected 和 retry audit 都进入 Integrity。

### 8.2 测试

- Bad patch unresolved、Gold patch resolved、P2P regression 单独归因；
- invalid/no patch、evaluator error、runtime error 不混入 resolved；
- 未 collected 或全 skipped 的测试不能伪装成通过；
- Evaluator 不读取 Agent Workspace 或 Agent 修改的测试文件；
- 任一 Patch byte 变化触发三方身份不一致；
- 删除/篡改 Event、Evaluation 或 Attempt 会使 Integrity 失败；
- Summary 重建与存储值 byte-equivalent 或 canonical-equivalent。

### 8.3 退出条件

- Acceptance `V-*` 和剩余 `E-*` 全部通过；
- v0.5 的现有 Gold/Bad controls 在新 Evaluator 上语义一致。

## 9. M6：Runtime Conformance、Legacy Replay 与真实 Agent 验证

### 9.1 交付

1. 冻结 Local CPU、Remote CPU、CUDA Overlay、CUDA Kernel Build Profile；
2. 统一 source loading、mount、timeout、resource、network 和 cleanup contract；
3. 资源只按 Attempt-owned identity 创建、列举和清理；
4. 执行 CLI/MCP 和 Local/Remote Canonical Sequence Conformance；
5. 回放 17 Baseline、17 Gold 和 51 Legacy Final Patch；
6. 运行至少一个真实 Codex CPU Canary；
7. 在资源可用时运行代表性 Remote CPU、CUDA Overlay、CUDA Kernel Canary；
8. 运行小规模真实 Codex 批量，检查 resume、summary 和 failure attribution。

### 9.2 验证口径

- Replay 验证 Evaluator 语义，不把旧补丁计为新 Agent 成绩；
- 硬件暂不可用时相关项只能为 `Blocked`，其余平台项继续验证；
- Conformance 使用合同断言、Fixture 和 Attempt 资源清单证明；
- 只验证本 Attempt 创建且持有 identity 的容器、子进程和文件；
- Legacy 差异必须逐项归因，不能用总通过率掩盖。

### 9.3 退出条件

- Acceptance `R-*` 全部达到其声明状态；
- 17+17+51 Replay 完整，或每个不可运行项有可复核的环境解释；
- 真实 Codex 标准路径至少有一个有效端到端 Attempt；
- 无静默 Legacy 语义变化。

## 10. M7：文档、Demo 与 Release Review

### 10.1 交付

1. 更新中英文 README、Quickstart、配置示例和开发指南；
2. 提供一条离线 Scripted smoke 和一条真实 Codex canary 命令；
3. 提供 Artifact 验证、Summary 重建和 Resume 示例；
4. 记录支持矩阵、已知限制、失败分类和 Comparability Key；
5. 生成 v0.6 release notes 和可离线展示的代表性 Artifact；
6. 对照 Acceptance Matrix 完成最终审查。

### 10.2 退出条件

- 新环境按文档可以完成安装、Dataset Validation 和离线 smoke；
- 真实 Codex 演示使用正式 Runtime，不使用专用捷径；
- 文档中的 CLI、Schema、Artifact path 和代码一致；
- 所有 Must 为 `Passed`，无开放 P0/P1；
- 未把 v0.5 历史结果冒充 v0.6 成绩，也不声称已经完成正式排名或反馈因果实验。

## 11. 每个里程碑的固定验证

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src \
  python -m unittest discover -s tests -p 'test_*.py'

PATH=.venv/bin:$PATH PYTHONPATH=src \
  python scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json --require-verified

git diff --check
git status --short
```

里程碑还必须运行对应 focused tests、Schema validation、Artifact semantic validation 和必要的 end-to-end command。测试结果记录准确的命令、退出码、通过数、失败数和 Artifact identity。

## 12. 范围控制与降级顺序

时间或环境不足时，按以下顺序降级：

1. 后移 Should 项，例如 Trace Card 美化、细粒度 Cost 和更多批量 canary；
2. 减少远程硬件 canary 的覆盖数量，但保留 Profile、Replay 和阻塞事实；
3. 不减少合同严格性、AgentTaskView 隔离、Patch Freeze、Fresh Evaluation、真实 Codex CPU Canary、Integrity 或 v0.5 Replay；
4. 不用增加 Task 数、Web UI、排行榜或第二框架补偿平台核心缺口。

## 13. 完成判定

最终完成条件由 [v0.6 验收矩阵](acceptance_matrix.md) 定义。只有全部 Must 为 `Passed`、没有开放 P0/P1、全量回归和真实 Codex 标准路径通过，项目状态才能从 `In Progress` 更新为 `v0.6 Completed`。
