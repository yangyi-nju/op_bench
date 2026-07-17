# OpBench Project State

更新时间：2026-07-17（Asia/Shanghai）

## Current baseline

| 项目 | 当前值 |
| --- | --- |
| 当前稳定版本 | v0.5 Completed |
| 数据集 | `pytorch_v0.5`，17 条 verified task |
| 正式实验 | 51 次真实 Codex attempt，37/51 resolved |
| 当前开发版本 | `opbench-v0.6.0` |
| 当前目标 | Demo → 规范 Agent 评测平台 |
| 当前阶段 | M1–M5 已完成，准备 M6 Runtime Conformance/Replay/真实 Agent 验证 |
| v0.6 产品代码 | Contracts/Manifest、AgentTaskView、Authoritative Workspace、Patch Freeze、Canonical Action Service、AttemptSession、hash-chain Trajectory、Fresh Evaluation、public/private Artifact、Evaluation-aware Resume、Integrity 与确定性 Summary 已实现 |
| 正式新实验 | 尚未启动 |

## Current decisions

- v0.6 使用一个版本号和一套完成标准，M1–M7 是内部工程检查点。
- v0.6 不扩数据集，集中建设规范 Agent Evaluation Runtime。
- 真实 Codex 是既有正式评测路径，不是 v0.6 才引入的新能力。
- Fake/Scripted Agent 仅用于确定性测试和故障注入。
- CLI/MCP 共享 Canonical Action Service，不各自实现评测规则。
- Agent Workspace 与 Fresh Evaluator 分离，只以 Frozen Patch 交接。
- Boundary 数据扩充属于 v0.7。
- 开发直接在当前本地分支按里程碑顺序推进。
- 验证只覆盖 OpBench 合同和当前 Attempt 持有的资源。

## Open items

| ID | 状态 | 内容 | 关闭条件 |
| --- | --- | --- | --- |
| V06-DOCS | Passed | 全局方案与 v0.6/v0.7 文档重写 | 入口一致、链接有效、回归验证通过 |
| V06-M1 | Passed | 协议、Schema、Run Manifest 与兼容策略 | C-01～C-08 已通过；60 focused tests、229 full tests、17-task migration 与示例重建通过 |
| V06-M2 | Passed | AgentTaskView、Authoritative Workspace 与 Patch Freeze | T-01～T-07、W-01～W-10 已通过；43 core、87 focused/compat、274 full tests、17-task migration、Schema/示例重建与 Legacy Action Bridge 回归通过 |
| V06-M3 | Passed | Canonical Action Service、CLI/MCP 与标准 Adapter 边界 | A-01～A-12 已通过；28 focused、302 full tests、17-task Dataset、示例 Manifest、tracked JSON 与 Legacy Action Bridge 回归通过；真实 Codex 标准 canary 保留为 M6 gate |
| V06-M4 | Passed | AttemptSession、Budget、Trajectory、Public Artifact 与 Resume Ledger | S-01～S-10、E-02～E-04 已通过；61 focused、194 runtime、363 full tests、17-task Dataset、示例 Manifest 与 tracked JSON 通过；审查 Critical/Important/Minor 均为 0；E-01 evaluation 与 E-05 private artifact 保留为 M5 gate |
| V06-M5 | Passed | Fresh Evaluator、private Artifact、Integrity 与 Summary | 62 focused、252 runtime、421 full tests，17-task Dataset、示例 Manifest、tracked JSON、compileall 与 diff check 通过；独立复审 Critical/Important/Minor 0/0/0；仅使用本地确定性 fixture，未启动 Agent、Docker、SSH、远程 Runtime 或网络探针 |
| REMOTE-CLEANUP | Backlog | RemoteDocker timeout/cleanup 需要收敛到 Attempt-owned container/process | v0.6 M6 Conformance 通过 |

## Next actions

1. 启动 M6 Runtime Profile 与 Local/Remote Canonical Sequence Conformance；
2. 回放 v0.5 Baseline、Gold 与 51 个 Legacy Final Patch；
3. 在 M6 合同与本地验证通过后执行真实 Codex CPU Canary；
4. v0.6 完成前不启动 v0.7 正式 Admission。

## Status rules

- `Pending`：尚未开始；
- `In Progress`：正在实现或验证；
- `Passed`：验收证据完整；
- `Failed`：执行完成但未满足验收；
- `Blocked`：外部或环境条件阻止执行；
- `Not Executed`：未运行，不得推断结果。
