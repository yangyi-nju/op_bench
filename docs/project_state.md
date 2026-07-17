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
| 当前阶段 | 设计与实施规划 |
| v0.6 产品代码 | 尚未在当前基线开始实现 |
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
| V06-M1 | Pending | 协议、Schema、Run Manifest 与兼容策略 | M1 Acceptance 全部通过 |
| REMOTE-CLEANUP | Backlog | RemoteDocker timeout/cleanup 需要收敛到 Attempt-owned container/process | v0.6 M6 Conformance 通过 |

## Next actions

1. 审阅本地未提交的全局方案、v0.6 和 v0.7 文档；
2. 确认 v0.6 Acceptance Matrix 作为实现完成标准；
3. 从 v0.6 M1 Contracts、Schema 与 RunManifest 开始产品实现；
4. M1 通过 focused/full tests 后更新本状态文件；
5. v0.6 完成前不启动 v0.7 正式 Admission。

## Status rules

- `Pending`：尚未开始；
- `In Progress`：正在实现或验证；
- `Passed`：验收证据完整；
- `Failed`：执行完成但未满足验收；
- `Blocked`：外部或环境条件阻止执行；
- `Not Executed`：未运行，不得推断结果。
