# OpBench Project State

更新时间：2026-07-23（Asia/Shanghai）

## Current baseline

| 项目 | 当前值 |
| --- | --- |
| 当前稳定版本 | v0.6 Completed |
| 数据集 | `pytorch_v0.5`，17 条 verified task |
| 正式实验 | v0.6 MCP：51/51 valid，35/51 resolved；v0.5 CLI 历史结果：37/51 |
| 当前开发版本 | `opbench-v0.7.0`（Planning） |
| 当前目标 | Dataset Factory、Boundary Slice 与 matched-runtime recovery |
| 当前阶段 | `opbench-v0.6.0` 全部 Must 已通过；准备进入 v0.7 |
| v0.6 产品代码 | 合同、TaskView/Workspace/Action/Session/Evaluation/Artifact、版本化 Runtime Profile、Attempt-owned Local/Docker/Remote Backend、Conformance、Legacy Replay、标准 Codex 进程 Adapter、v1 Orchestrator、公开 Demo 与开发/发布文档均已实现 |
| v0.6 MCP 实验 | 真实全量实验已完成，报告见 `docs/v0.6/experiment_report.md` |

## Current decisions

- v0.6 使用一个版本号和一套完成标准，M1–M7 是内部工程检查点。
- v0.6 不扩数据集，集中建设规范 Agent Evaluation Runtime。
- 真实 Codex 是既有正式评测路径，不是 v0.6 才引入的新能力。
- Fake/Scripted Agent 仅用于确定性测试和故障注入。
- CLI/MCP 共享 Canonical Action Service，不各自实现评测规则。
- Agent Workspace 与 Fresh Evaluator 分离，只以 Frozen Patch 交接。
- v0.6 MCP 全量结果用于平台验证；它与 v0.5 的 Adapter、模型、CLI 和
  Runtime 身份不同，不作因果质量对比。
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
| V06-M6 | Passed | Runtime Conformance、Legacy Replay、标准真实 Codex 与 Resume | 原 M6 freeze 的本地/确定性证据通过；目标恢复后代表性 Remote CPU/CUDA canary 与 17+17+51 精确回放全部通过，详见 `docs/v0.6/m6_verification.md` 的关闭附录 |
| V06-M7 | Passed | 双语 Quickstart、公开 Scripted Demo、开发者指南、代表性 Artifact 与 Release Review | 干净 Python 3.12 venv 中 527 full tests 与 17-task Dataset 通过；25 release-focused tests、Demo resume/Integrity/resource cleanup、合同/JSON/link/wording review 通过，详见 `docs/v0.6/m7_verification.md` |
| V06-RELEASE | Passed | `opbench-v0.6.0` 统一发布 | R-01～R-12、D-01～D-10 与全部 Must 已通过；85/85 精确 Replay、代表性 Remote CPU/CUDA canary、581 full tests、零开放 P0/P1 |
| V06-MCP-EXPERIMENT | Passed | 17 task × 3 repeat 真实 MCP 全量实验 | 51/51 valid；35 resolved、15 F2P failed、1 P2P regression；0 infrastructure-invalid、0 retry；Trace/Integrity/Cleanup 全部通过 |
| REMOTE-CLEANUP | Passed | RemoteDocker timeout/cleanup 收敛到 Attempt-owned exact handles | create/start/command/cleanup 异常注入、精确清理账本和 Remote blocked artifact 均通过 |

## Next actions

1. 在文档中保留 v0.6 的 85-case Replay 与代表性 Runtime canary 冻结 hash，
   `runs/` 只发布三文件 MCP 全量实验最终报告；
2. 按 `docs/v0.7/design.md` 启动 Dataset Factory 与 Boundary Slice 实施；
3. v0.7 正式 Admission 继续执行 verified-only、精确 Runtime 和历史成绩不改写约束；
4. 反馈因果与跨 Agent 正式研究仍留在后续版本，不从 v0.6 平台验证推断结论。

## Status rules

- `Pending`：尚未开始；
- `In Progress`：正在实现或验证；
- `Passed`：验收证据完整；
- `Failed`：执行完成但未满足验收；
- `Blocked`：外部或环境条件阻止执行；
- `Not Executed`：未运行，不得推断结果。
