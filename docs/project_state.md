# OpBench Project State

更新时间：2026-08-11（Asia/Shanghai）

## Current baseline

| 项目 | 当前值 |
| --- | --- |
| 当前稳定基线 | `opbench-v0.7.0` 50-task 质量版已完成；历史 25-task 冻结仍可复现 |
| 当前开发版本 | v0.7 50-task 质量扩展（Completed） |
| 数据集目标 | 已冻结恰好 50 条：14 retained historical + 21 new + 15 replacement；50/50 Admission verified |
| 实验目标 | 36 条新增/替换各 3 次、14 条 retained 各 1 次，共 122 个 fresh logical Attempts；旧 18-attempt Boundary Validation 仅作历史证据 |
| 当前目标 | v0.7 已完成质量优先数据扩充、真实 Admission、Agent 实验和最终发布审计；原 v0.8 Device/API 扩充已合并回本版本 |
| 当前阶段 | p7 重新准入、p8 36/36 Admission、Prompt 双重审查、p9 freeze、50/50 fresh replay、17/17 cohorts 与 122/122 valid Agent Attempts 均已完成；报告三件套可确定性重建 |
| 50-task 最终组成 | CPU 35 / CUDA 15；hard 46 / medium 4；contract family 为 API behavior 27、efficiency/safety 6、result 6、tensor metadata 4、gradient 2、mutation/state 5；Boundary/Precision/Device 派生切片为 31/5/15 |
| 当前本地验证 | p9 release、四份 Dataset、50/50 replay、122-attempt contract/runner/report validator、确定性报告重建、compile/JSON/diff 和公开树隐私扫描均通过；最终全量回归 1101/1101 通过（1260.411 秒） |
| v0.7 P1 产品代码 | Candidate/Decision/FactoryAdmission/DatasetFreeze 合同、B1–B5 taxonomy、确定性筛选、证据门状态机、不可变 Artifact Store、Validator/Freeze/Screening CLI 与 synthetic fixtures 已实现 |
| v0.7 P2 产品代码 | Matched-runtime Compatibility 合同/Schema、真实 probe/validation CLI、fail-closed promotion、两个 digest-pinned wheel 镜像和 task-local compatibility/Admission evidence 已实现；#129154/#144073 均恢复为 verified |
| v0.7 P3 数据资产 | 10 条真实候选的确定性漏斗、6 条覆盖 B1–B5 的 verified Boundary Task、6 份 Compatibility/Admission/review 和 6 条完整 8 阶段 Factory chain 已冻结 |
| v0.7 P4 数据与验证 | 25-task cumulative、6-task Boundary、8-task Precision Dataset 已内容寻址冻结；真实 Codex 18/18 valid，14 resolved、3 F2P failed、1 no patch，18 trace/Integrity/Cleanup 全部通过；报告为 non-leaderboard |
| v0.7 P5 发布 | Dataset Card、双语 README/docs index、设计/路线图/状态/CHANGELOG 已同步；generated Artifact 全部逐字节重建；868/868 tests、compileall、JSON、Dataset、link、安全文本与 diff gates 通过 |
| v0.7 Dataset hashes | cumulative `sha256:3695622dd2619a760d510ef49e0a9dbff637c98790ad3263c521bae8e99c9518`；Boundary `sha256:2890f5937a5b2c7f5a12c870fc9cc550f0f16ff065467245ecf65223b5976a01`；Precision `sha256:508ec6928d94c159499ae84bf4f37e594b2bdafdef89b04369f481deeddb2c8d`；Device `sha256:b598fdfe94af9921132b147ab693477de8fb360dabe7e5f611792e5f38c0f138` |
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
- `opbench-v0.7.0` 的 Validation Cohort 是 non-leaderboard、非反馈因果的
  Task/平台验证，不与历史 Adapter/模型结果作因果质量比较。
- exact-source Runtime 已修正 source-build timeout，并按 CPU/CUDA profile
  分离 build commands；二者都是发布复现合同的一部分。
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
| V07-P1 | Passed | Factory Contract、Boundary taxonomy 与离线确定性筛选/Freeze | 73 Factory tests、45 compatibility tests、741 full tests 通过；4 个 Factory Schema 可解析；v0.5 17-task verified Dataset 有效；9 条 synthetic fixture 为 5 accepted / 2 deferred / 2 rejected，两次输出逐字节一致；未使用网络、真实 Agent、Docker、SSH、CUDA 或远程 Runtime，未发布正式 v0.7 Dataset |
| V07-P2 | Passed | Matched Runtime、Compatibility Evidence 与两条 Precision P4 恢复 | 两个官方 torch wheel、一个 torchvision companion 和两个实测 image ID 已冻结；目标任务 2/2 恢复为 verified，仓库仍有 7 条不在本阶段范围内的历史 deprecated Task；#129154/#144073 各 6/6 compatibility checks、Baseline F2P 0/1/P2P 1/1、Gold F2P 1/1/P2P 1/1；47 focused、793 full tests 通过；未执行 source-build fallback，正式 Dataset 纳入留到 P4 |
| V07-P3 | Passed | 真实 Boundary Task 制作、Compatibility、Admission 与 Factory Promotion | 10 条候选为 6 accepted / 2 deferred / 2 rejected，#147433 人工判为上游 revert；6 条任务覆盖 B1–B5，合计 36/36 compatibility checks，逐条 Baseline F2P 0/1/P2P 1/1、Gold F2P 1/1/P2P 1/1，人工 review 与 8 阶段 Factory chain 均 verified；90 focused、53 final focused、818 full tests 通过，详见 `docs/v0.7/boundary_tasks.md` |
| V07-P4 | Passed | Dataset Freeze、Release Composition 与真实 Codex Validation Cohort | cumulative/boundary/precision 为 25/6/8 verified Tasks；5 cohorts、18/18 valid、14 resolved、3 F2P failed、1 no patch、0 accepted-cohort retry；五个 root fresh Integrity 14/14、resource ownership/cleanup 和 18 traces 全部通过；68 focused、865 full tests、确定性重建、compile/JSON/Dataset/diff gates 通过，详见 `docs/v0.7/validation_report.md` |
| V07-P5 | Passed | Dataset Card、双语入口、完成记录与可复现发布门 | `opbench-v0.7.0` 的三份 Dataset hash、18/18 cohort 与 non-leaderboard 边界已冻结；全部 generated Artifact 逐字节重建；最终 868/868 tests 及 compile/JSON/Dataset/link/safety/diff gates 通过 |
| V07-LEGACY-RELEASE | Passed | 2026-07-28 历史 v0.7 freeze | P1–P5 全部 Passed；25/6/8 verified Dataset、B1–B5、P1–P5、真实 Codex Validation 和公开文档合同一致 |
| V07-QX-READMISSION | Passed | 旧 25 条任务质量重新准入 | 14 retained、10 retired、1 deferred，证据见 `factory/v0.7/p7/historical_readmission.json` |
| V07-QX-BUILD | Passed | 36 条新增/替换任务质量构建与 Admission | `factory/v0.7/p8/accepted_tasks.json` 为 36/36 verified；29 条 staging Task 的真实 Runtime F2P/P2P、Prompt evidence、Admission 和 Registry binding 已原子提升并完成审计 |
| V07-QX-FREEZE | Passed | 50-task 质量版与四个 Dataset view 冻结 | 14 retained + 21 new + 15 replacement；50/50 verified；累计/Boundary/Precision/Device 为 50/31/5/15，p9 release 可确定性重建 |
| V07-QX-REPLAY | Passed | 50-task fresh replay | 50/50 verified；bundle/registry/release identity 全部重建一致 |
| V07-QX-EXPERIMENT | Passed | 50-task fresh Agent 实验 | 17/17 cohorts、122/122 valid logical Attempts、122/122 完整 MCP traces；42 resolved、52 F2P failed、28 invalid patch；8 条无效 retry 历史保留且不进入 Agent 分母 |
| V07-QX-RELEASE | Passed | 50-task v0.7 最终发布 | replay、122 Attempts、Integrity、资源、隐私、确定性报告、文档与最终 Hash 审计通过；1101/1101 全量测试通过；结果为 non-leaderboard |

## Next actions

1. 基于冻结的 v0.7 报告分析 Agent 失败模式与 taxonomy 覆盖；
2. 保持 Dataset/Prompt/Runtime/Evaluator/Scoring 身份不可变，变更时创建新实验；
3. 正式跨 Agent 与 Feedback Ablation 继续留在后续研究版本。

## Status rules

- `Pending`：尚未开始；
- `In Progress`：正在实现或验证；
- `Passed`：验收证据完整；
- `Failed`：执行完成但未满足验收；
- `Blocked`：外部或环境条件阻止执行；
- `Not Executed`：未运行，不得推断结果。
