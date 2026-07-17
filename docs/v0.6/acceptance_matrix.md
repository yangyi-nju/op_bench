# OpBench v0.6 验收矩阵

日期：2026-07-17

状态：验收条件已冻结，尚未执行

目标版本：`opbench-v0.6.0`

## 1. 使用规则

本矩阵是 v0.6 完成判定的唯一逐项清单。设计文档说明“为什么”和“是什么”，实施计划说明“按什么顺序做”，本文件说明“观察到什么才算完成”。

状态只允许：

- `Pending`：尚未开始；
- `In Progress`：正在实现或验证；
- `Passed`：要求和证据全部满足；
- `Failed`：已经执行但不满足；
- `Blocked`：环境或外部条件阻止执行；
- `Not Executed`：未执行，不推断结果。

优先级：

- `P0`：泄漏、宿主破坏、伪造结果或评分身份失效；
- `P1`：核心语义错误、终态错误、不可重建或 Legacy 静默变化；
- `P2`：重要兼容、诊断、覆盖或体验缺陷；
- `P3`：非阻塞改进。

发布规则：所有 `Must` 必须为 `Passed`，没有开放 P0/P1。硬件相关 Must 若仍为 `Blocked`，只能说明平台主体可用，不能将 v0.6 标记为完整发布。

## 2. C — Contracts 与 Manifest

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| C-01 | Must/P1 | RunManifest、TaskView、Action、Session、Evaluation、Event、Result、Integrity 均有显式 schema version | Schema 文件与 valid/invalid tests | Passed |
| C-02 | Must/P1 | wire object 严格拒绝未知版本、缺失身份、非法枚举和非规范值 | Negative tests | Passed |
| C-03 | Must/P1 | canonical JSON 与 SHA-256 可跨进程确定性重建 | Round-trip/hash tests | Passed |
| C-04 | Must/P1 | Dataset、Task、Source、Environment、Agent、Prompt、Policy、Runtime 和 Scoring 身份进入 Manifest | Manifest fixture 与字段断言 | Passed |
| C-05 | Must/P1 | Comparability Key 对影响可比性的配置变化敏感 | Mutation matrix tests | Passed |
| C-06 | Must/P1 | Attempt Identity 由 Cohort、Task、Agent、Repeat 和有效配置确定 | Identity tests | Passed |
| C-07 | Must/P1 | expected task × agent × repeat matrix 在运行前冻结 | Manifest/integrity test | Passed |
| C-08 | Must/P1 | v0.5 Dataset/Task/Environment 可被兼容层读取，默认值确定 | 17-task migration test | Passed |

M1 本地证据（2026-07-17）：

- `python -m unittest tests.test_runtime_canonical ... tests.test_runtime_manifest_cli -v`：60/60 通过；
- `python -m unittest discover tests -v`：229/229 通过；
- `scripts/validate_dataset.py datasets/pytorch_v0.5/dataset.json --require-verified`：17 条 task 通过；
- `scripts/validate_runtime_contract.py configs/examples/v0.6_run_manifest.example.json`：Schema 与 RunManifest 派生身份重建通过；
- 兼容测试显式断言不调用 `subprocess.run` 或 `socket.create_connection`，拒绝未 verified Dataset、artifact traversal/symlink 和 task 根外文件；Schema 拒绝嵌套身份角色互换，语义分派拒绝跨字段非法状态，identity-bearing JSON payload 构造后不可变；示例不包含本机路径或远程 host；M1 未执行 Agent、Docker、SSH 或远程 Runtime 验证。

## 3. T — AgentTaskView 与信息边界

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| T-01 | Must/P0 | Adapter 只接收 AgentTaskView，不接收 FullTaskSpec | Type/API tests | Passed |
| T-02 | Must/P0 | Gold Patch、Hidden Test 内容、Admission 证据和答案来源字段不可见 | Projection/deny tests | Passed |
| T-03 | Must/P0 | PR/Issue 中直接泄漏修复答案的字段按 policy 清理或拒绝 | Sanitization fixtures | Passed |
| T-04 | Must/P1 | AgentTaskView 只使用显式白名单投影，新增 FullTask 字段不会自动透传 | Forward-field test | Passed |
| T-05 | Must/P1 | AgentTaskView、Capability、Budget 和公开 Test 描述通过 Schema 校验 | Schema tests | Passed |
| T-06 | Must/P1 | AgentTaskView 内容进入 Attempt/Manifest identity | Mutation/hash test | Passed |
| T-07 | Must/P0 | Public Artifact 不含 Credential、本机路径、Hidden/Gold 或私有输出 | Artifact scanner fixtures | Passed |

M2 TaskView 本地证据（2026-07-17）：

- Agent Adapter 的完整 task-bearing 输入类型只有 `AgentTaskView` 与其 `task_view` identity；构造器拒绝 `FullTaskSpec`、不匹配 identity 和绕过投影直接构造的敏感 View；
- 投影只复制固定字段，新增 `future_answer` 不进入公开对象；Gold/Hidden/Admission、PR/commit/diff/issue-comment 答案链接、Credential、Unix/Windows 本机绝对路径、Private Output、camelCase 敏感键与非 JSON opaque bytes 均被递归拒绝；
- AgentTaskView、Capability、Budget、Public Test 由独立 Schema 校验；RunManifest 冻结每个完整 AgentTaskView，ExpectedAttempt 记录其 identity，公开内容变化会改变 Comparability Key、Cohort ID 和 Attempt ID；
- 17 条 v0.5 verified task 均可离线生成并扫描 AgentTaskView；未启动 Agent、Docker、SSH、远程 Runtime 或网络探针。

## 4. W — Workspace 与 Patch Freeze

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| W-01 | Must/P1 | 同一 Attempt 的 Action、Test、Diff 和 Freeze 绑定同一 Workspace ID | End-to-end assertion | Passed |
| W-02 | Must/P0 | path traversal、workspace 外路径和 symlink escape 被拒绝 | Negative tests | Passed |
| W-03 | Must/P1 | read/write/apply 支持受控 regular file，拒绝非法类型和 mode | File policy tests | Passed |
| W-04 | Must/P1 | Freeze 开始后不再接受 mutation | Race/state tests | Passed |
| W-05 | Must/P1 | Freeze 收敛 in-flight Action 后只生成一个 final patch | Concurrency tests | Passed |
| W-06 | Must/P1 | Patch 正确表示 add/modify/delete，空补丁有显式语义 | Patch fixtures | Passed |
| W-07 | Must/P0 | Patch 通过 scope、size、mode、symlink 和 binary policy | Policy matrix tests | Passed |
| W-08 | Must/P1 | Frozen Patch 可在干净 base 上严格应用，不使用 fuzz | Clean apply test | Passed |
| W-09 | Must/P0 | Session Result、Patch Artifact、EvaluationSpec 三方 hash 一致 | Identity integration test | Passed |
| W-10 | Must/P1 | Agent Workspace 中未进入 Patch 的缓存、测试改动和未跟踪状态不影响评分 | Contamination test | Passed |

M2 Workspace/Freeze 本地证据（2026-07-17）：

- Workspace identity 由 Source、Base Commit、materialization mode 与完整 policy 确定，不含本机路径；read/write/delete/test binding/diff/freeze 结果引用同一 identity；
- traversal、absolute/backslash、Git pathspec magic、symlink parent/final、directory/FIFO、非法 mode、越界/超限/二进制写入均 fail closed；写入使用同一 mutation authority 和原子 replace；
- Freeze 先关闭新 mutation，再等待并串行收敛 in-flight mutation；并发与重复调用返回同一个不可变 FrozenPatch，失败后 workspace 保持 `freeze_failed` 且不重新开放；
- add/modify/delete/empty、patch/file size、mode、symlink、binary 和 scope fixture 均通过；base snapshots 从 recorded HEAD tree/blob 独立构造并与 root-fd 工作树快照核对，拒绝 staged、assume-unchanged 与 skip-worktree 偏差；patch 在隔离且无全局/仓库 diff 配置的临时 Git stage 中生成，精确路径由独立 parser 复核，再对提交基线快照执行 `git apply --check --index`，不使用 fuzz；
- FrozenPatch 原始 bytes 的 SHA-256 在 SessionResult、PatchArtifact metadata 与 EvaluationSpec 中一致；篡改任一方或原始 bytes 均失败；scope 外 tracked test 改动、未跟踪 cache 和二进制状态不改变 patch bytes/hash；
- M2 核心 focused tests 43/43、包含 Manifest/Schema/Legacy/Action Bridge 的 focused/兼容回归 87/87、全量回归 274/274、17-task Dataset Validation、示例 Manifest 重建与 JSON 语法校验全部通过。

## 5. A — Canonical Actions、Adapter 与 MCP

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| A-01 | Must/P1 | 所有 Action 使用版本化 Request/Observation 和稳定 Error Code | Contract tests | Passed |
| A-02 | Must/P1 | Service 对每次请求重新验证 session、capability、path、selector、budget 和 state | Authority negative tests | Passed |
| A-03 | Must/P1 | 重复 action_id 返回同一结果且不重复副作用 | Idempotency tests | Passed |
| A-04 | Must/P1 | list/search/read 有确定的范围、数量和输出字节限制 | Boundary tests | Passed |
| A-05 | Must/P1 | write/apply_patch 原子执行，失败不留下部分 mutation | Failure injection tests | Passed |
| A-06 | Must/P0 | command_run 只执行 Capability Policy 允许的命令、cwd 和参数形态 | Allow/deny matrix | Passed |
| A-07 | Must/P0 | test_run 只接受 Test Registry 中的 selector | Selector tests | Passed |
| A-08 | Must/P1 | vcs_diff 输出 canonical patch，session_finish 幂等 | Action tests | Passed |
| A-09 | Must/P1 | CLI 与 MCP 调用同一 Canonical Action Service | Dependency/API assertion | Passed |
| A-10 | Must/P1 | 相同 Scripted Sequence 经 CLI/MCP 得到等价 Patch、Error、Budget 和 Event | Conformance test | Passed |
| A-11 | Must/P1 | Codex Adapter 不实现 Workspace、Evaluator 或 Scoring 规则 | Boundary tests/review | Passed |
| A-12 | Must/P1 | 现有 codex_action_bridge 在迁移期保持 Legacy 可用 | Legacy regression | Passed |

M3 证据（2026-07-17）：九个 Action 全部经同一 `CanonicalActionService`
执行；CLI/MCP scripted sequence 的 Observation、Error、Budget、canonical
Patch 与 ActionExchange 审计流精确等价。ActionExchange 是本阶段 transport
conformance 的 action event 证据，M4 再将其绑定到 append-only、hash-chained
`EventRecord` 生命周期流。命令策略按最长 argv prefix 选择后使用
command-specific schema，cwd/path/selector/deadline/budget/state 每次服务端重验；
backend 无法改写授权 command/cwd，也无法把异常、宿主路径或注册测试命令元数据
传给 Adapter。首次 finish 使用一次性控制面 reservation，后续 finish 仍受预算和
幂等约束。标准 Adapter 只获得重新扫描的 `AgentLaunchInput` 与 JSON-only client；
同进程 queue 是 API/data-minimization boundary，不是不可信 Python 的安全沙箱，
不可信 Adapter 必须使用保持相同 JSON 合同的进程/IPC 隔离。现有 v0.5
`codex_action_bridge` 回归保持通过，真实 Codex 到标准 Adapter 的迁移和 canary
保留为 M6 release gate。本阶段 28/28 focused、302/302 full tests、17-task
Dataset、示例 Manifest、tracked JSON 和 diff check 全部通过；最终审查为
Critical 0 / Important 0，且未启动 Agent、Docker、SSH、远程 Runtime 或网络探针。

## 6. S — AttemptSession、Budget、Termination 与 Resume

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| S-01 | Must/P1 | 状态转换只符合冻结状态机 | Transition table tests | Pending |
| S-02 | Must/P1 | 每个 Attempt 恰好一个 terminal event/result | Race/fault tests | Pending |
| S-03 | Must/P1 | Finish、Timeout、Cancel、Exit 和 Error 竞争按固定优先级收敛 | Concurrency matrix | Pending |
| S-04 | Must/P1 | wall-clock、action、test、command、output budget 在服务端记账 | Boundary tests | Pending |
| S-05 | Must/P1 | budget 耗尽后拒绝新工作并保留已完成证据 | Exhaustion tests | Pending |
| S-06 | Must/P1 | provider_error、runtime_error、platform_error 不计为 Agent 修复失败 | Attribution tests | Pending |
| S-07 | Must/P1 | 完成有效 Attempt 后 resume 不重复运行 | Resume test | Pending |
| S-08 | Must/P1 | Task/Agent/Prompt/Policy/Budget/Protocol 变化产生新 Attempt Identity | Mutation matrix | Pending |
| S-09 | Must/P1 | retry append-only，聚合去重但保留审计链 | JSONL/summary tests | Pending |
| S-10 | Must/P1 | 重复 finish/resume 不改变已冻结 patch、terminal 或 summary | Idempotency test | Pending |

## 7. E — Trajectory、Artifact 与 Integrity

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| E-01 | Must/P1 | 必需 lifecycle/action/test/budget/freeze/evaluation/terminal 事件齐全 | Event coverage test | Pending |
| E-02 | Must/P1 | Event sequence 连续且前序 hash chain 可重算 | Integrity tests | Pending |
| E-03 | Must/P1 | 每个公开 Action Request 有且只有一个 Observation | Pairing test | Pending |
| E-04 | Must/P1 | 大输出存独立 Artifact，Event 记录 hash、size、media type | Output fixture | Pending |
| E-05 | Must/P0 | public/private artifact 隔离，公开侧无敏感值 | Redaction tests | Pending |
| E-06 | Must/P1 | Manifest、Task、Source、Policy、Patch、Evaluation 和 Summary 引用闭合 | Graph integrity test | Pending |
| E-07 | Must/P1 | missing、duplicate、unexpected Attempt 和 retry 都被发现 | Matrix mutation tests | Pending |
| E-08 | Must/P1 | 修改或删除任一受保护 Artifact 会使完整性验证失败 | Tamper tests | Pending |
| E-09 | Must/P1 | results.jsonl 与 summary.json 可从原始 Artifact 重建 | Rebuild test | Pending |
| E-10 | Must/P1 | 重建结果与存储结果 canonical-equivalent | Byte/canonical compare | Pending |

## 8. V — Fresh Evaluation 与结果归因

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| V-01 | Must/P0 | Evaluator 使用新 Source/Container，不读取 Agent Workspace | Isolation test | Pending |
| V-02 | Must/P1 | 评分前校验 Base Source 与 Frozen Patch Identity | Negative tests | Pending |
| V-03 | Must/P0 | Hidden/Evaluation-only 资产只在 Agent 终止后注入 | Lifecycle test | Pending |
| V-04 | Must/P1 | Patch 严格应用失败被归因为 invalid_patch | Bad patch test | Pending |
| V-05 | Must/P1 | F2P/P2P 记录 collected/executed/skipped/failed，未执行不能当通过 | Selector execution tests | Pending |
| V-06 | Must/P1 | Bad control unresolved、Gold control resolved | Control matrix | Pending |
| V-07 | Must/P1 | P2P 失败单独归因为 regression | Regression fixture | Pending |
| V-08 | Must/P1 | attempt_validity、agent_terminal、evaluation_outcome 三轴独立 | Result schema/tests | Pending |
| V-09 | Must/P1 | Infrastructure Invalid 不进入 resolved denominator，但保留原始记录 | Aggregate tests | Pending |
| V-10 | Must/P1 | Evaluation Result 和 Summary 绑定 scoring/evaluation protocol version | Hash/version tests | Pending |

## 9. R — Runtime、Replay 与真实 Agent

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| R-01 | Must/P1 | Local CPU、Remote CPU、CUDA Overlay、CUDA Kernel Build 有版本化 Runtime Profile | Profile schemas/configs | Pending |
| R-02 | Must/P1 | Profile 冻结 source loading、image、mount、timeout、resource、network 和 cleanup policy | Manifest assertions | Pending |
| R-03 | Must/P0 | 只管理当前 Attempt 创建并持有 identity 的容器、子进程和文件 | Fixture/integration tests | Pending |
| R-04 | Must/P1 | Local/Remote 对相同 Canonical Sequence 保持 Action、Patch、Result 语义 | Conformance matrix | Pending |
| R-05 | Must/P1 | v0.5 17/17 Baseline Failure Replay | Replay artifact | Pending |
| R-06 | Must/P1 | v0.5 17/17 Gold Success Replay | Replay artifact | Pending |
| R-07 | Must/P1 | v0.5 51/51 Legacy Final Patch Replay | Replay artifact | Pending |
| R-08 | Must/P1 | Replay 差异逐 Task/Environment/Protocol 归因，不改写历史成绩 | Difference report | Pending |
| R-09 | Must/P1 | 至少一个真实 Codex CPU Attempt 经标准 Adapter 完成 read/edit/test/finish | Manifest + trajectory + result | Pending |
| R-10 | Must/P1 | 代表性 Remote CPU、CUDA Overlay、CUDA Kernel Profile 完成 canary | Runtime artifacts | Pending |
| R-11 | Must/P1 | 小规模真实 Codex 批量验证 resume、integrity、summary 和 attribution | Cohort artifact | Pending |
| R-12 | Must/P0 | 验收命令只覆盖 OpBench 合同和当前 Attempt 持有的资源 | Test/command scope review | Pending |

## 10. D — 文档、兼容与发布

| ID | 级别 | 验收要求 | 必需证据 | 状态 |
| --- | --- | --- | --- | --- |
| D-01 | Must/P1 | README、设计、实施计划、CLI help、Schema 和 Artifact layout 一致 | Link/command review | Pending |
| D-02 | Must/P1 | v0.5 默认路径在迁移期兼容，新 Runtime 显式选择 | Legacy CLI tests | Pending |
| D-03 | Must/P1 | 旧 results.jsonl/summary 保持可读，Legacy 与新 Cohort 分离 | Compatibility tests | Pending |
| D-04 | Must/P2 | 提供离线 Scripted smoke、真实 Codex canary、resume、verify、rebuild 示例 | Executed quickstart | Pending |
| D-05 | Must/P1 | 新环境可完成安装、全量测试和 v0.5 Dataset Validation | Clean environment record | Pending |
| D-06 | Must/P1 | 全量测试、focused tests、Schema 和 Artifact validation 全通过 | Verification record | Pending |
| D-07 | Must/P1 | 发布时没有开放 P0/P1 | Issue/finding review | Pending |
| D-08 | Must/P1 | 发布说明不把 v0.5 结果标记为 v0.6，不声称正式排名或反馈因果结论 | Release wording review | Pending |
| D-09 | Must/P2 | 支持矩阵、已知限制和阻塞项准确记录 | Documentation review | Pending |
| D-10 | Must/P1 | `project_state.md`、CHANGELOG 和版本状态同步 | Final diff review | Pending |

## 11. 最终验收记录模板

完成实现后在本节追加一次冻结记录，不覆盖原矩阵：

```text
Platform commit:
Dataset identity:
Action protocol:
Evaluation protocol:
Scoring specification:
Agent/model/adapter:
Runtime profiles:
Full test command/result:
Dataset validation command/result:
Replay 17+17+51 result:
Real Codex canary/cohort:
Artifact root and manifest hash:
Open P0/P1:
Blocked items:
Release decision:
```

在这些字段和所有 Must 证据未完成前，v0.6 状态保持 `In Progress`。
