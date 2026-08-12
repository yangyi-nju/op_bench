# OpBench v0.7 Dataset Factory 与 Boundary Slice 设计

日期：2026-07-17（2026-08-12 补充最终版本边界）

状态：历史设计已完成；50-task 质量版及最终验证均已完成

> 版本边界（2026-08-12）：本文主体记录 2026-07-28 已冻结的旧 v0.7
> Factory/Boundary 发布，相关 Hash 与实验结论保持有效，但它不再是当前 v0.7 的
> 最终完成条件。当前权威目标是
> [50-task 质量扩展发布合同](quality_expansion.md)：14 条历史保留任务加 36 条
> 新增或替换任务，并完成 122 个 fresh logical Attempts。该合同现已完成：
> 50/50 fresh replay、17/17 cohorts、122/122 valid Attempts 和最终发布审计均通过。
> 下文的 21–25 条目标、25/6/8 Dataset 与“Completed”段落应理解为历史里程碑
> 证据，不是当前 50-task release 的规模或实验结果。

## 1. 版本定位

`opbench-v0.7.0` 的目标是验证 v0.6 规范评测平台能否稳定支撑数据生产，并把 OpBench 从单一 precision 维度扩展到第二个真实算子问题维度：boundary。

本版本交付两项相互关联的能力：

1. 可复用的 Candidate→Task Bundle→Admission→Dataset Freeze 数据工厂；
2. 4–6 条高质量 verified Boundary Task，以及两条 matched-runtime Precision 候选的明确结论。

v0.7 只在 v0.6 的版本化 Task、Runtime、Evaluation、Artifact 和 Integrity 合同上构建，不重新定义 Agent Runtime。它会运行 Admission、Gold、Replay 和参考 Agent Validation Cohort，用来证明 Task 有效性与区分度，但不在本版本发布正式多 Agent 排名或反馈因果结论。

## 2. 输入依赖

开始正式 Admission 前，以下 v0.6 能力必须可用：

- FullTaskSpec、AgentTaskView、RunManifest 和 RuntimeProfile；
- Fresh Evaluator 与 collected/executed/skipped counters；
- Canonical Action/Adapter 和真实 Codex 标准路径；
- Frozen Patch、三方 Patch Hash 和 Artifact Integrity；
- Dataset Validation、Summary Rebuild 和 Replay；
- Local/Remote CPU、CUDA Overlay、CUDA Kernel Build Profile。

若 v0.6 尚未达到完成条件，可以进行只读候选检索和初筛，但不冻结 v0.7 Dataset，也不使用临时评分语义完成 Admission。

## 3. Release Contract

### 3.1 数据集目标

以 v0.5 的 17 条 verified task 为历史数据基线：

| 增量 | 目标 | 数据集或切片 |
| --- | ---: | --- |
| 新 Boundary Task | 4–6 条 verified | `pytorch_v0.7_boundary` |
| 恢复 Precision Task | 0–2 条 verified | `pytorch_v0.7_precision` |
| v0.7 累计数据集 | 21–25 条 verified | `pytorch_v0.7` |

4 条 Boundary 是期望的最小有效切片，不是降低 Admission 标准的理由。若最终只有 3 条高质量样本，可以发布 3 条并记录 coverage gap，累计数据集相应为 20–22 条。若候选质量足够，可以超过 6 条，但必须先评估 GPU、Kernel Build 和真实 Agent 验证成本。

Precision Slice 从 v0.5 的 6 条继承。#129154/#144073 若重新通过 Admission，则 Precision Slice 扩为 7–8 条并覆盖 P4；若仍不能建立匹配环境，则继续保持 deprecated，不阻塞 Boundary 主线。

### 3.2 数据发布验证目标

- 每条新 Task 完成 Baseline Failure、Gold Success、F2P/P2P 和 Artifact Integrity；
- 累计、Boundary、Precision 三个 Manifest 都能独立校验和内容寻址；
- 新 Task 运行真实 Codex 参考 Validation Cohort，默认每 Task 3 Repeat；
- inherited slice 可以按需要执行代表性 replay，不把 v0.5 旧 Attempt 拼入新 Cohort；
- Summary 必须通过 dataset × agent × repeat 完整性校验；
- `environment_unavailable`、`infrastructure_invalid` 保留审计记录，但不伪装成有效 Attempt；
- Validation Cohort 用于发现坏 Task、天花板/地板效应和 Runtime 问题，不形成跨 Agent 排行结论。

## 4. Boundary 分类

分类按根因，不按最终表象。边界值触发错误索引、shape 推导或参数验证属于 boundary；低精度计算、dtype promotion 或数值算法导致误差仍属于 precision。

| 子类 | 定义 | 典型症状 | 典型修复 |
| --- | --- | --- | --- |
| **B1. Empty / zero-size** | 空 Tensor、某维为 0、空 reduction 或空 batch | crash、错误 identity、NaN、错误 shape | empty fast path、正确 reduction identity、跳过非法 launch |
| **B2. Scalar / degenerate shape** | 0D scalar、size-1、rank 退化、特殊 broadcast | 索引不存在、错误 squeeze、shape 不一致 | scalar branch、规范化 rank、修正 broadcast 推导 |
| **B3. Integer / size overflow** | numel、stride、offset、index 或 shape product 超过整数范围 | wraparound、负 size、越界、错误 kernel 选择 | 安全整数类型、checked arithmetic、分块计算 |
| **B4. Parameter endpoints** | dim/axis、k、groups、padding、dilation、range 端点或非法组合 | 未抛异常、异常不正确、错误输出 | 参数规范化、范围校验、明确异常 contract |
| **B5. Kernel launch / grid bounds** | grid/block 上限、尾块、超大 shape、设备 launch 边界 | CUDA assert、未处理 tail、launch failure、错误结果 | grid-stride loop、bounds guard、修正 launch 参数 |

### 4.1 与其他维度的边界

- 极端输入导致 `log/exp` NaN/Inf，根因是数值算法不稳定：precision P4；
- 大 shape 的 `numel` 溢出后选择错误 kernel：boundary B3；
- CPU/CUDA 对普通输入行为不同：compatibility，属于 v0.8；
- CPU/CUDA 只在 empty 或最大 grid 边界表现不同：boundary；
- 单纯性能退化、额外同步或慢 kernel：performance，不进入 v0.7。

同一 PR 涉及多个根因时，以 Hidden F2P 直接断言的根因为主分类，在 tags 中记录次要维度。不得为了分类完整性复制同一 Task 到多个正式 Slice。

## 5. 可接受的 Bug 语义

v0.7 接受以下 F2P 形式：

1. Tensor 值、shape、dtype、device 或 layout 错误；
2. segfault、CUDA assert、越界、内部 assert 或确定性 runtime crash；
3. 合法边界输入被错误拒绝；
4. 非法输入未抛异常，或异常类型/消息不符合上游 contract；
5. 边界路径错误 skip、silent fallback 或返回未初始化结果。

异常消息类 Task 必须满足至少一个条件：上游测试明确固定消息、消息属于公开 API contract，或消息会影响调用方判断。只做拼写、标点或风格统一且没有行为回归的 PR 不进入数据集。

## 6. Candidate Search

### 6.1 来源

继续使用已经验证的 ghstack-aware 路径：

1. clone PyTorch mirror，使用 `git log` 在目标时间窗内搜索 commit；
2. 从 commit body 的 `Pull Request resolved` 反查 PR；
3. 使用 `gh pr view` 补齐 issue、文件、diff 和讨论；
4. `gh pr list --state merged` 仅作为非 ghstack PR 的补充；
5. 把 accepted、rejected、deferred 和 rejection reason 固化为筛选 Artifact。

首选 author date 时间窗继续使用 `2024-01-01..2025-04-30`。窗口外候选只有在 Source Snapshot、Image/Wheel 和 ABI/API compatibility 可以独立固定时才允许进入 Admission。

### 6.2 Keyword Packs

| 子类 | 关键词示例 |
| --- | --- |
| B1 | `empty tensor`, `zero size`, `numel == 0`, `empty reduction`, `zero batch` |
| B2 | `scalar`, `0-d`, `zero dimensional`, `degenerate shape`, `rank 0`, `size one` |
| B3 | `overflow`, `int32`, `numel overflow`, `stride overflow`, `index overflow`, `large tensor` |
| B4 | `invalid dim`, `axis bounds`, `k == 0`, `groups`, `padding`, `out of range`, `validation` |
| B5 | `grid limit`, `block limit`, `tail block`, `launch bounds`, `CUDA illegal memory`, `large index` |

每个子类初始检索 2–3 组关键词，候选池目标 3–5 条。子类稀缺时记录 coverage gap，不纳入根因不符的 PR。

### 6.3 自动硬过滤与人工复审

硬过滤：

- author date 落在稳定窗口，或存在明确的新环境冻结方案；
- title 不含 revert/reland，PR 不是纯 refactor/cleanup；
- 修改文件数通常不超过 3，总改动通常为 20–200 行；
- Source 修复与可定位测试同时存在；
- Base Commit 可以获得 sparse snapshot；
- GPU Task 能在现有 V100（sm_70）资源运行，不依赖 H100、FP8 或 flash-attn 3；
- Kernel Build 能通过内容寻址缓存和增量 ccache 完成。

文件数和行数只是筛选阈值。高质量修复因生成代码或必要的双端校验略超阈值时，可以人工记录例外，但不能放宽到大规模重构。

人工复审至少确认：真实用户可观测问题、算子相关性、最小修复范围、潜在泄漏、可复现性、F2P/P2P 设计、硬件成本和与现有 Task 的重复度。

## 7. Dataset Factory

### 7.1 状态机

```text
discovered
→ screened
→ bundled
→ preflight_passed
→ baseline_reproduced
→ gold_resolved
→ reviewed
→ verified
→ frozen
```

任一阶段失败时进入 `rejected` 或 `deprecated`，记录稳定 reason code 和证据引用。状态不能只靠人工修改字符串晋升；Admission Artifact 必须满足对应 Schema 和 Hash。

### 7.2 Task Bundle

每条候选至少包含：

- Task Manifest 和规范化 Issue；
- Base Source Identity 与 Runtime Profile；
- Gold Patch；
- Hidden F2P 和 P2P Test Asset；
- 可选 Public Test；
- Patch Scope；
- Candidate/PR provenance；
- Admission Evidence；
- Operator taxonomy 和 failure contract tags。

### 7.3 自动化与人工判断边界

自动化负责检索、元数据抓取、Patch/Test 适用性、环境预检、Baseline/Gold 执行、Schema/Hash、重复度提示和 Dataset Freeze。

人工判断负责根因分类、Issue 改写质量、答案泄漏、Test 是否真正命中缺陷、P2P 代表性、修复范围合理性和最终 Admission 决定。自动生成的 Task 不得绕过这些判断直接进入 verified Dataset。

## 8. Admission 规则

### 8.1 通用要求

每条 Task 必须满足：

- Base Snapshot 上 F2P 稳定失败，failure signature 与 Issue 一致；
- Gold Patch 后相同 F2P selector 通过；
- P2P 覆盖普通路径和至少一个相邻边界，Gold 后全部通过；
- Test 有明确断言，不以进程 exit 0 代替通过；
- Test 不被 skip、xfail 或 capability guard 静默绕过；
- Patch Scope 足够小，真实 Agent 能在 Task Budget 内定位；
- CPU/GPU 资源能由声明的 Runtime Profile 稳定满足；
- Baseline、Gold、Task、Source、Environment 和 Test identity 完整闭合。

### 8.2 Boundary 附加要求

- 不使用真实 OOM、随机 allocator 状态或不可控 wall-clock timeout 作为 F2P；
- 超大 shape 优先使用 meta/fake tensor、mocked launch 参数、small-index surrogate 或低内存构造；
- crash Task 必须在隔离 Runtime 中稳定终止，不能影响后续 Attempt；
- exception Task 同时断言异常类型，只有 contract 要求时才断言完整消息；
- CUDA Test 显式同步，避免异步错误落到后续 P2P；
- surrogate 必须证明与真实缺陷共享同一根因和修复路径。

### 8.3 测试执行真实性

Admission Evidence 必须记录：

1. 实际 collected/executed/skipped Test 数；
2. 声明的 F2P/P2P 未执行时结果为 `test_not_executed`；
3. Baseline 的结构化 failure signature；
4. Gold 在同一 selector、Runtime 和 Source Identity 上的通过结果；
5. 每一阶段的 Manifest、Patch、Evaluation 和 Artifact hash。

## 9. Matched-Wheel / Source-Build 副线

### 9.1 问题

Python Overlay 只有在 Source Snapshot 的 Python API 与容器内 Wheel/Runtime 相容时才可信。#129154 和 #144073 的 Base Commit 与现有 torch 2.6 wheel 存在 API 代差，测试会在目标 bug 断言前因无关 AttributeError 或 compile API 不匹配失败。修改 Hidden Test、猴子补 API 或跳过无关错误会掩盖环境问题，因此不可作为恢复手段。

### 9.2 环境选择顺序

1. **Matched wheel**：优先使用与 Base Commit 足够接近且 ABI/API 匹配的官方 Wheel；
2. **Source-built wheel**：从固定 Snapshot 构建，缓存为内容寻址资产；
3. **Source build**：只有 Wheel 无法覆盖 compile/kernel 路径时才使用完整 Source Build Runtime。

### 9.3 Compatibility Evidence

至少记录：

- Source Commit SHA；
- Wheel/Build Artifact digest；
- `torch.__version__`、CUDA Runtime 和 Python ABI；
- Source Loading Mode；
- 目标模块确实来自 Snapshot 的证明；
- 最小 compatibility check 及结果；
- Build Flags、GPU Arch 和 ccache key。

Compatibility Check 只证明环境能够加载目标路径，不能代替 F2P。环境可用后仍需完整 Baseline/Gold Admission。

### 9.4 恢复候选

| Task | Precision 子类 | 当前阻塞 | v0.7 目标 |
| --- | :---: | --- | --- |
| `129154_exp_decomp_numerics` | P4 | CUDA refs API 与 torch 2.6 wheel 不匹配 | matched CUDA wheel 或 source-built runtime |
| `144073_vector_norm_scalar_overflow` | P4 | CPU compile/refs API 与 torch 2.6 wheel 不匹配 | matched CPU compile wheel 或 source build |

恢复成功后保留原 Task ID 和 PR provenance，生成新的 Admission Evidence，并把状态从 deprecated 改为 verified。失败则保留 deprecated 和本轮新增诊断，不反复修改测试语义。

## 10. 数据模型与版本

Boundary Task 使用现有 operator 字段：

```json
{
  "operator": {
    "problem_dimension": "boundary",
    "problem_subclass": "B3",
    "problem_type": "numel-integer-overflow"
  },
  "tags": ["failure_contract:crash-oob"]
}
```

约束：

- `problem_dimension`：新增 Task 为 `boundary`，恢复 Task 保持 `precision`；
- `problem_subclass`：Boundary 使用 `B1..B5`，Precision 使用 `P1..P5`；
- `problem_type`：使用稳定根因名称，不直接复制 PR Title；
- `failure_contract`：`wrong-result`、`exception`、`crash-oob` 或 `silent-acceptance`；
- Schema 按 dimension 校验 subclass 前缀；
- Dataset、Task、Source、Runtime、Evaluation 和 Admission 分别带版本与内容 hash；
- 历史未分类 Task 不强制回填，报告时进入 `unclassified`。

## 11. 评测与报告

沿用 v0.5 八维指标，并使用 v0.6 三轴结果和完整性合同：

1. resolved rate；
2. patch conciseness；
3. pass-to-pass kept rate；
4. strict resolved rate；
5. regression rate；
6. tier-weighted score；
7. per-dimension/subclass/type breakdown；
8. median evaluator runtime。

v0.7 报告额外给出：

- B1–B5 的 Task/Attempt/Resolved 分布；
- failure contract 分组；
- Boundary 新增与 restored Precision 分开统计；
- Admission rejection funnel 和稳定 reason；
- Runtime、Agent、Task、Evaluator、Infrastructure Failure 分解；
- inherited slice replay 与新 Validation Cohort 分离；
- environment retry/raw record 与 logical Attempt 完整性；
- Dataset Factory 的人工时间、环境时间和每条 verified Task 成本。

所有汇总从 RunManifest 和原始 Attempt Artifact 重建。Agent、Prompt、Budget、Runtime 或 Scoring 不同的结果不能拼接为一个 Cohort。

## 12. 实施阶段

### P1：Factory Contract 与检索（Passed，2026-07-26）

- 冻结 Candidate、Decision、Admission、Dataset Freeze Schema；
- 实现 B1–B5 keyword packs 和 ghstack-aware 检索；
- 生成 accepted/rejected/deferred Artifact；
- 验证 v0.6 Task/Evaluation 合同可承载新增字段。

P1 采用先离线冻结合同与 synthetic fixture 的实施顺序；`git_log` 和
`github_pr_list` producer 值已保留在合同中，真实 ghstack-aware 检索随 P2/P3
候选工作启用。已交付四类合同、四个 Schema、B1–B5 packs、确定性筛选、
证据门 Admission、不可变 Artifact Store、synthetic Freeze 和三个 CLI。
9 条离线候选结果为 5 accepted、2 deferred、2 rejected，两次运行逐字节一致。
验证通过 73 Factory tests、45 compatibility tests、741 full tests、17-task
v0.5 Dataset 与全部 Schema/compileall/diff checks。P1 未使用 live network、
真实 Agent、Docker、SSH、CUDA 或远程 Runtime，未发布正式 v0.7 Dataset。

详细设计、执行计划和规则证据：

- `docs/superpowers/specs/2026-07-26-v0.7-p1-factory-contract-design.md`
- `docs/superpowers/plans/2026-07-26-v0.7-p1-factory-contract.md`
- `docs/v0.7/candidate_search.md`

### P2：Matched Runtime（Passed，2026-07-27）

- 依次尝试 Matched Wheel、Source-built Wheel、Source Build；
- 生成内容寻址环境资产和 Compatibility Evidence；
- 对 #129154/#144073 给出 verified 或 deprecated 结论。

P2 实现了严格 Compatibility Evidence 合同、Schema、probe/validation CLI
和 fail-closed promotion gate，并在远端构建两个 digest-pinned matched-wheel
镜像。#129154 使用 torch 2.4.0+cu124；#144073 使用 torch 2.7.0+cpu、
匹配的 torchvision 0.22.0+cpu companion，以及同 Base Commit 的单个
Inductor 测试辅助 overlay。两个任务均通过 6/6 compatibility checks、
preflight、Baseline Reproduced 与 Gold Resolved，并从 deprecated 原子提升为
verified。两者都不需要 source-build fallback。

P2 恢复了两条 Precision P4 Task，但按 Release Contract 不在本阶段提前修改
正式 Dataset Manifest；Precision/Cumulative Slice 的纳入由 P4 Freeze 完成。
本轮目标 2/2 恢复，仓库仍有 7 条不在 P2 范围内的历史 deprecated Task。
最终验证通过 47 项 matched-runtime focused tests 与 793 项全仓测试。
详细复现步骤、镜像/Artifact digest、Admission 计数和诊断见
`docs/v0.7/setup_matched_runtime.md`。

### P3：Boundary Task 制作与 Admission（Passed，2026-07-27）

- Issue、Hidden/Public、Gold、Manifest、Scope；
- Preflight、Baseline、Gold、P2P 和人工复审；
- 目标 4–6 条 verified Boundary Task；
- 对重复、脆弱、成本过高候选保留拒绝证据。

P3 冻结了 10 条真实 PyTorch 候选，自动筛选结果为 6 accepted、2 deferred、
2 rejected；人工复核把随后被上游 revert 的 #147433 从 deferred 判为 rejected，
#127448 保持 deferred。最终 6 条任务全部完成 6/6 Compatibility checks、
Baseline F2P 0/1/P2P 1/1、Gold F2P 1/1/P2P 1/1、人工复核和 8 阶段
Factory Admission，覆盖 B1–B5 且均为 verified。

四条任务使用 digest-pinned matched wheel，#143792 与 #147352 使用 exact Base
Commit CPU full-source build。B3 使用小 tensor 加极端整数检查 checked arithmetic；
B5 使用调用同一生产 predicate 和 Gold 路径的低内存 surrogate，不申请超大
Tensor。完整候选漏斗、Runtime/Source digest、Admission 和日志边界见
`docs/v0.7/boundary_tasks.md`。

### P4：Dataset Freeze 与 Validation Cohort（Passed，2026-07-28）

- 生成 cumulative/boundary/precision manifests；
- 运行真实 Codex 新 Task 3-repeat Validation Cohort；
- 重建 Integrity 和 Slice Summary；
- 处理地板/天花板、异常失败和环境漂移。

P4 冻结了 25-task cumulative、6-task Boundary 与 8-task Precision Dataset。
三个 generated Dataset hash 分别为
`sha256:4d7bde25e747bcc041aa5105ce5ce881a3f1e9fe2a7545667cdbc2c14d85064a`、
`sha256:810a9cc85c576f44edd2672197ab83b7dfee7f674e597c76c78050bd119d606a`
和
`sha256:65818466a02e99466386cb8e038dc4da59d91dcb3bea7b83c8901d31a96aa8eb`。
Boundary Freeze、Release Composition、三份 Summary 与公开验证报告均在独立
临时目录重建并逐字节一致。

真实 Codex Validation Cohort 使用 `gpt-5.6-sol`、
`codex-cli 0.146.0-alpha.3.1` 和 `codex_mcp_canonical`，覆盖 5 个 Runtime
Profile、6 个 Task、18 个 Attempt。18/18 选择均 valid，结果为 14 resolved、
3 f2p_failed、1 no_patch；17 finished、1 timeout；18 条 trace 完整且
MCP protocol error 为 0。五个 root 的 fresh Integrity 均与持久化结果一致，
每个 14/14 checks 通过，资源 ownership/cleanup 全部通过。

预验收审计发现并修复了远端命令开始前 SSH 断连被误归因为逻辑 selector 失败的
问题；最终 source cohort 从统一修复快照完整重跑，6/6 resolved 且无 transport
痕迹。被污染的预验收 root 未进入正式报告。Accepted cohort 全部来自
`retry_index=1`，公开 retry 数为 0。

P4 聚焦测试 68/68 通过；全仓验证 865/865 通过（1183.222 秒），同时通过
compileall、四个 verified Dataset、全部 tracked JSON、确定性重建和 diff gate。
完整结果与限制见 `docs/v0.7/validation_report.md`。该结果是平台与 Task 验证，
保持 non-leaderboard、非反馈因果范围。

### P5：发布（Passed，2026-07-28）

- 冻结 Dataset Card、Candidate Report、Admission Evidence 和 Cohort Report；
- 更新 README、CHANGELOG、文档索引和项目状态；
- 确认没有把 Validation Cohort 表述为正式多 Agent 排名。

P5 发布 `opbench-v0.7.0`，Dataset Card 与中英文入口均绑定三份冻结
Dataset 和 P4 报告。最终 release gate 重建 P3 screening、P4 Boundary Freeze、
三份 Dataset release、validation contract 与公开验证报告，全部逐字节一致；
四份 verified Dataset、Schema/JSON、compileall、链接、安全文本和 diff gate
全部通过。

发布身份冻结为：

- cumulative：
  `sha256:4d7bde25e747bcc041aa5105ce5ce881a3f1e9fe2a7545667cdbc2c14d85064a`；
- Boundary：
  `sha256:810a9cc85c576f44edd2672197ab83b7dfee7f674e597c76c78050bd119d606a`；
- Precision：
  `sha256:65818466a02e99466386cb8e038dc4da59d91dcb3bea7b83c8901d31a96aa8eb`。

真实 Codex cohort 为 18/18 valid、14 resolved、3 F2P failed、1 no patch、
0 accepted-cohort retries。最终全仓验证为 868/868 tests。P3 的 exact-source
路径同时固化了 source-build timeout correction，并把 CPU 与 CUDA build
commands 分开选择，防止 CPU source profile 继承 CUDA 构建参数。这些结果只证明
发布 Artifact、平台路径和 Task 的可复现性，保持 **non-leaderboard**、
非跨 Agent 排名和非反馈因果范围。

Matched Runtime 与 Candidate Search 可以并行推进，但正式 Admission 必须使用同一冻结 v0.6 Evaluation Protocol。

## 13. 历史 P1–P5 完成标准

2026-07-28 的 25-task 历史里程碑满足以下条件；最终 50-task release 的 18 项
完成标准及实际 Agent 结果见 [`validation_report.md`](validation_report.md)：

1. [x] Dataset Factory 的状态、Schema、reason code 和 Artifact 可复用；
2. [x] `datasets/pytorch_v0.7/dataset.json` 全部 entry 为 verified 且 evidence hash 有效；
3. [x] Boundary Slice 为 6 条 verified Task，完整覆盖 B1–B5；
4. [x] #129154/#144073 均恢复为 verified，并进入 Precision/Cumulative Freeze；
5. [x] 每条新 Task Baseline Reproduced、Gold Resolved、P2P Kept、Test Executed；
6. [x] cumulative/boundary/precision Manifest 和 Summary 可从冻结输入逐字节重建；
7. [x] 真实 Codex Validation Cohort 18/18 valid，0 accepted-cohort retries；
8. [x] 868/868 全量测试、Schema、Dataset Validation、Integrity、JSON、
   compileall 和 `git diff --check` 通过；
9. [x] Dataset Card 已报告 taxonomy、来源、环境、局限、污染风险和 rejected funnel；
10. [x] README、CHANGELOG、docs index 与冻结结果一致；
11. [x] 没有降低 Admission 标准以满足数量目标；
12. [x] 明确保持 non-leaderboard，不发布正式跨 Agent 排名或反馈因果结论。

## 14. 风险与降级

| 风险 | 控制与降级 |
| --- | --- |
| 候选稀缺 | 保证根因和 F2P 质量，允许 B1–B5 空缺，不用非 Boundary PR 填充 |
| 大 Shape 不可复现 | 优先 meta/fake/surrogate；只能依赖超量内存或超长时间时拒绝 |
| Crash 污染环境 | Attempt 隔离、严格超时和 Runtime-owned cleanup；会影响后续任务则拒绝 |
| Matched Runtime 成本过高 | 不阻塞 Boundary；不可稳定缓存的 Nightly/Build 不进入 Registry |
| Validation 成本增长 | CPU 合理并发、GPU/Kernel 串行、相同 Cohort Resume，不拼接旧结果 |
| 数据泄漏 | AgentTaskView 白名单、来源字段审查、Public Artifact 扫描 |
| 协议漂移 | Dataset Freeze 绑定 v0.6 Protocol/Scoring；变化时创建新 Cohort |

## 15. 计划产物

| 路径 | 用途 |
| --- | --- |
| `docs/v0.7/design.md` | 本设计 |
| `docs/v0.7/candidate_search.md` | Keyword Pack、筛选规则和候选报告 |
| `docs/v0.7/setup_matched_runtime.md` | Matched Wheel/Source Build 制作与验证 |
| `docs/v0.7/boundary_tasks.md` | P3 候选漏斗、任务、Runtime 与 Admission 报告 |
| `docs/v0.7/dataset_card.md` | 数据来源、Taxonomy、Admission、限制和统计 |
| `docs/v0.7/validation_report.md` | 真实 Codex Validation Cohort 和失败分析 |
| `runs/v0.7_pr_screening/` | Candidate/Rejected/Decision Artifact |
| `datasets/pytorch_v0.7/dataset.json` | 累计数据集 |
| `datasets/pytorch_v0.7_boundary/dataset.json` | Boundary Slice |
| `datasets/pytorch_v0.7_precision/dataset.json` | 更新后的 Precision Slice |
| `datasets/pytorch_v0.7_device/dataset.json` | CUDA Device 派生切片 |
| `runs/v0.7_validation/` | RunManifest、Attempt、Integrity 和 Summary |

## 16. 后续边界

原定 v0.8 的 Device/API 数据扩充已并入当前 v0.7 质量扩展，并统一使用
`docs/v0.7/quality_expansion.md` 的多轴 Taxonomy、Prompt、复杂度和 Admission
合同。完整 Agent 输出/隐藏推理 trace、正式跨 Agent 比较与 Feedback Ablation
仍留给后续研究版本。
