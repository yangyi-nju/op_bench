# OpBench v0.7 Dataset Factory 与 Boundary Slice 设计

日期：2026-07-17

状态：设计方向已确认，等待 v0.6 平台完成后实施

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

### P1：Factory Contract 与检索

- 冻结 Candidate、Decision、Admission、Dataset Freeze Schema；
- 实现 B1–B5 keyword packs 和 ghstack-aware 检索；
- 生成 accepted/rejected/deferred Artifact；
- 验证 v0.6 Task/Evaluation 合同可承载新增字段。

### P2：Matched Runtime

- 依次尝试 Matched Wheel、Source-built Wheel、Source Build；
- 生成内容寻址环境资产和 Compatibility Evidence；
- 对 #129154/#144073 给出 verified 或 deprecated 结论。

### P3：Boundary Task 制作与 Admission

- Issue、Hidden/Public、Gold、Manifest、Scope；
- Preflight、Baseline、Gold、P2P 和人工复审；
- 目标 4–6 条 verified Boundary Task；
- 对重复、脆弱、成本过高候选保留拒绝证据。

### P4：Dataset Freeze 与 Validation Cohort

- 生成 cumulative/boundary/precision manifests；
- 运行真实 Codex 新 Task 3-repeat Validation Cohort；
- 重建 Integrity 和 Slice Summary；
- 处理地板/天花板、异常失败和环境漂移。

### P5：发布

- 冻结 Dataset Card、Candidate Report、Admission Evidence 和 Cohort Report；
- 更新 README、CHANGELOG、文档索引和项目状态；
- 确认没有把 Validation Cohort 表述为正式多 Agent 排名。

Matched Runtime 与 Candidate Search 可以并行推进，但正式 Admission 必须使用同一冻结 v0.6 Evaluation Protocol。

## 13. 完成标准

v0.7 只有同时满足以下条件才标记 Completed：

1. Dataset Factory 的状态、Schema、reason code 和 Artifact 可复用；
2. `datasets/pytorch_v0.7/dataset.json` 全部 entry 为 verified 且 evidence hash 有效；
3. Boundary Slice 形成有区分度的 verified 集合，目标 4–6 条，缺口有明确记录；
4. #129154/#144073 均有 verified 或带新增环境证据的 deprecated 结论；
5. 每条新 Task Baseline Reproduced、Gold Resolved、P2P Kept、Test Executed；
6. cumulative/boundary/precision Manifest 和 Summary 可以从原始 Artifact 重建；
7. 真实 Codex Validation Cohort 完整，所有缺失或无效 Attempt 有稳定归因；
8. 全量测试、Schema、Dataset Validation、Integrity 和 `git diff --check` 通过；
9. Dataset Card 报告 taxonomy、来源、环境、局限、污染风险和 rejected funnel；
10. README、CHANGELOG、docs index 与冻结结果一致；
11. 没有降低 Admission 标准以满足数量目标；
12. 不发布本版本未支持的正式跨 Agent 排名或反馈因果结论。

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
| `docs/v0.7/dataset_card.md` | 数据来源、Taxonomy、Admission、限制和统计 |
| `docs/v0.7/validation_report.md` | 真实 Codex Validation Cohort 和失败分析 |
| `runs/v0.7_pr_screening/` | Candidate/Rejected/Decision Artifact |
| `datasets/pytorch_v0.7/dataset.json` | 累计数据集 |
| `datasets/pytorch_v0.7_boundary/dataset.json` | Boundary Slice |
| `datasets/pytorch_v0.7_precision/dataset.json` | 更新后的 Precision Slice |
| `runs/v0.7_validation/` | RunManifest、Attempt、Integrity 和 Summary |

## 16. 后续边界

v0.8 在已验证的平台和数据工厂上增加 Device/API Compatibility Slice，并冻结 Evaluation/Scoring Specification RC。v0.9 再冻结 Dataset、Agent、Prompt、Feedback、Budget、Hardware 和 Repeat，开展正式多 Agent 比较与 Feedback Ablation。
