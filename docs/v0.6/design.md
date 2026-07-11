# OpBench v0.6 设计方案

日期：2026-07-11

状态：讨论结论已确认，待实现

## 0. 已确认决策

v0.6 采用“双轨增量、统一发布”的版本结构：

1. 主线新增**边界条件问题**维度，接受错误 tensor 结果、crash/越界、缺少异常、异常类型或消息错误，以及非法输入被静默接受等可观测 bug。
2. 副线建设 matched-wheel/source-build 环境，允许恢复 v0.5 deprecated 的 precision task #129154 和 #144073。恢复成功后进入 v0.6 累计数据集和 precision slice，不计入 boundary slice。
3. 冻结数据集后使用同一 Codex 版本、同一 remote execution policy 对累计清单全量运行 3-repeat。旧版本实验结果只用于纵向参考，不与新结果拼接计分。

v0.6 不以填满分类表为目标。候选数量是检索目标，正式数据集只接收 admission evidence 完整、baseline 可稳定复现、gold 可稳定解决的 task。

## 1. 版本定位

OpBench 的最终产物是一个按版本冻结、可独立复现的真实算子问题 benchmark。每个版本需要同时交付：

- verified 累计数据集 manifest；
- 可复用的问题维度 slice；
- 每条 task 的 source/environment/test/gold admission evidence；
- 同一 agent/runtime policy 下的完整实验 artifact；
- 可从原始 attempt 重新计算的结构化 summary 和实验报告。

v0.5 证明了 precision 维度、remote replay、3-repeat 和完整性校验可以形成发布闭环。v0.6 的核心不是简单增加 task 数，而是验证这套方法能否扩展到第二个问题维度，同时解决“source commit 与预装 wheel API 不匹配”这一类环境真实性问题。

## 2. Release Contract

### 2.1 数据集目标

以 v0.5 的 17 条 verified task 为基线：

| 增量 | 目标 | 进入的 slice |
| --- | ---: | --- |
| 新 boundary task | 4-6 条 verified | `pytorch_v0.6_boundary` |
| 恢复 precision task | 0-2 条 verified | `pytorch_v0.6_precision` |
| v0.6 累计数据集 | 21-25 条 verified | `pytorch_v0.6` |

其中 4 条 boundary 是期望的最小有效切片，不是降低 admission 标准的理由。如果最终只有 3 条高质量样本，则质量降级方案是发布 3 条、累计数据集为 20-22 条，并记录 coverage gap；如果候选质量足够，可以超过 6 条，但必须评估全量实验成本。

precision slice 从 v0.5 的 6 条继承。#129154/#144073 若重新 admission verified，则 precision slice 扩为 7-8 条，并首次覆盖 P4；若仍无法建立匹配环境，则继续 deprecated，不阻塞 boundary 主线发布。

### 2.2 实验目标

- 累计数据集目标：21-25 tasks x 3 repeat，即 63-75 个有效 attempt；质量降级方案最低为 20 tasks / 60 attempts。
- Boundary slice 目标：4-6 tasks x 3 repeat，即 12-18 个有效 attempt；质量降级方案为 3 tasks / 9 attempts。
- Precision slice：6-8 tasks x 3 repeat，即 18-24 个有效 attempt。
- 所有 summary 必须通过 dataset x agent x repeat 完整性硬校验。
- `environment_unavailable/environment_error` 保留原始审计记录，但不占有效 attempt；timeout、runner error 和 agent patch 行为按既定终态规则处理。

## 3. Boundary 分类

分类按**根因**，不按最终表象。边界值触发错误索引、shape 推导或参数验证属于 boundary；低精度计算、dtype promotion 或数值算法导致误差仍属于 precision。

| 子类 | 定义 | 典型症状 | 典型修复 |
| --- | --- | --- | --- |
| **B1. Empty / zero-size** | 空 tensor、某维为 0、空 reduction 或空 batch | crash、错误 identity、NaN、错误 shape | empty fast path、正确 reduction identity、跳过非法 launch |
| **B2. Scalar / degenerate shape** | 0D scalar、size-1、rank 退化、特殊 broadcast | 索引不存在、错误 squeeze、shape 不一致 | scalar branch、规范化 rank、修 broadcast 推导 |
| **B3. Integer / size overflow** | numel、stride、offset、index、shape product 超过整数范围 | wraparound、负 size、越界读写、错误 kernel 选择 | 使用安全整数类型、checked arithmetic、分块计算 |
| **B4. Parameter endpoints** | dim/axis、k、groups、padding、dilation、range 端点或非法组合 | 未抛异常、异常不正确、错误输出 | 参数规范化、范围校验、明确异常 contract |
| **B5. Kernel launch / grid bounds** | grid/block 上限、尾块、超大 shape、设备 launch 边界 | CUDA assert、未处理 tail、launch failure、错误结果 | grid-stride loop、bounds guard、launch 参数修正 |

### 3.1 与其他维度的边界

- 极端输入导致 `log/exp` NaN/Inf，根因是数值算法不稳定：precision P4。
- 大 shape 的 `numel` 溢出后选择错误 kernel：boundary B3。
- CPU/CUDA 对普通输入行为不同：后续 device/API compatibility。
- CPU/CUDA 只在 empty 或最大 grid 边界表现不同：本版本 boundary。
- 单纯性能退化、额外同步或慢 kernel：performance，v0.6 不接收。

同一 PR 同时涉及多个根因时，task 以 hidden F2P 断言直接验证的根因为主分类，并在 tags 中记录次要维度。不要为了分类完整性复制同一 task 到多个正式 slice。

## 4. 可接受的 Bug 语义

v0.6 接受以下 fail-to-pass 形式：

1. tensor 值、shape、dtype、device 或 layout 错误；
2. segfault、CUDA assert、越界、内部 assert 或确定性 runtime crash；
3. 合法边界输入错误抛异常；
4. 非法输入未抛异常，或异常类型/消息不符合上游 contract；
5. 边界路径错误 skip、silent fallback 或返回未初始化结果。

异常消息类 task 必须满足至少一个条件：上游测试明确固定该消息、消息属于公开 API contract，或错误消息会影响调用方判断。只做拼写、标点或风格统一且没有行为回归的 PR 仍视为 cleanup，不进入数据集。

## 5. Candidate Search

### 5.1 来源

继续使用 v0.5 已验证的 ghstack-aware 路径：

1. clone PyTorch mirror，使用 `git log` 在目标时间窗内搜索 commit；
2. 从 commit body 的 `Pull Request resolved` 反查 PR；
3. `gh pr view` 补齐 issue、文件、diff 和讨论；
4. `gh pr list --state merged` 仅作为非 ghstack PR 的补充。

首选 author date 时间窗继续使用 `2024-01-01..2025-04-30`。窗口外候选只有在 source snapshot、镜像和 wheel compatibility 可以独立固定时才允许进入 admission，避免 nightly ABI/API 漂移。

### 5.2 Keyword packs

| 子类 | 关键词示例 |
| --- | --- |
| B1 | `empty tensor`, `zero size`, `numel == 0`, `empty reduction`, `zero batch` |
| B2 | `scalar`, `0-d`, `zero dimensional`, `degenerate shape`, `rank 0`, `size one` |
| B3 | `overflow`, `int32`, `numel overflow`, `stride overflow`, `index overflow`, `large tensor` |
| B4 | `invalid dim`, `axis bounds`, `k == 0`, `groups`, `padding`, `out of range`, `validation` |
| B5 | `grid limit`, `block limit`, `tail block`, `launch bounds`, `CUDA illegal memory`, `large index` |

每个子类初始检索 2-3 组关键词，候选池目标 3-5 条。筛选记录仍保留 accepted/rejected 及具体理由，不因某个子类稀缺而纳入非 boundary PR。

### 5.3 自动硬过滤

以 v0.5 规则为基础：

- author date 落在稳定窗口，或有明确的新环境冻结方案；
- title 不含 revert/reland，PR 不是纯 refactor/cleanup；
- 修改文件数通常不超过 3，总改动通常为 20-200 行；
- source 修复与可定位测试同时存在；
- base commit 可以获得 sparse snapshot；
- GPU task 必须可在 V100（sm_70）运行，不依赖 H100、FP8 或 flash-attn 3；
- kernel build 必须能通过增量 ccache 完成。

文件数和行数是筛选阈值，不是事实定义。若一个高质量边界修复因生成代码或必要的两端校验略超阈值，可人工记录例外，但不得放宽到大规模重构。

## 6. Admission 规则

### 6.1 通用要求

每条 task 必须满足：

- base snapshot 上 F2P 确实失败，失败原因与 issue 一致；
- gold patch 后 F2P 通过；
- P2P 覆盖普通路径和至少一个相邻边界，gold 后全部通过；
- hidden test 有明确断言，不以进程 exit 0 代替测试通过；
- test 不得被 skip、xfail 或 capability guard 静默绕过；
- patch scope 足够小，agent 可以在 task 时间预算内定位；
- CPU/GPU 资源需求可在正式服务器稳定满足。

### 6.2 边界任务附加要求

- 不使用真实 OOM、随机 allocator 状态或不可控 wall-clock timeout 作为 F2P。
- 超大 shape 优先使用 meta/fake tensor、mocked launch 参数、small-index surrogate 或低内存构造。
- crash 类 task 必须能在隔离容器中稳定返回非零状态，不能损坏宿主或后续 attempt。
- exception 类 task 同时断言异常类型；只有 contract 要求时才断言完整消息。
- CUDA 边界测试必须显式同步，避免异步错误落到后续 P2P。

### 6.3 测试执行真实性

v0.5 诊断曾发现 skip 被 exit 0 掩盖、异常被粗分类的问题。v0.6 admission 在平台层增加：

1. 记录实际 collected/executed/skipped test 数；
2. F2P/P2P 声明的 test 若未执行，结果为 `test_not_executed`，不能记 passed；
3. baseline failure 保存结构化 failure signature（exception type、assertion 或 exit signal）；
4. gold 的通过必须来自同一个 test selector 和 runtime；
5. admission evidence 保存 test execution counters，dataset validation 校验字段存在。

## 7. Matched-Wheel / Source-Build 副线

### 7.1 问题

Python overlay 只有在 source snapshot 的 Python API 与容器内 wheel/runtime 相容时才可信。#129154 和 #144073 的 base commit 与 torch 2.6 wheel 存在 API 代差，导致测试在目标 bug 断言前就因无关 AttributeError 或 compile API 不匹配失败。继续 patch task 定义会掩盖环境错误，因此 v0.5 正确地将其 deprecated。

### 7.2 环境选择顺序

对每条待恢复 task 按以下顺序选择：

1. **Matched wheel**：优先找到与 base commit 足够接近且 ABI/API 匹配的官方 wheel。
2. **Source-built wheel**：从 snapshot 构建 wheel，缓存为内容寻址环境资产。
3. **Source build**：只有 wheel 无法覆盖 compile/kernel 路径时才使用完整 source-build runtime。

不允许用更改 hidden test、猴子补 API 或跳过无关失败的方式伪造 compatibility。

### 7.3 Compatibility evidence

环境 registry 和 admission evidence 增加或明确记录：

- source commit SHA；
- wheel/build artifact digest；
- `torch.__version__`、CUDA runtime 和 Python ABI；
- source-load mode；
- 目标模块从 snapshot 而非 site-packages 加载的证明；
- 最小 compatibility probe 及结果；
- build flags、GPU arch 和 ccache key。

compatibility probe 只证明环境可用，不能代替 F2P。环境通过后仍需完整 baseline/gold admission。

### 7.4 两条恢复候选

| Task | Precision subclass | 当前阻塞 | v0.6 目标 |
| --- | :---: | --- | --- |
| `129154_exp_decomp_numerics` | P4 | CUDA refs API 与 2.6 wheel 不匹配 | matched CUDA wheel 或 source-built runtime |
| `144073_vector_norm_scalar_overflow` | P4 | CPU compile/refs API 与 2.6 wheel 不匹配 | matched CPU compile wheel 或 source build |

恢复成功后保留原 task ID 和 PR provenance，生成新的 admission evidence，并把 status 从 deprecated 改为 verified。失败则保留 deprecated 和本轮环境诊断，不反复修改测试语义。

## 8. 数据模型

boundary task 使用现有 operator 字段：

```json
{
  "operator": {
    "problem_dimension": "boundary",
    "problem_subclass": "B3",
    "problem_type": "numel-integer-overflow"
  }
}
```

约束：

- `problem_dimension`: v0.6 新 task 必须为 `boundary`，恢复 task 保持 `precision`；
- `problem_subclass`: boundary 使用 `B1..B5`，precision 继续使用 `P1..P5`；
- `problem_type`: 使用稳定、可读的根因名称，不直接复制 PR title；
- `tags`: 新 task 必须包含一个 `failure_contract:*` tag，值为 `wrong-result`、`exception`、`crash-oob` 或 `silent-acceptance`，用于报告分组；
- 历史未分类 task 不在 v0.6 强制回填，聚合时继续进入 `unclassified`。

schema 应按 dimension 校验 subclass 前缀，避免 `boundary + P3` 之类组合进入正式数据集。

## 9. 评测与报告

沿用 v0.5 八维指标：

1. resolved rate；
2. patch conciseness；
3. pass-to-pass kept rate；
4. strict resolved rate；
5. regression rate；
6. tier-weighted score；
7. per-problem dimension/subclass/type breakdown；
8. median evaluator runtime。

v0.6 报告必须额外给出：

- 按 B1-B5 的 task/attempt/resolved rate；
- 按 failure contract 分组：wrong result、exception、crash/OOB、silent acceptance；
- boundary 新增 task 与 restored precision task 分开统计；
- v0.5 的 17-task inherited slice 在新 agent 版本下的结果；
- v0.5 -> v0.6 波动归因，明确区分 dataset 增量、agent 版本和 runtime 变化；
- environment retry/raw record 与 logical attempt 的完整性说明。

正式 full summary 与两个 slice summary 都使用 `--expected-repeat 3 --require-complete`。任何 logical transient、缺 baseline、缺 attempt 或 unexpected attempt 都阻塞 release。

## 10. 实施阶段

```text
Phase 0: 设计与平台契约                                    [1-2 天]
  - boundary taxonomy / schema
  - test execution counters 与 test_not_executed
  - matched-wheel compatibility evidence 结构

Phase 1: 环境副线                                          [3-5 天]
  - #129154 matched CUDA runtime 探索
  - #144073 matched CPU compile runtime 探索
  - source-built artifact cache 与 digest
  - 可恢复则重新 admission，不可恢复则记录诊断

Phase 2: Boundary candidate screening                       [2-4 天]
  - B1-B5 keyword packs
  - git log / ghstack PR 反查
  - 自动硬过滤 + 人工软复审
  - candidates / rejected / summary 产物

Phase 3: Task 制作与 admission                              [2-3 周]
  - issue / hidden / gold / task manifest
  - preflight + remote admission
  - 目标 4-6 条 verified boundary task

Phase 4: 数据集冻结与实验                                  [2-4 天]
  - 生成 cumulative / boundary / precision manifests
  - 同一 Codex 版本全量 3-repeat
  - 三份完整性校验 summary
  - experiment report / README / CHANGELOG
```

环境副线与候选检索可以并行，但 evaluator/test counter 变更必须先完成，避免新 task admission 使用不同判定口径。

## 11. 完成标准

v0.6 只有同时满足以下条件才标记 completed：

1. `datasets/pytorch_v0.6/dataset.json` 状态为 verified，所有 entry evidence hash 有效；
2. boundary slice 至少形成一个有区分度的 verified 集合，coverage gap 明确记录；
3. #129154/#144073 均有明确结论：verified 或带新诊断的 deprecated；
4. `preflight_task.py --dataset datasets/pytorch_v0.6/dataset.json` 全部 OK；
5. 每条新 task baseline reproduced、gold resolved、P2P kept；
6. 累计数据集完成同一 agent 的 3-repeat，完整性检查无缺失；
7. boundary 和 precision slice summary 可由原始 JSONL 重新聚合；
8. 全部单测、schema 校验和 `git diff --check` 通过；实验原始 patch 中的格式问题除外；
9. 实验报告包含累计、boundary、precision、tier、subclass 和 v0.5 对比；
10. README、CHANGELOG、docs index 与实际冻结结果一致。

## 12. 风险与降级策略

### 候选稀缺

不使用非 boundary PR 填配额。优先保证根因和 F2P 质量，允许 B1-B5 有空缺，并把候选检索结果留给后续版本。

### 大 shape 不可复现

优先寻找 meta/fake/surrogate 复现。若 bug 只能通过超过服务器内存或超长时间触发，暂不 admission，不把 OOM 当作稳定 F2P。

### Crash 污染环境

每个 crash task 在独立容器执行，强制超时、进程清理和 GPU health probe。若一次 crash 会使 GPU reset 或影响后续 task，该候选不适合当前基础设施。

### Matched environment 构建成本过高

恢复 precision 是副线，不阻塞 boundary release。构建产物必须内容寻址并可复用；无法稳定缓存的 nightly/source build 不进入正式 registry。

### 全量实验成本增长

GPU/kernel task 保持串行，CPU 并发从 1-3 实测。允许拆 batch，不允许复用旧 agent 版本结果拼成累计分数。若 rate limit 中断，使用同一 output-dir resume。

## 13. 计划产物

| 文件 | 用途 |
| --- | --- |
| `docs/v0.6/design.md` | 本设计方案 |
| `docs/v0.6/candidate_search.md` | B1-B5 keyword packs、筛选规则和 schema |
| `docs/v0.6/setup_matched_runtime.md` | matched wheel/source-build 环境制作与验证 |
| `docs/v0.6/experiment_report.md` | v0.6 正式实验报告 |
| `runs/v0.6_pr_screening/` | boundary candidates、rejected 和 summary |
| `datasets/pytorch_v0.6/dataset.json` | v0.6 累计数据集 |
| `datasets/pytorch_v0.6_boundary/dataset.json` | boundary slice |
| `datasets/pytorch_v0.6_precision/dataset.json` | 更新后的 precision slice |
| `runs/v0.6_codex/summary.json` | 累计完整性与八维指标 |
| `runs/v0.6_boundary_codex/summary.json` | boundary slice 指标 |
| `runs/v0.6_precision_codex/summary.json` | precision slice 指标 |

## 14. 后续路线

- v0.7：设备/API 兼容，包括 CPU/CUDA 行为一致性、跨 device 参数、dispatch 与 `.to(device)` 传播。
- v0.8：性能问题，引入硬件基线、噪声模型和性能 regression 判定。
- v0.9：在问题维度和评分口径稳定后开展正式多 agent 对比。
- 后续版本再评估 TorchVision、TorchAudio、JAX 等横向框架扩展。
