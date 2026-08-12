# OpBench v0.7 50-task Quality Validation 报告

日期：2026-08-11

状态：Passed（50/50 fresh replay；122/122 valid logical Attempts）

## 1. 当前结论

OpBench v0.7 已完成 50 条 verified PyTorch 算子修复任务的 fresh replay 和冻结的
真实 Codex 验证实验。50 条任务由 14 条 retained historical、21 条 new 和 15 条
replacement 组成；36 条新增/替换任务均通过真实 Runtime Admission，没有把本地
preflight 或静态 patch check 当作正式证据。fresh Baseline/Gold replay 为 50/50，
17 个 cohort 的 122/122 logical Attempts 均由 append-only ledger 选出有效结果，
122 条 MCP 工具轨迹完整。

最终 Attempt 结果为 42 resolved、52 F2P failed、28 invalid patch；没有 P2P failed
或 no-patch 结果。8 个 infrastructure-invalid 历史 retry 均被保留审计并补跑为有效
logical Attempt，没有混入 Agent 失败分母。报告三件套已从原始 cohort root 重建并
与正式产物逐字节一致。

最终结果只描述冻结的单一 Agent/模型/Runtime 组合，是明确的
**non-leaderboard** evidence；它不构成跨 Agent 排名，也不支持反馈因果结论。

## 2. 冻结身份

| 字段 | 值 |
| --- | --- |
| Dataset | `pytorch_v0.7`，50 verified Tasks |
| Dataset file hash | `sha256:3695622dd2619a760d510ef49e0a9dbff637c98790ad3263c521bae8e99c9518` |
| Dataset contract digest | `sha256:a6f2b7c9f54e79e30a698ca0b64b72f5b6e644fb0502a82946796d9ca56cde54` |
| Quality release digest | `sha256:e35318dd47554f90556bc132c71338a8fb34e952bbe355cac45c0d1dc8de1839` |
| Validation contract | `factory/v0.7/p9/validation_contract.json`（schema v2） |
| Validation contract file SHA-256 | `9dfdb7ddc1ef71170e6264baca2bcc8ce6f089b6ef01dc3dc973bae0706cd8f2` |
| Platform | `opbench-v0.6.0` Runtime Protocol |
| Agent Adapter | `codex_mcp_canonical` |
| Model | `gpt-5.6-sol` |
| Codex CLI | `codex-cli 0.147.0-alpha.1.2` |
| Action / Evaluation / Scoring | `action-v1` / `evaluation-v1` / `scoring-v1` |
| Frozen matrix | 17 cohorts、50 Tasks、122 logical Attempts |

合同额外绑定 Agent spec、system/task Prompt、Prompt renderer source、Evaluator、
Retry、Termination、Scoring、每个 Runtime Profile、Capability、Budget、RunManifest
以及 50 个 AgentTaskView 的内容 digest。修改任何一个身份都必须形成新实验，不能
继续写入当前分母。

## 3. Attempt 矩阵

```text
36 new/replacement Tasks × repeats 1/2/3 = 108 Attempts
14 retained historical Tasks × repeat 1     =  14 Attempts
                                             --------------
                                             122 Attempts
```

17 个 cohort 按完全相同的 Runtime Profile 分区。它们覆盖：

- 35 个 CPU 与 15 个 CUDA Task；
- 34 个 CPU overlay、1 个 CPU source build、13 个 CUDA overlay、2 个 CUDA
  kernel build Task；
- 35 个 compile 与 21 个 eager Task；
- 49 个 forward 与 4 个 backward Task。

同一 Task 可同时覆盖多个 mode/phase，因此后两组计数不是互斥分布。 retained Task
只运行 1 次，是用户批准的资源分配合同；36 条新增/替换运行 3 次以观察稳定性。

## 4. 执行与完整性门禁

执行严格分为两步：

1. 对冻结 50-task release 执行 fresh Baseline/Gold replay，要求 50/50 bundle、
   Registry、Runtime 和 selector identity 重建一致；
2. 先运行 CPU、CPU compile、CUDA overlay、CUDA kernel 四类 canary，再在同一冻结
   contract 与 append-only ledger 下 resume 完整 122 Attempts。

每个正式 Attempt 必须同时满足：

- logical Attempt ID 与 contract 预期矩阵一致；
- Agent 只收到扫描后的 AgentTaskView；
- Session、Frozen Patch 和 fresh Evaluation 绑定同一 patch bytes/hash；
- F2P/P2P 使用 evaluator-owned structured evidence；
- Adapter trace 完整且没有协议错误；
- fresh Integrity graph 通过；
- Attempt-owned Runtime 资源精确回收；
- ledger 只选择一个 valid retry 进入逻辑分母。

基础设施无效（`infrastructure-invalid`）记录保留在 append-only 审计中，但不计作
Agent 失败。补跑必须沿用同一 logical Attempt 身份并由 retry policy 选择最终有效
结果，不能删除失败历史。

## 5. 失败归因与评分口径

最终机器报告区分：

- `resolved`、`f2p_failed`、`p2p_failed`、`no_patch`；
- Agent、Runtime、Evaluator 与 Infrastructure failure；
- retry、terminal reason、trace/integrity/resource gate；
- origin、difficulty、contract family、failure type、device、mode、phase 与派生切片。

Boundary、Precision、Device 是可重叠派生切片，不是互斥顶层 Dataset。独立 slice
评分只在至少包含 3 条 Task 时报告；样本不足、全 resolved 或全 unresolved 的分组只
作为描述性观察，避免小桶制造虚假精确度。完整 Agent 自然语言输出与隐藏思考过程
不进入 v0.7 记录或评分合同。

## 6. 最终执行结果

| 门禁 / 指标 | 结果 |
| --- | ---: |
| p8 新增/替换 Runtime Admission | 36/36 verified |
| p9 cumulative / Boundary / Precision / Device | 50 / 31 / 5 / 15 verified |
| fresh Baseline/Gold replay | 50/50 passed |
| Agent cohort / logical Attempt | 17/17；122/122 valid |
| 完整 MCP trace | 122/122 |
| resolved | 42/122（34.4%） |
| F2P failed | 52/122 |
| invalid patch | 28/122 |
| infrastructure-invalid retry history | 8（provider 1；runtime 7） |
| 全部观测 Attempt resolved 的 Task | 15 |
| 没有观测 Attempt resolved 的 Task | 29 |

`agent=80` 表示 122 个最终有效 Attempt 中未 resolved 的 80 个结果；`runtime=7`
则是 8 条无效 retry 历史中的归因子集，二者不是可相加的同一分母。所有 122 个最终
有效 Attempt 的 Agent terminal 均为 `finished`，MCP protocol error 为 0。15/29 的
ceiling/floor 只描述当前冻结配置；retained Task 仅运行一次，不能把该组误读为
跨重复稳定性。

按主要视图观察：CPU 为 33/85 resolved，CUDA 为 9/37；Boundary 为 31/77，
Precision 为 3/5，Device 为 9/37。分层数字用于定位覆盖和失败模式，不是独立采样
总体或跨 Agent 排名。

机器可读证据位于：

- `runs/v0.7_quality_replay/index.json`：50-task fresh replay 索引；
- `runs/v0.7_quality_validation/index.json`：17-cohort 执行索引；
- `runs/v0.7_quality_validation_report/experiment_index.json`：脱敏 Attempt 索引；
- `runs/v0.7_quality_validation_report/experiment_summary.json`：聚合结果；
- `runs/v0.7_quality_validation_report/experiment_report.md`：确定性机器报告。

原始 replay/cohort 目录包含 evaluator、Runtime 和资源审计细节，保留为本地私有
证据并由 `.gitignore` 明确排除；公开树只纳入通过隐私扫描的顶层索引和聚合报告。

## 7. 18 项完成标准审计

| # | 结论 | 结构化证据 |
| ---: | --- | --- |
| 1 | Passed | `archives/v0.7-pre-quality.json` 绑定 `4f5addc`、旧 Dataset Hash 与 18-attempt cohort 身份 |
| 2 | Passed | `factory/v0.7/p7/historical_readmission.json`：25/25，14 retained、1 deferred、10 retired |
| 3 | Passed | cumulative Dataset：50 个唯一 verified Task |
| 4–5 | Passed | 50 份最终 Prompt evidence 与 AgentTaskView 自动/语义审查；无 provenance、答案路径、Gold/Hidden 或修复指令 |
| 6–7 | Passed | 50/50 taxonomy/complexity evidence；hard 46、medium 4、easy 0 |
| 8 | Passed | CPU 35、CUDA 15 |
| 9 | Passed（有公开缺口） | compile 35；backward 4、gradient 2、唯一并集 5；满足代表性硬门，低于早期 6 条搜索目标的缺口已公开 |
| 10 | Passed | Boundary/Precision/Device 由同一 Task truth 确定性派生为 31/5/15，并公开 distributed 等缺口 |
| 11 | Passed | fresh replay 50/50，Baseline/Gold/F2P/P2P/Runtime/Integrity 全部通过 |
| 12 | Passed | 36 条 new/replacement × 3 = 108 valid Attempts |
| 13 | Passed | 14 条 retained × 1 = 14 valid Attempts |
| 14 | Passed | 最终 17 个 cohort 均为 `infrastructure_invalid=0`；8 条历史无效 retry 不进入最终分母 |
| 15 | Passed | archive、release、Prompt、contract 与 cohort 身份分离；旧 18 Attempts 未拼接 |
| 16 | Passed | release、validation contract 与报告三件套 fresh 重建逐字节一致 |
| 17 | Passed | 1101/1101 tests、Schema/compile/JSON/link/safety/privacy/diff gates 通过 |
| 18 | Passed | Dataset Card、双语入口、路线图、状态、CHANGELOG 与本报告发布一致且明确 non-leaderboard 限制 |

第 9 项采用后续批准的质量优先、非配额合同：CPU/CUDA 保持数值硬门，compile 与
backward/Autograd 必须有代表性；搜索目标未满时公开缺口，不能用错误标注、简单任务
或降低 Admission 标准补数。该修订已同步回设计稿，不把 5 条唯一覆盖写成 6 条。

## 8. 历史 18-attempt 证据

2026-07-28 的旧发布曾在 6 条 Boundary Task 上运行 3 repeats，共 18/18 valid，
结果为 14 resolved、3 f2p_failed、1 no_patch、0 accepted-cohort retry。该实验当时
通过 trace、Integrity 与 cleanup 门禁，其历史事实仍有效。

但旧实验的 Dataset、CLI、Prompt/Task membership 和 repeat matrix 与当前合同不同，
因此只能作为历史平台证据，不能拼接到 122 Attempts、不能用于补齐缺失 repeat，
也不能证明 50-task v0.7 已完成。旧 25/6/8 Dataset 已保存在
`archives/v0.7-pre-quality/`。

## 9. 可复现性与公开边界

验证冻结合同：

```bash
PYTHONPATH=src python scripts/build_v07_quality_validation_contract.py \
  --output /tmp/opbench-v07-validation-contract.json
cmp factory/v0.7/p9/validation_contract.json \
  /tmp/opbench-v07-validation-contract.json
```

从 17 个 cohort root 重建公开报告：

```bash
PYTHONPATH=src python scripts/summarize_mcp_experiment.py \
  --contract factory/v0.7/p9/validation_contract.json \
  --input-root runs/v0.7_quality_validation \
  --output-root runs/v0.7_quality_validation_report
```

重建后的 `experiment_index.json`、`experiment_summary.json` 和
`experiment_report.md` 已逐字节匹配正式产物。公开产物仅包含内容身份、受限
Attempt 索引、聚合计数和确定性 Markdown。主机、
账号、凭据、controller path、原始 Agent 文本、完整 evaluator 日志与私有 selector
内容不进入提交。

## 10. 限制

- 单一 Agent Adapter、模型和冻结硬件配置不能估计跨模型方差；
- retained 与 new/replacement 的 repeat 数不同，聚合时必须同时报告 per-Task 与
  per-Attempt 口径；
- 50 条仍是 PyTorch-only 研究规模，细分结果需要谨慎解释；
- GPU 只覆盖冻结的单一目标硬件类别与 Runtime；
- backward phase 为 4 条、gradient family 为 2 条且唯一并集为 5 条，低于早期
  6 条搜索目标；这是公开覆盖缺口，不影响“必须有代表性”的最终质量合同；
- source/kernel build 的 wall-clock 主要反映构建成本，不是 Agent 推理速度；
- 未执行反馈可见性对照，不支持 feedback causality；
- 公开上游历史可能进入训练数据，不能声称无污染。

v0.7 的 fresh replay、Agent 122/122 valid、确定性报告、Integrity、资源清理、
公开树隐私扫描和文档门禁均已通过；最终全量回归为 1101/1101（1260.411 秒）。
后续修改冻结的 Dataset、Prompt、Agent、Runtime、Evaluator、预算或评分身份都
必须创建新实验，不能继续写入本结果分母。
