# OpBench v0.7 Boundary Validation Cohort 报告

日期：2026-07-28

状态：Passed

## 1. 结论

OpBench v0.7 在冻结的 `pytorch_v0.7_boundary` Dataset 上完成了一个真实
Codex Validation Cohort：6 个 Boundary Task、5 个 Runtime Profile、每 Task
3 次 repeat，共 18 个有效 Attempt。结果为 **14 resolved、3 f2p_failed、
1 no_patch**；Agent terminal 为 17 finished、1 timeout。

这是单一 Agent/模型配置下的 Task 与平台验证，是明确的
**non-leaderboard** 结果。它不构成跨 Agent 排名，不证明反馈因果关系，也不把
18 次 Attempt 外推为模型总体能力。

机器可读的
[index](../../runs/v0.7_validation_report/experiment_index.json)、
[summary](../../runs/v0.7_validation_report/experiment_summary.json) 和
[report](../../runs/v0.7_validation_report/experiment_report.md)
由五个通过 Integrity gate 的正式 cohort 确定性生成。两次独立生成的三个文件
逐字节一致。

## 2. 冻结身份

| 字段 | 值 |
| --- | --- |
| Dataset | `pytorch_v0.7_boundary` |
| Dataset digest | `sha256:eaaa93301975ebcf3507c1efe18b600c729ae1e978696bb331546ca59013f0cf` |
| Platform | `opbench-v0.6.0` |
| Agent Adapter | `codex_mcp_canonical` |
| Model | `gpt-5.6-sol` |
| Codex CLI | `codex-cli 0.146.0-alpha.3.1` |
| MCP protocol | `2025-06-18` |
| Attempt contract | 6 Tasks × repeats 1/2/3 = 18 |

18 条 Adapter trace 全部完整：initialize 18、tools/list 18、
protocol error 0。五个正式 cohort 的 fresh Integrity 结果均与持久化结果一致，
每个 root 14/14 checks 通过，Runtime resource ownership 与 cleanup gate 均通过。

## 3. Runtime 分区与结果

| Runtime Profile | Task | 结果 |
| --- | --- | ---: |
| `remote-cpu-boundary-torch2.2-py311-v1` | `pytorch__117065__index_copy_zero_dim` | 3/3 resolved |
| `remote-cpu-boundary-torch2.3-py311-v1` | `pytorch__118762__weight_norm_default_dim` | 3/3 resolved |
| `remote-cpu-boundary-torch2.4-py311-v1` | `pytorch__126461__cummin_rank_zero` | 2/3 resolved、1/3 no_patch |
| `remote-cpu-source-boundary-py311-v1` | `pytorch__143792__addmv_empty_matrix` | 3/3 resolved |
| `remote-cpu-source-boundary-py311-v1` | `pytorch__147352__storage_offset_overflow` | 3/3 resolved |
| `remote-cuda-boundary-torch2.6-cu124-v1` | `pytorch__139751__triton_ygrid_mask` | 0/3 resolved、3/3 f2p_failed |

Accepted cohort 的 18 个 ledger 选择均为 `valid`，全部来自
`retry_index=1`，因此公开 summary 的 retry 数为 0。没有把
infrastructure-invalid Attempt 计入分母。

## 4. floor / ceiling 与异常复核

五条 Task 出现 **ceiling（3/3 resolved）**。逐条检查 patch、F2P/P2P 摘要、
Action 配对和 selector 执行后：

- 三条 CPU overlay Task 中，#117065 与 #118762 的三次 F2P/P2P 都是 1/1；
- 两条 exact-source Task 的六次 F2P/P2P 都是 1/1，且每次 patch 都由独立
  evaluator 从当前 authoritative source 构建后执行；
- 未发现 Task 泄漏、selector 未执行、环境漂移或 transport 输出冒充测试结果。

`pytorch__126461__cummin_rank_zero` 的一个 no_patch Attempt 没有生成 patch，
其余两个 repeat 均通过 F2P/P2P。这是 Agent terminal 的真实行为，不是 evaluator
漏测。

`pytorch__139751__triton_ygrid_mask` 出现 **floor（0/3 resolved）**。人工复核
确认三次都是 Agent patch 的真实回归：

- repeat 1/2 访问了测试 surrogate 不具备的状态，F2P 失败；
- repeat 3 的 patch 产生语法/缩进错误，且 Agent terminal 为 timeout，但冻结的
  patch 仍被 evaluator 正常评测；
- 三次均有完整 trace 和实际 selector 证据，不是 GPU、容器或 SSH 故障。

这些 floor / ceiling 观察用于提示后续扩大样本和 Agent 分布，未反向修改
Admission threshold、Hidden Test、Gold Patch、Dataset membership 或评分规则。

## 5. source loading 验证

`remote-cpu-source-boundary-py311-v1` 不依赖已安装 wheel 覆盖真实源码。每个
registered test 在 selector 前执行 source preparation：把 Agent 最终工作区作为
authoritative source，同步到隔离 Runtime，完成 CPU full-source build，再运行
F2P/P2P。准备失败或超时会跳过 selector 并产生基础设施无效结果。

两个 source Task 的 6 个有效 Attempt 全部 resolved，F2P/P2P 各 6/6；这同时验证
了 current Agent source 对 selector 可见，以及 evaluator 每次仍从 fresh workspace
独立准备源码，没有复用 Agent 侧测试状态。

## 6. transport 审计与 retry 归因

正式 source cohort 前的预验收审计发现：OpenSSH 在远端命令开始前断连时，旧实现
可能把退出码 255 错归因为逻辑 selector 失败。平台修复分两层：

1. 识别 KEX、connection reset/closed 等 pre-execution transport 信号，并在同一
   selector deadline 内做有界重试；
2. 在远端命令真正开始时写入 controller-only sentinel。sentinel 前断连可以安全
   retry；sentinel 后断连标记为 infrastructure-invalid，绝不重放可能已经开始的
   evaluator 命令。

被该问题污染的预验收 root 没有进入正式报告。修复后从统一代码快照重跑完整
source cohort；6/6 有效结果的控制器证据中均无 KEX、connection reset/closed
痕迹。五个 accepted cohort 的 18 个结果均无需 retry，因此机器摘要记录 0 retries。

## 7. 可复现性与公开边界

正式分区由
[`validation_contract.json`](../../factory/v0.7/p4/validation_contract.json)
冻结。公开报告可用以下等价命令重建；每个 `--run-root` 必须指向对应的、已完成
Integrity gate 的正式 cohort：

```bash
PYTHONPATH=src python scripts/summarize_mcp_experiment.py \
  --run-root <torch-2.2-cohort> \
  --run-root <torch-2.3-cohort> \
  --run-root <torch-2.4-cohort> \
  --run-root <source-cohort> \
  --run-root <cuda-cohort> \
  --output-dir <empty-output-dir> \
  --expected-model gpt-5.6-sol \
  --expected-cli-version 'codex-cli 0.146.0-alpha.3.1' \
  --contract factory/v0.7/p4/validation_contract.json
```

公开产物只包含内容身份、聚合计数、受限 Attempt 索引和确定性 Markdown。主机、
账号、凭据、控制器路径、原始 Agent 文本和无界 evaluator 日志均不进入提交。

## 8. 限制

- 只有一个 Agent Adapter、一个模型和三个 repeat，不能估计跨模型方差；
- 六条 Boundary Task 的 ceiling/floor 分布可能受样本规模影响；
- GPU 仅覆盖冻结的单一硬件类别与 CUDA Runtime；
- Validation Cohort 描述当前 Task/platform/Agent 组合，不是正式排行榜；
- 未执行反馈可见性对照实验，因此不支持反馈因果结论。

P4 Validation Cohort 结论为 **Passed**。本报告只确认冻结 Dataset、Runtime、
source loading、Evaluation、Trace 与 Integrity 合同能共同产生可信、可审计的
18-Attempt 结果。
