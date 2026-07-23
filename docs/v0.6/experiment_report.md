# OpBench v0.6 实验报告

日期：2026-07-22～2026-07-23

## 0. 版本结论

v0.6 平台化后的真实 Agent 全量实验已经完成。实验使用冻结的
`pytorch_v0.5` 数据集，通过独立的 `codex_mcp_canonical` Adapter，
让 Codex 经 invocation-local stdio MCP 调用 OpBench 的九个 Canonical
Action。17 条 verified PyTorch task 各执行 3 次，共得到 **51/51 个有效
Attempt**。

最终结果为 **35/51 resolved（68.6%）**、15 个 F2P failed 和 1 个
P2P regression。全部 51 个 Agent terminal 都是 `finished`；本次正式矩阵有
**0 次基础设施无效**、**0 次逻辑重试**，因此失败结果是应保留的真实 Agent
结果，不是网络抖动或超时造成的无效记录。

| 实验合同 | 结果 |
| --- | --- |
| Dataset | 17 条 verified task，digest 冻结 |
| Agent matrix | 17 task × 3 repeat = 51 Attempt |
| 有效性 | 51/51 valid，0 missing，0 blocked |
| Agent terminal | 51/51 `finished` |
| Evaluation | 35 resolved，15 F2P failed，1 P2P regression |
| MCP trace | 51/51 完整，747 次 tool call，0 protocol error |
| Integrity | 4 × 14/14 root；51 × 10/10 per-Attempt |
| Runtime cleanup | 51/51 |
| Retry | 0 infrastructure-invalid，0 logical retry |

这次实验验证了 v0.6 能通过真实 MCP Agent 完成 TaskView、Action、Trace、
Patch Freeze、Fresh Evaluation、Integrity、Resume/Retry 判定和精确资源清理
的端到端闭环。它是单 Agent、单冻结数据集的平台验证，不是 MCP 与 CLI 的
因果 A/B 实验，也不是跨 Agent 排名。

## 1. 实验目标

本轮实验回答四个问题：

1. v0.6 新增的 MCP、Trace、Artifact 和三轴结果语义能否承载真实 Agent
   全量评测，而不只在 scripted test 或 canary 中成立；
2. 17-task × 3-repeat 矩阵能否在四种 Runtime Profile 上完整结束，并把
   Agent 失败与基础设施失败分开；
3. 每个 Attempt 的 Action、Patch、Evaluation、Integrity 和 Cleanup
   是否形成可核验的一致证据链；
4. 清理私有运行根后，是否仍能保留一份确定性、脱敏、适合版本控制的最终报告。

本轮不扩充数据集，不更改 task、测试、Gold Patch 或评分规则，也不执行
CLI/MCP 双全量对照。

## 2. 冻结配置

```text
Platform:        opbench-v0.6.0
Dataset:         pytorch_v0.5
Dataset digest:  sha256:ff9d0c2999d1175a45165b387e0731dcaa211a190d994b176441ce81a0382abc
Adapter:         codex_mcp_canonical
Transport:       invocation-local mcp-stdio
Action protocol: action-v1
Evaluation:      evaluation-v1
Scoring:         legacy-v0.5-resolved-v1
Model:           gpt-5.6-sol
Codex CLI:       codex-cli 0.145.0-alpha.27
Repeat:          3
Resume policy:   retry_infrastructure
```

Agent 只通过九个 Canonical MCP Action 访问任务。Provider network 仅供
host-side Codex 调用；Task network 保持禁止。Remote Runtime 只连接配置中
指定的精确目标，没有执行 ping、端口/服务扫描、目标发现、资源枚举或宽泛清理。

17 条任务按运行需求划分为四个独立 Cohort：

| Runtime Profile | Task | Attempt |
| --- | ---: | ---: |
| Remote CPU | 12 | 36 |
| Remote CPU Compile | 1 | 3 |
| CUDA Overlay | 2 | 6 |
| CUDA Kernel | 2 | 6 |
| **合计** | **17** | **51** |

四个 Profile 产生四个 Comparability Key。本文可以报告总计，但不会把四个
Cohort 当成运行环境完全相同的一组样本。

## 3. 总体结果

### 3.1 按 Runtime Profile

| Runtime Profile | Attempt | Resolved | F2P failed | P2P regression | Resolved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `remote-cpu-pytorch-2.6-py311-v1` | 36 | 21 | 14 | 1 | 58.3% |
| `remote-cpu-compile-pytorch-2.6-py311-v1` | 3 | 2 | 1 | 0 | 66.7% |
| `remote-cuda-overlay-pytorch-2.6-cu124-v1` | 6 | 6 | 0 | 0 | 100.0% |
| `remote-cuda-kernel-pytorch-2.6-cu124-v1` | 6 | 6 | 0 | 0 | 100.0% |
| **合计** | **51** | **35** | **15** | **1** | **68.6%** |

CUDA Overlay 和 CUDA Kernel 在当前各 6 个 Attempt 中都是 6/6，但样本量很小，
不能据此推断 CUDA task 普遍更容易。该结果只说明冻结的两条 Overlay task 和
两条 Kernel task 在本次 Agent/Profile 组合下全部解决。

### 3.2 逐 Task 结果

表中的时长是公开索引记录的单次 Agent duration 中位数，不包含对平台准备、
最终审计和报告生成的完整墙钟时间。

| Task | Profile | A1 | A2 | A3 | Resolved | Agent 中位时长 |
| --- | --- | --- | --- | --- | ---: | ---: |
| `124385__load_state_dict_prefix` | CPU | resolved | resolved | resolved | 3/3 | 180.6s |
| `129138__linear_add_bias_autocast` | CPU Compile | F2P failed | resolved | resolved | 2/3 | 189.0s |
| `132616__cuda_mem_get_info` | CUDA Overlay | resolved | resolved | resolved | 3/3 | 178.7s |
| `132835__njt_sdpa_autocast` | CUDA Overlay | resolved | resolved | resolved | 3/3 | 190.3s |
| `139372__histc_int8_cuda_bounds` | CUDA Kernel | resolved | resolved | resolved | 3/3 | 919.3s |
| `139999__masked_mean_bool_upcast` | CPU | resolved | resolved | resolved | 3/3 | 143.6s |
| `140557__layer_norm_decomp_precision` | CPU | resolved | resolved | F2P failed | 2/3 | 182.5s |
| `143455__set_submodule` | CPU | F2P failed | resolved | F2P failed | 1/3 | 145.5s |
| `144009__softmax_ilpreduce_size` | CUDA Kernel | resolved | resolved | resolved | 3/3 | 847.4s |
| `147599__lazylinear_state_forward` | CPU | resolved | resolved | resolved | 3/3 | 137.4s |
| `149693__lazylinear_init` | CPU | resolved | resolved | resolved | 3/3 | 150.2s |
| `150975__autograd_backward_inputs` | CPU | P2P regression | F2P failed | F2P failed | 0/3 | 146.4s |
| `160952__bilinear_lazy_check` | CPU | resolved | resolved | resolved | 3/3 | 141.9s |
| `161488__lbfgs_wolfe` | CPU | F2P failed | F2P failed | F2P failed | 0/3 | 143.8s |
| `162340__nn_arg_length` | CPU | F2P failed | F2P failed | F2P failed | 0/3 | 144.8s |
| `163961__dataloader_subset` | CPU | F2P failed | F2P failed | F2P failed | 0/3 | 150.9s |
| `168295__autograd_create_graph` | CPU | resolved | resolved | resolved | 3/3 | 152.9s |

稳定性分布为：10 条 task 稳定 resolved（3/3），4 条稳定未解决（0/3），
2 条为 2/3，1 条为 1/3。全部 51 个 Attempt 都产生了一份非空 Patch，
且每份 Patch 恰好修改一个文件。

### 3.3 Precision slice

沿用 v0.5 的 Precision subclass，只做描述性切片：

| Subclass | Task | Attempt | Resolved | Rate |
| :---: | ---: | ---: | ---: | ---: |
| P1 数值累积误差 | 1 | 3 | 3 | 100.0% |
| P2 dtype/分解转换损失 | 1 | 3 | 2 | 66.7% |
| P3 混合精度不一致 | 2 | 6 | 5 | 83.3% |
| P4 数值不稳定 | 0 | 0 | N/A | N/A |
| P5 CUDA kernel 精度 | 2 | 6 | 6 | 100.0% |
| **Precision slice** | **6** | **18** | **16** | **88.9%** |

P4 仍然没有通过 admission 的 task，因此保持 N/A，而不是按 0% 计入。

## 4. MCP、Action 与 Trace 观察

每个 Attempt 都完成一次 MCP initialize 和 tool list。最终证据包含：

| 项目 | 结果 |
| --- | ---: |
| MCP initialize | 51 |
| MCP tool list | 51 |
| MCP tool call | 747 |
| MCP protocol error | 0 |
| Server terminal `client_closed` | 51 |
| Read/Edit/Diff/Finish coverage | 51/51 |
| Registered `test_run` coverage | 0/51 |

747 次 Action 调用分布如下：

| Action | Calls |
| --- | ---: |
| `workspace_search` | 271 |
| `workspace_apply_patch` | 157 |
| `workspace_read` | 132 |
| `command_run` | 80 |
| `vcs_diff` | 52 |
| `session_finish` | 51 |
| `workspace_list` | 2 |
| `workspace_write` | 2 |

Agent 没有调用注册的 `test_run`。80 次 `command_run` 全部被策略拒绝为
`capability_denied`；Fresh Evaluator 仍在 Agent Session 之外对全部 51 个 Patch
执行冻结的 F2P/P2P 测试。因此“Agent 未主动运行注册测试”和“平台没有评分”
是两件不同的事。

公开 Trace 共记录 80 个 `capability_denied`、131 个 `invalid_request` 和
151 个 `path_denied`。这些是 Canonical Action Service 返回给 Agent 的可恢复
观察，不是 MCP 协议错误、网络超时或基础设施无效。

公开索引中的 Agent duration 中位数为 151.4 秒，范围 128.3～949.8 秒。
51 份 Patch 总计 39,817 bytes，单份 532～2,358 bytes，平均约 781 bytes；
每份均只改一个文件。这些轨迹统计是行为观察，不能单独证明某类 Action 反馈
导致了成功或失败。

## 5. 失败与稳定性分析

正式矩阵中的 16 个非 resolved 结果全部归因于 Agent：

- `161488__lbfgs_wolfe`、`162340__nn_arg_length` 和
  `163961__dataloader_subset` 均为 3 次 F2P failed；
- `150975__autograd_backward_inputs` 为 2 次 F2P failed 和 1 次
  P2P regression，是唯一出现回归的 task；
- `129138__linear_add_bias_autocast` 与
  `140557__layer_norm_decomp_precision` 各有 1 次 F2P failed；
- `143455__set_submodule` 有 2 次 F2P failed。

这里的 F2P failed 表示补丁没有满足冻结的 fail-to-pass 测试；P2P regression
表示补丁解决了目标失败，但破坏了 pass-to-pass 约束。Agent terminal
`finished` 只表示交互正常结束，不等于补丁通过评分。

在最终正式结果中，Provider、MCP 和 Runtime 的失败归因均为 0。早期诊断过程中
发现并修复了大源码同步续传、同步后 ccache seed copy 的瞬时连接恢复，以及
C/C++ 注释被公开路径扫描器误报三个问题。诊断/无效根不进入正式分母；修复后
CUDA Kernel 正式 Cohort 为 6/6 valid、6/6 resolved、0 retry。修复没有改动
Dataset、Task、评分协议或已成功的早期 Cohort Artifact。

## 6. 与 v0.5 的描述性对照

| 项目 | v0.5 | v0.6 MCP 实验 |
| --- | --- | --- |
| Dataset | `pytorch_v0.5`，17 task | 同一冻结 Dataset，17 task |
| Attempt | 51 | 51 |
| Adapter | `codex_action_bridge` | `codex_mcp_canonical` |
| Model | `gpt-5.6-terra`，low effort | `gpt-5.6-sol` |
| Codex CLI | `0.144.0-alpha.4` | `0.145.0-alpha.27` |
| Action transport | Legacy action bridge | invocation-local MCP stdio |
| Result | 37/51（72.5%） | 35/51（68.6%） |

表面上 v0.6 比 v0.5 低 2 个 resolved Attempt（-3.9 个百分点），但 Adapter、
模型、CLI、Action 表面、Runtime Profile、执行协议和采样都发生了变化，不能把
差异解释成“MCP 使 Agent 变好或变差”。本轮没有运行相同模型、相同 CLI、
相同 Runtime 下的 CLI/MCP 随机对照，因此只保留描述性事实。

## 7. 完整性、确定性与清理

四个完整 controller-private Cohort root 在清理前分别通过 14/14 root
Integrity；51 个 per-Attempt 报告分别通过 10/10。资源验证器对 51/51
Attempt 证明精确 ownership 和 cleanup 完成。51 个 Attempt ID 跨四组唯一，
model、CLI、Adapter、Dataset、Profile、Evaluation 和 selected retry 绑定完整。

只读报告生成器对相同的四个 immutable root 运行两次，输出逐字节一致：

| 公开文件 | SHA-256 |
| --- | --- |
| [`experiment_report.md`](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_report.md) | `3a3ea200a58713bc7f5060a70a098e3e542bce6cb4c6965257baf273f4b77caf` |
| [`experiment_index.json`](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_index.json) | `4ab5a112291b7c8c4ba6df319765a1bf1fb225fcd1030a33078f3efacad114ff` |
| [`experiment_summary.json`](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_summary.json) | `8c03361f088704d56730f8adc9a2e01b7a6d5115a78f2c49d1807e6ff76d2443` |

仓库只保留以上三个确定性公开输出。本轮 MCP 实验的正式 Cohort root、预跑
canary、失败诊断和中间 retry root 在验证后从工作区移除，避免把大量可再生
中间产物当成发布内容。

## 8. 复现与证据边界

完整命令、四组 task 划分、resume 规则和安全边界见
[真实 MCP Agent 实验手册](mcp_agent_experiment.md)；逐项 Integrity、Trace、
Cleanup、确定性和隐私验证见
[实验验证记录](mcp_agent_experiment_verification.md)。

公开报告不包含 `private_evaluation.json`、
`private_runtime_resources.json`、Provider raw output、Credential、目标地址、
用户/密钥路径、PID/PGID、容器原名、远程 workspace 或控制器绝对路径。
14/14 Integrity 是在脱敏前对完整 controller-private root 得出的；仅凭仓库中的
三个公开报告文件不能重新执行依赖 private handle 的完整 Integrity graph。

## 9. 结论

v0.6 的平台实现和真实 MCP 全量实验均已闭环：矩阵完整、失败分母合法、Trace
完整、评分独立、资源精确清理，最终公开证据确定且脱敏。35/51 是本次冻结
Agent/Profile/协议组合的观测结果；它证明平台可用于真实 Agent 评测，但不证明
MCP 的因果质量效果，也不代表其他模型、Agent、框架或数据集上的表现。
