# OpBench v0.7 质量扩展发布合同

日期：2026-08-03

状态：In development

## 1. 版本边界

2026-07-28 冻结的 25-task cumulative、6-task Boundary 和 8-task Precision
数据集仍是可复现的历史基线，其 Hash、Admission 与 18-attempt Validation
Cohort 不做追溯改写。

当前开发工作把原计划中的后续数据扩充合并回 v0.7，并以质量优先的 50-task
发布取代“25 条即为本版本最终规模”的旧完成条件。最终 v0.7 只有在本文件定义的
50-task 数据、Admission、Agent 实验和发布审计全部通过后才能再次标记为
Completed。历史 25-task 冻结证明旧发布真实存在，但不能证明当前目标完成。

## 2. 最终规模

正式累计数据集必须恰好包含 50 条任务：

```text
14 retained historical tasks
+ 36 new or replacement tasks
= 50 final tasks
```

- 14 条历史任务由 `factory/v0.7/p7/historical_readmission.json` 的重新准入决定；
- 36 条新增或替换任务必须逐条通过当前质量合同和全新 Admission；
- deferred、draft、环境不可用、仅静态校验或只有上游测试的候选均不计数；
- 不为凑齐 50 条降低复杂度、运行真实性、Prompt 或重复度标准。

`factory/v0.7/p8/accepted_tasks.json` 是新增任务的当前机器可读进度账本。
其中 `required_task_count=36` 是完成条件，`task_count` 只表示已正式接纳的数量。

## 3. 合理而不过细的 Taxonomy

每条任务使用四个主要轴：

1. `contract_family`：结果、Tensor 元数据、状态/别名、梯度、API 行为或安全/效率；
2. `failure_type`：错误结果、意外异常、缺失异常、崩溃/卡死、不确定性或性能回归；
3. `execution_context`：CPU/CUDA、eager/compile、forward/backward、是否 distributed；
4. 可选 trigger：empty、scalar、极值、端点参数、特殊 layout、混合精度、动态
   shape、alias 或设备特有条件。

`contract_detail_tags`、`root_cause_tags` 和 `component_tags` 只在事实明确且有助于
检索时填写，允许为空。Boundary、Precision 和 Device 是从主要轴导出的分析
切片，不再作为互斥的顶层数据桶，也不要求为覆盖表格而复制任务。

## 4. Prompt 与 Agent 可见边界

Agent 只接收 `AgentTaskView`。正式 Prompt 必须：

- 描述用户可观测的失败、输入条件和期望行为；
- 不出现答案文件路径、内部符号、修复方向、Gold/Hidden Test 或 PR/Commit；
- 不通过“建议查看某目录/文件”变相缩小定位范围；
- 不复述上游 PR 中的 root cause、代码片段或修改步骤；
- 允许给出必要的公开 API、设备和运行模式，但不得把实现细节伪装成约束。

每条任务在正式 Admission 身份闭合后，必须对最终渲染 Prompt 与私有答案索引执行
重叠扫描，并完成独立语义复核。修改 Prompt、Gold、Hidden Test、Patch Scope 或
selector 后，旧复核立即失效。

## 5. 难度与任务价值

最终集合只接受 medium 或 hard，不接受只需机械改一行、从报错直接复制修复、纯
重命名/格式化、测试维护或没有公开行为影响的任务。

复杂度从三方面评分，每项 0–2：

- localization：需要跨抽象层、生成路径或非显然入口定位；
- diagnosis：需要理解多个运行时/算子语义或区分相近根因；
- repair/regression：修复需要维护相邻路径、设备、dtype、shape、alias 或兼容性。

总分 5–6 可作为 hard 候选；总分 4 只有在 blind pilot 接受且第二位 reviewer
确认后才能作为 medium。总分不超过 3、命中 hard rejection 或与现有任务语义
重复时必须淘汰。Gold 行数本身不决定难度，但单文件、小补丁和参考 Agent 全部
快速解决都属于风险信号。

## 6. Hidden Test 与 Admission

每条新增或替换任务至少包含：

- 一个直接证明目标失败的 F2P；
- 一个保护相邻正常路径的 P2P；
- 行为断言，而不是只搜索生成源码、只检查退出码或断言实现私有名称；
- 与 Base Commit 完全一致的源码身份；
- 与目标代码路径兼容的冻结 Runtime；
- Baseline F2P 失败、P2P 通过；Gold 后 F2P/P2P 全通过；
- collected/executed/skipped、failure signature、Patch 与所有内容 Hash 的闭合证据。

静态 `git apply --check`、语法检查和上游测试只能作为草案门禁，不能替代真实
Admission。Archive 或 synthetic snapshot 若 HEAD/Git tree 不等于声明的 Base，
必须重建或明确保持 deferred。

## 7. Agent 实验合同

最终实验采用 fresh、相互独立的 Attempts：

- 36 条新增或替换任务：每条 3 次；
- 14 条 retained historical：每条 1 次；
- 最终期望矩阵：`36 × 3 + 14 × 1 = 122` 个有效 logical Attempts。

每个 Attempt 必须绑定同一冻结 Dataset、Task、Prompt、Agent/Model、Runtime、
预算、Evaluator 和评分身份。基础设施无效或环境不可用的记录保留审计，但不进入
有效分母；补跑必须通过 append-only retry/ledger 选择一个有效 logical Attempt。

实验报告必须给出 resolved、F2P/P2P、no-patch、Agent/Runtime/Evaluator/
Infrastructure failure、重试和按主要 Taxonomy 轴的分布。结果仍是当前配置的
描述性证据，不自动构成跨模型排行榜或反馈因果结论。

## 8. Trace 范围

本轮继续保留现有工具调用、Action、Adapter、Evaluator 与 Integrity trace，证明
评测过程可审计。完整 Agent 自然语言输出、隐藏推理过程和将 trace 纳入评分的方案
暂不进入 v0.7 完成条件，避免在数据扩充主线中引入额外隐私、协议和评分变量。

## 9. 发布完成门

v0.7 重新标记 Completed 前必须同时满足：

1. 最终累计 Dataset 恰好 50 条，14 retained + 36 new/replacement；
2. 50/50 Task Manifest、源码、Runtime、Admission、Prompt 与复杂度证据有效；
3. 122/122 logical Attempts 有有效结果，或缺失项被补跑至完整矩阵；
4. Dataset、Agent 实验 summary/report 可以从原始账本确定性重建；
5. 全量 replay、Integrity、隐私扫描、tracked JSON、文档链接和测试通过；
6. Dataset Card、CHANGELOG 和发布 Hash 更新为最终 50-task 身份；
7. 不存在把 draft、旧 25-task 基线或静态检查误写成当前完成证据的表述。
