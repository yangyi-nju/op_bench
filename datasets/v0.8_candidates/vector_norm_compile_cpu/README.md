# CPU 编译 vector_norm 溢出候选

**四个真实控制均符合预期，仍未准入。** 物理结果见[控制记录](validation/runtime_controls_r1.json)。本题要求实际执行 CPU Inductor 编译后
的 `torch.linalg.vector_norm`，验证算子数值和元数据，不以调用内部 helper 代替编译。
来源为 [PyTorch #144073](https://github.com/pytorch/pytorch/pull/144073)；迁移基线和
执行条件与旧题不同，旧 `verified` 或 wheel overlay 结果不继承。

## 问题与合同

当归约轴只有一个元素时，非零有限阶数的范数应为该元素的绝对值。原分解却先算
`abs(x)**ord` 再开方：float32 的 `16**41` 可溢出，`16**-41` 可下溢，从而给出
非有限值或零。公开复现使用 `(4,1,4)` 输入、`dim=1, ord=-41`，同时包含正负
`16` 与 `0.0625`，不依赖随机数或旧 issue 中不足以触发 float32 溢出的 `[2.0]` 示例。

题面声明有限、非空、rank 2/3 的 CPU float32/float64 输入、非负归约维度及
`keepdim`。单元素归约测试 `ord=±41`；普通回归覆盖 `-1,0,1,2,±inf`、多元素、
全归约、非连续布局及 float64。复杂数、低精度、动态/空 shape、负维度、梯度、
显式 dtype 转换和性能不在该候选合同中。

## 题面修订与既有证据

当前任务修订为 `candidate.2`，评分仍为 `candidate.1`。本次只澄清容差句子的
适用范围：float32 单元素归约中 `ord=-41/+41` 的比较使用
`abs(actual-reference) <= 1e-6 + 1e-6*abs(reference)`；其他 float32 检查使用
`2e-5 + 2e-6*abs(reference)`，包括回归组的 `ordinary_singleton`、`ord=2`。
float64 容差不变。实现、输入、oracle、评分、公开命令、源码和环境均未改变。

本轮已完成的 `candidate.1` 控制继续保留原任务身份；它们可支持未改变的实现、
输入与评分行为，不能重标为 `candidate.2` 的物理执行。本次没有新增 Agent 证据，
也不把题面变更后的尝试自动并入旧实验。单题数据清单升为 `candidate.2`，
[聚合候选清单](../dataset.json)升为 `0.8.0-candidate.2`；原计划保留原数据身份。

| Case | 分组 | 内容 |
| --- | --- | --- |
| `singleton_reductions` | F2P | 批次单元素轴、多个单元素轴、keepdim、非连续输入及全局单元素 |
| `float32_regressions` | P2P | 普通范数、多元素负阶、零阶、无穷阶及正常单元素归约 |
| `float64_regressions` | P2P | ±41 单元素归约及多轴 L2 |

独立 oracle 由标准库 Decimal 的 80 位标量计算生成，单元素直接使用绝对值，再
按输出 dtype 舍入；不导入 Torch。公开固定 transport 对每个输出返回
`[is_finite, finite_value_or_zero]`，因此原溢出仍形成有效协议观测，由控制器依据
私有参考拒绝。shape、dtype、CPU 设备另由固定 transport 检查。

## 完整源码与编译证据

新题明确使用 `240aa77ad01c4f0cd9b2417748272f2f617c112f`（与 LayerNorm 候选相同）。
原 PR parent 是 `06e9deabb623e004eb6024e703a976c5748d51e6`；两个基线的
`vector_norm` 函数经静态检查相同，参考补丁可应用；本轮四个控制已在新基线的完整框架上实际构建、编译和评分。
详见[迁移审查](migration_review.md)。

`environment` 完整复用 [LayerNorm r3](../layer_norm_cpu/task.json)：依赖镜像仍为
`opbench/pytorch-layernorm-source:240aa77-cpu-r2`，环境修订为
`240aa77-cpu-gcc12-openblas-r3`，禁用 NNPACK/FBGEMM。环境名字标识同一构建输入；
这不是一道 LayerNorm 题。prepare/build 和镜像不插入本题 worker、grader 或答案。

评分从完整 `build/opbench-package` 加载 Torch 模块、扩展和框架库，再执行
`torch.compile(fullgraph=True, backend="inductor", dynamic=False)` 返回的函数。
每个 case 使用工作区 runtime 内的新编译缓存，关闭图缓存和错误抑制，并观察
生成的 CPU kernel 数及加载的缓存动态库；不能直接返回 eager 结果冒充编译结果。
这些计数与路径是普通编译/加载证据，与候选 Torch 共享进程，可被恶意代码伪造；
它们不是普遍的执行归属认证。最终数值判定仍在候选会话结束后的独立控制器进行。

## 实际控制与审查

[2026-09-09 控制矩阵](validation/runtime_controls_r1.json)复用完整 CPU r3 可信基线，
每个补丁仍执行正常构建。四个控制都通过离线结果校验，无清理错误；所有 case
进程正常退出并给出有效数值观测，失败来自数值不符。

| 控制 | F2P 不符编码条目 | float32 / float64 P2P | 评价耗时（秒） |
| --- | --- | --- | --- |
| baseline | 40 / 126 | 均通过 | 1295.530725 |
| reference | 0 / 126 | 均通过 | 651.375204 |
| alternative | 0 / 126 | 均通过 | 571.167050 |
| mutation | 38 / 126 | 均通过 | 565.974419 |

每个输出由有限标志和值两个条目编码，**126 个编码条目对应 63 个算子输出**。
三个批次共 **230 个编码条目、115 个输出、16 项操作**，属于一个逻辑任务；
不符数不是错误输出数或失败任务数。四个控制的三个批次分别记录生成
`6 / 7 / 4` 个 kernel，并加载 `6 / 7 / 3` 个编译缓存动态库。这是批次级见证，
不证明每项操作都有独立内核，也不是对恶意候选的普遍执行认证。

另一次 AI 辅助独立 oracle 审查重算了 16 项操作，230 个参考编码条目全部精确一致；
发现的容差文字歧义已由 `candidate.2` 澄清。对已有 `ordinary_singleton` 观测
补做更严格容差的诊断复算，四个控制均通过；这不改变保存的分数，也不是物理重跑。
审查仍不是独立人工准入；有限判例没有覆盖声明域的全部组合，负阶零输入仍未测试。
本轮没有新增真实模型、GPU、梯度或性能证据。

## 运行控制

先完成[远端执行指南](../../../docs/v0.8/remote_execution.md)中的控制器、完整源码
和镜像准备。使用同源码、同环境的 ready CPU r3 基线，以下路径为部署占位符：

```bash
opbench check-task \
  --task datasets/v0.8_candidates/vector_norm_compile_cpu/task.json \
  --reference datasets/v0.8_candidates/vector_norm_compile_cpu/controls/reference.patch \
  --alternative datasets/v0.8_candidates/vector_norm_compile_cpu/controls/alternative.patch \
  --mutation datasets/v0.8_candidates/vector_norm_compile_cpu/controls/mutation.patch \
  --baseline-artifact /srv/opbench-runs/cpu-r3-baseline \
  --output /srv/opbench-runs/vector-norm-compile-controls
```

若没有 ready 基线，先用本题 `build-baseline` 构建。每个控制仍独立应用补丁并运行
正常完整框架构建。私有评分首次触发的 Inductor 编译属于独立评分；Agent 自测编译
属于求解预算。共享 CPU/GPU 主机的运行仅用于正确性，不提供性能结论。

reference 是上游单元素特化；alternative 以单元素轴的 `abs` 后归约保留 shape；
mutation 只修 `x.numel()==1`，遗漏批次中的单元素归约。实测 baseline/mutation
失败 F2P 且通过 P2P，reference/alternative 全通过。构建、导入、编译见证失败
不能被称为检测到了预期的数值缺陷。

`task.public_commands` 提供内联公开复现和完整构建入口，不需修改环境 prepare
来安装新文件。独立候选索引为 [dataset.json](dataset.json)，亦收录于[总候选清单](../dataset.json)。
[静态检查](validation/static_review.json)仅记录数学、补丁、生成器与结构检查；
物理控制结果另按执行时的任务修订保存，不代表真实模型成绩或正式准入。上游修改仍需保留 PyTorch
原有来源与许可；OpBench 自有许可证等待学校确认。
