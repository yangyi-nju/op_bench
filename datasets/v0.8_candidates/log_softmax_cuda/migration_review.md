# #144009 迁移审查记录

日期：2026-09-09。结论：**完整源码构建及修订 2 的核心四组、桥接控制均已符合预期，仍未正式准入。**
[实际执行记录](validation/runtime_controls_r2.json)包含五组结果及全部有效的离线核验；
这些是共享 V100 上的正确性证据，没有真实模型成绩或性能结论。

## 上游事实与基线

[官方 issue #143644](https://github.com/pytorch/pytorch/issues/143644)报告 float64 CUDA softmax/log_softmax 在某些较宽维度产生错误数值，最初复现宽度为 517。[官方 PR #144009](https://github.com/pytorch/pytorch/pull/144009)在 2025-01 修正对齐前导段之后的剩余长度处理；[官方 diff](https://github.com/pytorch/pytorch/pull/144009/files)修改 `ilpReduce` 与 `WriteFpropResultsVectorized`，增加 `(5,513)` double 的归一化回归。

本任务保留历史指定基线 `2409b49a33c0ef594d89f9f477d56abad47e65bf`，在该提交的完整源码上工作。[该版本构建说明](https://github.com/pytorch/pytorch/blob/2409b49a33c0ef594d89f9f477d56abad47e65bf/README.md)要求递归子模块和 CUDA 构建依赖。本次使用 `setup.py build` 将整个配置框架的 Python/native 产物留在工作区，符合两阶段会话的产物契约；不执行把包安装到系统 site-packages 的 `develop/install`。

代码检查表明，对齐前导段有可能已经处理完一行，但仍减去整个 `blockDim.x`，使剩余长度计算无效。历史描述笼统称作“unsigned underflow”不够准确：此基线相关参数也使用有符号 `int/index_t`，并与 CUDA 无符号维度成员参与运算；审查重点是长度不应越过已耗尽边界，而不是预设某一种整数类型解释。

## 新评分相对于旧题的实质变化

旧 hidden test 只检查 `exp(log_softmax).sum == 1`。这不能排除输出均匀分布等错误修复。新任务比较私有独立数学 oracle 中的每个数值，分开 float64 与 float32 容差，并检查解析梯度。shape/dtype/device 的普通运行检查在公开 worker 中；数值预期与判定不进入候选进程。

任务/评分修订 2 修正了另一处边界：工作区 worker 是可编辑的自测副本，不能同时
充当最终评分入口。正式评分改为执行冻结任务命令中的同一公开 driver 正文，仅以
argv 提供工作区位置；由它加载并调用构建后的 Torch API。driver 不含私有参考值或
判定，但修改自测桥接副本不再替代 API 修复。环境 prepare/build 保持不变。固定
driver 与候选 Torch 仍共享 Python 进程，因此不宣称解决任意运行时拦截或 CUDA
计算归属认证。

三组 case 的 19 个输入均为小矩阵，不借题目扩展为性能测试或大张量内存压力测试。F2P/P2P 分组先由执行路径分析提出，现已由完整 baseline 的 F2P 失败、两个 P2P 通过及正确控制全通过确认；本轮没有为制造通过而修改分组。

绝对/相对容差预先写入题面及 oracle。double 使用 `atol=1e-10, rtol=1e-11`；float32 使用 `atol=2e-5, rtol=2e-6`，允许普通归约与输出舍入，但与本缺陷的明显数值偏差及均匀输出错误相隔较远。本地标量 `math.fsum/exp/log` 对 Decimal oracle 的最大差约为 `3.6e-15`；小矩阵梯度中心差分误差低于 `7e-11`。随后 reference/alternative 在声明的 GPU 输入和容差下全部通过，baseline 与错误变体被数值比较拒绝；这一证据限于当前域和环境，不推及其他精度或硬件。

## 控制补丁与替代解

| 控制 | 改变与预期 | 完整 CUDA 控制的实际结果 |
| --- | --- | --- |
| baseline | 空补丁；尾部 double 数值应失败，已有回归应通过 | F2P 有 4,629 个不匹配值；两个 P2P 全通过 |
| reference | 两个前导段对剩余长度的递减均不超过现有长度；其他算法保持原有行为 | 正常构建并加载，三个 case 全通过 |
| alternative | 对齐前导段已经耗尽整行时，在 reduction 返回已累积的局部值，在 output helper 直接返回；有剩余时维持原循环 | 正常构建并加载，三个 case 全通过；不要求沿用参考写法 |
| mutation | 包含 reference，再把 `F.log_softmax` 输出改为 `ret*0 - log(width)`，保留形状、CUDA 张量和可微连接，但忽略输入差异 | 三个 case 均产生有效观测后数值失败，不是语法、构建或缺输出失败 |
| bridge_only | 只把工作区自测 worker 改为计算公开标量公式，不改 Torch API；在完整基线上应保持原缺陷 | 与 baseline 相同的 F2P 不匹配数，两个 P2P 通过；没有绕过固定评分入口 |

五组结果的离线核验全部有效，清理错误均为空。可信基线由任务/评分修订 1 生产，
修订 2 的环境、源码和 prepare/build 没有变化，因此复用其构建；每个控制仍执行
正常候选构建，结果记录当前任务/评分修订 2。生产者身份与控制身份分别保留，
不能把旧修订评分冒充本轮证据。

这些控制只用于准入审查，不构成候选修改权限。任何保持声明语义与执行合同的其他源代码、构建或公开 worker 修改都可被提交，但编辑自测 worker 不改变正式 driver。若替代解改为复合 ATen CUDA 算法，必须依据正确性及真实执行链路审查；不要求与参考修改同文件、同 helper 或同写法。若转到 CPU 计算或使用固定测试输入表，则须单独识别其违反 CUDA 执行合同或泛化语义的风险，不能由统一输出协议自动证明其合格。

## 环境与发布缺口

- 当前远端硬件事实为四张共享 V100-SXM2-32GB；构建目标 SM 7.0 与此匹配。调度器选择某张设备并不表示独占。设备 UUID、实际镜像 ID、CUDA/toolchain 版本应随真实结果保存。
- 依赖镜像不含预装 torch；完整候选重新生成自己的 CPU/CUDA 库。禁用的 cuDNN、attention、分布式、MAGMA 及其他可选后端不属于本题 softmax 执行链。若必须改环境选项或依赖版本，需记录实际配方与环境修订，不能只保留原标签。
- 只读根、无网络、工作区产物及模型信息边界由核心执行器负责。公开 worker 的导入/设备诊断不是对恶意代码的通用证明；私有 oracle 的数值比较已经与候选进程分开。
- 五组完整源码控制和离线数值复核已完成；原生 canary 的最终结果、真实模型试验、独立人工合同/替代解审查及正式数据准入仍不能由控制组结果代替。当前没有真实模型调用或能力成绩；后续执行进度由[项目验证记录](../../../docs/v0.8/validation_report.md)维护。
- 学校对项目发布许可证的决定未完成。正式发布需保留上游 PyTorch 来源与许可义务，不在本次迁移中替学校作法律决定。
