# #144073 编译题重定位审查

## candidate.2 题面澄清

AI 辅助独立 oracle 审查重算 16 项操作，230 个参考编码条目全部精确一致；该审查
仍不是独立人工准入。审查发现，`candidate.1` 的容差文字将严格 float32 容差笼统写成
所有 singleton 比较，但 `ordinary_singleton` 的 `ord=2` 属于普通回归组。
`candidate.2` 仅将该句明确为 `ord=-41/+41` 的 float32 singleton 比较使用
`1e-6 + 1e-6*abs(reference)`；其余 float32 比较继续使用
`2e-5 + 2e-6*abs(reference)`。这澄清了原先针对 ±41 目标组的意图，未改变实际
oracle 容差、数学、输入、worker、补丁、构建或公开命令。`scoring_revision`
仍为 `candidate.1`，源码与共享 CPU r3 环境不变。

重新生成 `task.json` 后，按解析字段与编辑前对象比较，仅 `statement` 和
`task_revision` 改变。单题与聚合数据清单分别升为 `candidate.2` 和
`0.8.0-candidate.2`，保留原执行计划与[版本化验证记录](validation/)的身份。
历史 `candidate.1` 控制可支持未变的实现、输入和评分行为，但不是物理
`candidate.2` 执行；本次未增加 Agent 运行，也不证明新旧题面下的模型结果可直接混算。

## 来源和基线决定

旧素材：[task.json](../../../tasks/pytorch/144073_vector_norm_scalar_overflow/task.json)、
[参考补丁](../../../tasks/pytorch/144073_vector_norm_scalar_overflow/artifacts/gold.patch)、
[隐藏测试](../../../tasks/pytorch/144073_vector_norm_scalar_overflow/artifacts/hidden_test.patch)。
原修复 `957faaadca78ec453d60f2fe986c1191e2e7c5b6` 的 parent 确认为
`06e9deabb623e004eb6024e703a976c5748d51e6`，2025-04-07 的源码版本为 `2.8.0a0`。
其要求 SymPy >=1.13.3，不能直接假定当前 2.6 环境与其一致。

本候选改用已准备的 `240aa77ad01c4f0cd9b2417748272f2f617c112f`：2024-11-19、
`2.6.0a0`、SymPy 1.13.1。Git 导出内容显示两个版本的 `vector_norm` 函数 AST
相同，但整个文件/框架不同；Inductor 的 decomposition 表显式包含
`aten.linalg_vector_norm`，原参考补丁在该树对应文件上可直接应用。
该静态比较支持重定位可行性；下述真实控制另行确认了新基线的编译后数值失败。
任务身份、题面和 metadata 均公开重定位；不能将原历史基线或 overlay 成绩冒充本题。

环境/source 字段完整沿用当前 LayerNorm r3，使兼容的可信 baseline 可复用；
不同题面、评分和私有参考不会参与该构建。新 driver 通过冻结任务命令内联执行，
公开复现同样内联，未为了放置题目文件修改 prepare/image。最终仍执行正常 build。

## 相对旧题的变化

- 旧测试依赖内部 `self.common`、随机 `(4,1,4)` 输入及无关的普通加法 P2P；新包
  直接编译公开 API，以固定极值和与范数相关的 shape/阶数/dtype 回归组成合同。
- 原 `issue.md` 的 `[2.0], ord=-41` 本身不发生 float32 次幂下溢；新公开复现用
  `±16` 与 `±0.0625`，数学上越过 float32 可表示次幂范围，真实控制已观察到编译后数值不符。
- 参考值来自独立 Decimal 标量数学，不把 eager 当作唯一判定。非有限输出显式
  编码为有限标志与占位值，便于保留溢出证据和离线复核。
- 固定公开 driver 记录完整源码加载与真实 Inductor 产物加载；不依据 Gold 文件
  名单、源码文本或内部 helper 写法判题。候选进程内见证存在可伪造限制，需单独审查。

## 控制与剩余缺口

reference 保留上游分支特化；alternative 用 `sum(abs(x), dim, keepdim)` 处理单元素
归约，提供不同于 reshape/squeeze 的合法写法；mutation 仅修全局一个元素，仍漏
批次单元素轴。[实际控制记录](validation/runtime_controls_r1.json)中的四个物理执行
均为任务 `candidate.1`、评分 `candidate.1`，不是当前题面 `candidate.2` 的重跑。

基线 F2P 有 40/126 个编码条目不符，mutation 为 38/126；reference 和 alternative
三个批次全部通过，两个 P2P 批次在全部四个控制中均通过。所有 case 进程正常退出，
数值观测有效，四份离线校验通过且无清理错误。评价耗时依次为 1295.530725、
651.375204、571.167050、565.974419 秒，包含独立工作区与构建等评价开销。

F2P 的 126 个编码条目代表 63 个输出；全题三个批次合计 230 个编码条目，代表
16 项操作的 115 个输出，不能把编码不符数当作错误输出数或独立任务数。每个控制
都记录三个批次生成 6/7/4 个 kernel、加载 6/7/3 个 fresh 编译缓存库。这确认了
这些批次出现实际编译产物加载，但不保证逐操作执行归属，也不消除候选进程内见证
可被伪造的限制。已有 `ordinary_singleton` 观测按严格容差补做诊断比较也全部通过，
不修改原分数或增加物理重复。

有限判例未穷尽声明域，负阶零输入仍未测试；域外 dtype、负轴、梯度、动态 shape
和性能也未覆盖。正式准入还需独立人工合同/替代解审查、稳定性与来源/分发许可证据；
本轮 CPU 编译控制没有运行真实模型或 GPU，不提供性能结论。
