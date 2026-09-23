# CUDA LogSoftmax 尾归约迁移候选

**状态：未正式准入；完整框架构建及修订 2 的五组 CUDA 控制已完成，全部符合预期。**
2026-09-09 的[实际控制记录](validation/runtime_controls_r2.json)保留来源、镜像、设备、
逐 case 数值结果及离线核验。它们是作者控制实验，不是真实模型成绩或独立人工准入。

上游基线固定为 `2409b49a33c0ef594d89f9f477d56abad47e65bf`，源码版本为 `2.6.0a0`。只允许控制器从该提交及其递归子模块导出完整配置框架源码；求解工作区不包含上游 Git 历史、此任务包的私有参考值或控制补丁。不能直接复制其他 HEAD 的缓存工作树来代替指定提交导出。[源码与子模块检查](validation/source_inspection.json)记录 37 个顶层、73 个递归 gitlink 均有匹配 checkout；后续完整构建的实际记录见上文。

`task.json` 使用 schema 2，当前任务/评分修订为 2，环境修订保持 1，范围为算子本体。`public/` 中的 worker 和复现程序通过公开 `prepare` 命令写入工作区的 `opbench_public/`；候选可修改这些自测副本。正式评分使用冻结任务命令内的公开 driver 正文，直接调用构建后的 Torch API；自测副本不决定评分入口。没有私有评分代码参与环境准备或候选构建，没有按参考补丁文件列表限制修改范围。

## 准备和运行

依赖镜像由 [Dockerfile](environment/Dockerfile) 与 [构建配方](environment/build_recipe.json)声明。基础镜像是纯 CUDA 工具链 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`，增加 Python 3.10 和构建依赖，不安装上游 torch wheel。镜像需要在评测前显式准备，评测本身保持无网络：

```console
docker build -t opbench/pytorch-cuda-source:cu124-sm70-r1 -f datasets/v0.8_candidates/log_softmax_cuda/environment/Dockerfile datasets/v0.8_candidates/log_softmax_cuda/environment
opbench evaluate --task datasets/v0.8_candidates/log_softmax_cuda/task.json --patch datasets/v0.8_candidates/log_softmax_cuda/controls/baseline.patch --output .opbench-runs/cuda-logsoftmax/baseline
```

执行需要声明的完整源码及 GPU 条件。缺少源码时，应由控制器明确执行准备/下载；已有源码和子模块无需重写。`python3 setup.py build` 编译完整配置的 CPU/CUDA 框架，产物位于工作区，不能用已安装 wheel 加单文件 overlay 代替。CPU 对照可从同一提交另建干净工作区并设置 `USE_CUDA=0`，但 CPU 通过不满足此 CUDA 任务。显式源码准备、基线复用及五控制命令见[GPU 主机执行指南](../../../docs/v0.8/remote_execution.md)。

环境按 12 CPU、96 GiB 内存、SM 7.0 编译目标配置。当前部署的 `gpus=device=2` 只是远端设备选择，管理员应在运行前按实际 GPU 调度显式绑定并保存设备身份。已发现的远端设备为共享 Tesla V100-SXM2-32GB，不能按旧主机别名当作 A10，也不能据共享 GPU 的结果报告独占性能。首轮完整构建需要较大磁盘与时间预算；CPU、CUDA 大构建应按机器可用资源顺序调度。

## 评分与控制

三个 numeric case 合计执行 19 个小型操作。批处理用于减少完整构建工作区的跨会话复制；每个 case 仍只收到当前输入批次，在新的无 grader 会话中执行，输出展开后的实际数值。`grader/segments.json` 给出各操作在展开向量中的位置。

正式命令为 `python3 -I -c SCRIPT WORKSPACE`。`SCRIPT` 来自公开
[`numeric_worker.py`](public/numeric_worker.py)，仅把 `main()` 的工作区定位改为
`Path(sys.argv[1]).resolve()`；它不从工作区再次读取或导入 `opbench_public/numeric_worker.py`。
driver 内没有 oracle、期望值或通过判定，仍只接收当前输入。任务要求修复 Torch API，
只在桥接脚本中另算数学公式不是 API 修复。修订 2 保持原 `prepare` 和构建命令不变，
仅修正题面及评分入口；旧评分修订的结果不应混作新入口的控制证据。

| Case | 已由修订 2 基线/正确控制确认的分组 | 覆盖 |
| --- | --- | --- |
| `double_tail_values` | F2P | float64 宽度 513/515/517、非连续输入、同执行链的 softmax、大正偏移 |
| `float32_forward_regressions` | P2P | float32 宽度 511/512/513/517、非连续、softmax、另一维/负维度、大有限正负值 |
| `double_neighbor_and_gradient_regressions` | P2P | float64 511/512/514、另一维、有限极值、softmax/log_softmax 的解析一阶梯度 |

私有 oracle 由 90 位十进制精度的独立标量数学计算生成，包含稳定 log-sum-exp、概率和解析梯度；不导入 torch 或候选代码。float64 容差为 `1e-10 + 1e-11*abs(reference)`，float32 为 `2e-5 + 2e-6*abs(reference)`，两个精度分批比较，避免用 float32 容差掩盖 double 错误。输入为精确二进制分数与有限偏移，float32 在参考端先按目标输入精度取值。

`controls/reference.patch` 是上游两处剩余长度修正；`alternative.patch` 在对齐前导段耗尽输入时直接结束对应 helper；`mutation.patch` 在参考修复上错误地令 log_softmax 返回均匀对数概率。后者能满足归一化性质，却不满足真实逐值语义。实际完整构建后的 CUDA 控制已分别验证正确修复和错误变体的表现；[控制审查](migration_review.md)说明取舍和剩余风险。

新增 [`bridge_only.patch`](controls/bridge_only.patch) 只修改工作区 worker，以标准库
计算公开数学公式，完全不修 Torch。它用于确认评分入口不会被自测桥接副本替代：
在完整 CUDA 基线上，该补丁保留了原 API 缺陷：F2P 不匹配数与空补丁一致，两个
P2P case 通过。修改自测桥接副本没有成为成功修复。

可在无 GPU 的控制器运行入口小例：

```console
PYTHONPATH=src python3 datasets/v0.8_candidates/log_softmax_cuda/controls/check_driver_entrypoint.py
```

小例把 bridge-only 补丁应用到准备命令生成的自测副本，确认该副本能输出三个批次的
数学值；冻结 driver 在同一无构建工作区仍拒绝缺失框架，且不产生观测。它只验证入口
选择与缺构建前置条件，不证明真实 CUDA API 控制已通过。

此前的 [oracle 作者检查](validation/oracle_review.json)、[补丁适用性](validation/patch_applicability.json)
和源码检查仍保留。随后在共享 V100 上完成可信基线构建（2113.620 秒），并为每个
控制独立应用补丁、正常构建和评分：

| 控制 | F2P | 两个 P2P | 实际判定 |
| --- | --- | --- | --- |
| baseline | 失败，4,629 个值不匹配 | 均通过 | 原缺陷复现 |
| reference | 通过 | 均通过 | 修复成功 |
| alternative | 通过 | 均通过 | 替代修复成功 |
| mutation | 失败 | 均失败 | 错误的均匀输出被拒绝 |
| bridge_only | 失败，4,629 个值不匹配 | 均通过 | 仅修自测桥接不能修复 API |

五组的观测均有效，离线核验全部通过，记录中的清理错误均为空。离线核验使用已保存
观测重新计算数值判定，不导入候选代码。原始证据与执行软件副本的位置记录在
[runtime_controls_r2.json](validation/runtime_controls_r2.json)。公开复现、原生 Agent
canary 和真实模型实验不能由这些控制结果代替；后续进度见[项目验证记录](../../../docs/v0.8/validation_report.md)。

## 执行证据的边界

固定的公开 driver 从当前 `build/lib.*` 加载完整 torch 包及 `_C` 扩展，检查映射的框架共享库来自候选工作区，要求可用 CUDA、SM 7.0、CUDA 输出并执行同步。它不以日志文本或库文件存在作为数值判题依据。最终比较在候选会话停止后由独立控制器完成。

这些路径、设备和同步诊断用于验证普通实现是否走到了预期运行环境，不构成任意恶意候选不可伪造设备/dtype 元信息的证明。固定 driver 仍与导入的候选 Torch 共享进程，刻意拦截导入、修改运行时对象或伪造元信息的问题尚未普遍解决。设备选择或调度路径的高风险替代解仍需审查；CPU 计算后复制到 CUDA 的规避不能仅凭 `.is_cuda` 被当作合格 CUDA 算子实现。性能、半精度、编译图、分布式和高阶梯度均未纳入本候选验收。

`authoring/assemble_task.py` 已同步生成修订 2 的固定公开入口；重新组装会保留同一公开
准备内容和构建配方。metadata 只以 `validation_records: "validation/"` 指向执行记录，
不复制容易过期的运行状态。修改公开 driver 或任务合同后仍须重新审查评分修订与控制证据。

`task.json`、`grader/`、`controls/`、`authoring/` 与评测输出属于控制器侧资产，不能整体挂给 Agent。学校对项目发布许可证的决定仍待确认；上游代码及参考修改的来源和原许可证应随正式发布保留。

## 原生提交与独立复评

[完整源码原生验证记录](validation/native_replay_r4.json)记录了一次已知修复脚本：
工作区只有一个基线提交、没有 remotes，原上游 commit object 不存在；脚本完成源码
修补、正常构建、公开复现和本地 commit，控制器随后冻结了对应补丁。原评分保留
两次 CUDA 显存不足及一组通过；对同一补丁的独立复评三组全通过，离线核验有效，
没有清理错误。原失败没有改写，复评没有重新运行 Agent，也不增加独立重复。

共享主机的复评期显存观测曾仅剩 227 MiB；该观测不能证明原失败时的根因。
这些记录用于验证工程流程，不是模型能力结果或正式任务准入；正式 GPU 实验仍需
明确并控制设备占用与资源条件。
