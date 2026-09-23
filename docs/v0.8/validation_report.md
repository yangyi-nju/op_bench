# v0.8 验证报告

更新于 2026-09-23。当前验收范围为：固定模型循环、受控 MCP 工具、独立
评分和唯一主指标 `resolved_rate`。逐项状态见[验收矩阵](acceptance_matrix.md)，
下文按日期记录实际验证，不把旧结果重标为新协议的实验。

当前协议为工具 `controlled-tools-2`、循环 `opbench-fixed-tools-3`、计时 `3`、
报告 `5`。最新完整回归是 9 月 22 日的 224 项测试；9 月 23 日仅补充注释、文档
并核对接口，未开展新的真实模型或 GPU 实验。目标提供商仍需在正式实验前逐个联调。

同目录的 `validation*.json` 是各自日期的原始快照，不是自动更新的“最新汇总”。
其中 [validation_summary.json](validation_summary.json) 属于 9 月 7 日；9 月 9 日
[旧原生 Codex 记录](validation_codex_20260909.json)属于已经退役的执行流程；
[9 月 14 日固定循环记录](validation_controlled_20260914.json)采用当时的工具版本。
原始 JSON 和日志中的旧路径、软件身份与结论保留不变。

## 2026-09-22 测试夹具与文档入口收敛

工程小题统一迁入 `tests/fixtures/`，保留数值、C++ 编译、PyTorch 集成三个范围；
删除独立 examples 入口、重复指南和未使用的数据/模型配置。数值 Docker 环境
revision 2 使用 Python/Git 镜像，夹具清单版本为 2；历史证据仍保留原环境身份。

- Python 3.12.13、轻量镜像 Docker 检查启用：224 项通过，无跳过，110.005 秒。
- 三套夹具各自的四组控制检查通过；12 份评分证据只读核验通过。
- 仓库外新 venv、非 editable 安装：迁移后的数值评分 resolved、verify 有效，
  主候选清单三题加载成功；其中两题源码本机尚未准备，未尝试拉取或运行。

证据位于 `runs/v0.8/validation/20260922-fixture-consolidation/`。本轮未调用真实模型；
C++ 与 PyTorch 使用已有镜像，只重建了数值镜像。CI 路径已同步，本机未执行
GitHub Actions 或 Python 3.13；历史版本文档和原始实验输出未改写。

## 2026-09-22 模块内部精简与运行记录整理

变体父评分与控制组使用 schema 2，仅引用独立子评分的 `result.json`，不再保存完整
子结果和展开的测试副本。四类控制组共用执行循环，删去可直接推导的计数。旧字段
别名在记录读取入口归一，报告协议为 5；重评要求原尝试有明确的冻结声明。统一
HarnessSpec 的构造校验、源码身份与本地进程环境规则，移除无消费者的基线返回字段
和 Codex 的重复 stdin 产物。活动源码由 5,573 行变为 5,513 行（含空行和注释）；
主要减少的是重复状态和产物，未新增服务层或通用框架。

- 最终核心回归：Python 3.12.13、Docker 启用，**224 项通过，无跳过，87.460 秒**。
- 仓库外新虚拟环境、非 editable 安装：4 次脚本化 HTTP 请求完成求解，评分 resolved；
  resume 没有新增调用，report、verify、原补丁 replay 和四组控制核验全部通过。
- 安装联调确认 `evaluation_path` 全部指向 `result.json` 文件；控制组没有内嵌
  cases/groups，原尝试未被恢复、报告或重评改写。最终安装证据为 `installed-r2/`。
- `installed-r1/` 和字段统一前的核心测试也通过，保留为本次修改过程中的记录；
  它们采用了短暂的目录路径字段，最终契约为文件路径，不把二者混为同一版格式。
- 运行目录按版本、用途与批次整理；搬迁 73 个目录，清理 14 个只含空目录的旧根目录。
  原始记录和四份历史 Markdown 保留，v0.8 旧路径通过本地链接定位。
- 从新路径只读复核 9 月 14 日真实 Codex 联调和上一轮安装实验：报告完整、各解决
  一题，独立评分核验通过。迁移没有调用模型或改写原证据，旧基线能否复用不作保证。
- 当前文档链接、历史导航与 `git diff --check` 通过；原始运行输出仍被 Git 忽略。

本轮证据统一位于 `runs/v0.8/validation/20260922-internal-cleanup/`，从其中的
`README.md` 进入。保存约定见[运行记录](../../runs/README.md)，旧路径去向见
[归档索引](../../runs/archive/README.md)。本轮未调用真实模型或重跑大型 GPU 构建。

## 2026-09-22 进一步架构与职责审查

保持同一个 Python 程序内的五个职责包，不新增服务层或插件框架。本轮集中解决
模块交叉、重复状态和已复现的实现缺陷，当前结构见[架构导读](../architecture.md)：

- `runner/experiment.py` 负责计划、调度与恢复；`attempt.py` 管理一次求解；
  `preparation.py` 管理可恢复的任务准备。基线路径重定位归 runtime，编排不改写其内部记录。
- 执行器与证据核验共用纯评分规则；核验不再导入执行器，报告不再自行实现变体判定。
- 命令执行和数值观测共用进程生命周期实现，统一超时、输出限制及子进程回收。
- 删除未使用的重复次数字段、工作区题面副本和内嵌循环结果；`attempt.json` 仅通过
  `harness_path`、`evaluation_path` 关联两种独立结果。控制组输出改为 `controls.json`，
  去掉固定的准入结论，完整数据准入仍属于 v0.9。
- 修复基线搬迁后本机运行环境仍指向旧路径、变体证据身份不一致仍被接受、损坏补丁
  编码和超大 duration 导致核验崩溃的问题。
- 各运行、准备和评分入口在写入前使用同一输出路径规则，拒绝写入源码或私有 grader
  及其符号链接别名，避免污染输入。
- 计划和尝试显式记录 harness 协议；未完成旧计划在工具或循环协议变化后不能继续
  调用模型。该轮结束时为 `opbench-fixed-tools-3`、`controlled-tools-2`、报告协议 4，
  已完成的历史记录保留原条件并可读取。

最终验证：

- Python 3.12.13、真实 Docker 启用：**220 项测试全部通过，无跳过，84.019 秒**。
  覆盖新边界、基线搬迁、进程回收、独立评分、恢复、报告和补丁重放；不做文件 hash 或数量断言。
- 仓库外全新虚拟环境、非 editable 安装并清除 `PYTHONPATH`：4 次脚本化 HTTP 请求
  完成读取、修改、公开测试与 finish，独立评分 resolved；resume 新增调用为 0，
  report、verify、原冻结补丁 replay 及 replay verify 全部通过。
- 安装后的 CLI 拒绝向源码目录输出，退出码 2 且未创建输出；四组评分控制符合预期，
  基线与错误补丁失败，参考与替代补丁成功，四组均通过离线 verify。
- 尝试记录没有重复内嵌循环结果、用量或评分；恢复、报告与重评没有修改原尝试。
- 静态导入审查无模块循环，runtime 无向求解、评分或报告的反向依赖。
  当前文档链接及历史导航检查通过，`git diff --check` 通过，历史版本原文保留。

本轮没有调用真实模型或重跑大型 GPU 构建。脚本化 HTTP 只验证工程链路，不作为
模型能力结果。证据位于本地忽略目录 `runs/v0.8/validation/20260922-deep-audit/`，
最终核心日志为 `core.log`，最终安装摘要为 `installed-final/summary.json`。
首轮测试因手工构造的旧测试计划缺少新增协议字段而失败，修正测试数据后完成回归；
原失败日志保留，不替换为成功结果。

## 2026-09-22 简化复核

保留五个职责包，删除批量外部补丁实验入口、内嵌评分缓存、报告版本链与未使用的
MCP stdio 入口；报告计算和工具参数校验各共用一份实现。实际依赖无 runtime 向
评分器或求解器的反向调用。具体职责、删减及接口变化见[架构导读](../architecture.md)。

- Python 3.12.13、真实 Docker 启用：**211 项测试全部通过，无跳过，83.790 秒**。
- 固定工具修改、冻结补丁、独立评分、计划恢复、报告重算与同补丁重评均通过行为验证。
- 旧内嵌评分不能覆盖独立证据；读取与重算不修改原尝试，新尝试与重放只保存评分路径。
- 整批工具参数在执行前验证：后续参数非法时，前面的文件写入也不会执行。
- 工具协议变化时，恢复会在调用模型前拒绝未完成的旧计划；完成的旧计划仍可读取汇总。
- HTTP 状态拒绝、错误内容类型、响应超限和读取超时均关闭响应；最终回归没有 ResourceWarning。
- 汇总只在结束或中断时执行一次；运行中按需手动 report，避免核验成本随尝试数平方增长。
- 当前 Markdown 链接与历史导航检查无断链，`git diff --check` 通过；历史原文保持不变。
- 仓库外全新虚拟环境、非 editable 安装并清除 `PYTHONPATH`：4 次脚本化 HTTP 请求完成
  读取、修改、公开测试与 finish，独立评分 resolved；resume 新增调用为 0，报告重算、
  verify 与原冻结补丁 replay 均通过。确认安装包包含本次工具版本恢复检查。
- 新尝试和重放的磁盘记录均不内嵌评分，独立结果存在；恢复、报告与重放未改变原尝试。

删除的测试仅针对退役功能，并补充上述新行为的回归，不用测试数量作为完成条件。
工具和循环协议递增到 `controlled-tools-2`、`opbench-fixed-tools-2`，旧记录保留原身份。
本轮没有调用真实模型或重跑大型 GPU 构建；下面的真实模型实验仍属于 9 月 14 日。
本地验证日志保存于 `runs/v0.8/validation/20260922-simplification/`。
最终安装证据位于其中的 `installed-final/`。首轮验证脚本在执行链路通过后读错摘要
字段，保留该错误记录；修正验证脚本并重新安装后完成最终确认，未将脚本错误记为模型结果。

## 2026-09-21 架构与文档整理

实现从平铺的 `benchmark/` 拆为 `data/`、`runner/`、`runtime/`、`evaluation/`、
`results/`，统一 CLI 为 `op_bench.cli`。补丁捕获和公共工作区约定移入 runtime，
基线构建不再调用评分器私有方法。CLI 命令、配置和结果格式保持不变，未保留旧
`op_bench.benchmark` 的内部导入路径。职责和阅读顺序见[架构导读](../architecture.md)。

恢复 56 份此前删除的版本设计、实施说明和实验报告，保留原文与原路径；
[历史导航](../history/README.md)说明当时的提交、命令及资产上下文。
当前文档和历史导航的链接检查通过；历史原文引用的退役资产不要求重新落入当前工作树。

- Python 3.12.13、Docker 启用：216 项核心测试全部通过，无跳过，耗时 87.386 秒。
- 全新虚拟环境安装、仓库外运行、清除 `PYTHONPATH`：CLI 入口可用，安装包无旧 benchmark 实现。
- 基线与错误补丁被拒绝，参考与替代修复通过，四组控制均通过离线 verify。
- 安装包通过本地 HTTP 脚本化响应完成读取、修改、公开测试和 finish，独立评分 resolved；
  总共 4 次响应请求，resume 无新增请求，报告重算和 verify 通过。
- 模块导入检查确认 runtime 不依赖模型循环、评分或报告；`git diff --check` 通过。

本轮没有调用真实模型或重跑大型 GPU 构建。脚本化响应只验证重构后的安装和执行链路，
不作为模型能力结果。原始日志与安装摘要保留在本地忽略目录
`runs/v0.8/validation/20260921-architecture/`。以下真实模型与框架证据仍属于原日期。

## 2026-09-14 真实模型联调

当时使用 `examples/controlled_models/experiment.docker.json`，数值题环境 revision 1
借用 PyTorch 镜像；原计划与任务快照保存在
`runs/v0.8/validation/20260914-controlled/`。该次联调
通过本地 Codex CLI **0.154.0-alpha.6.2 / gpt-5.5 / high** 完成数值 softmax 工程题。
模型显式选择与本机配置一致，未自动换模型；沿用已有登录，未复制凭据。

| 项目 | 实测结果 |
| --- | --- |
| 物理尝试 | `codex-controlled-r2 / attempt-000001`，单题一次 |
| 模型与工具 | `controlled-tools-1`；5 次模型调用、7 次工具调用；列出文件、读取源码、修改、公开测试、finish |
| 求解循环 | 52.961 秒；受共同预算限制 |
| 独立评分 | `resolved`；F2P 2/2、P2P 1/1，候选重新构建 |
| 运行边界 | Docker 网络 none，无私有 grader 与模型凭据，源码只有新建基线历史 |
| 报告 | planned=1、resolved=1、resolved_rate=1、complete=true |
| 恢复与复核 | resume、report、verify 均成功；模型请求仍为 5 次，无重复求解 |
| 资源回收 | 求解和评分无清理错误 |

模型生成补丁，OpBench 执行受控工具；没有向模型注入参考补丁或修复脚本。
CLI 在独立空目录返回 JSON 决策，不接管题目目录。原生工具关闭，已观察事件没有
原生工具执行。临时 provider 直接选择 HTTPS，不修改用户配置。事件保留 Codex
开发中特性提示；CLI 自报 token 用量保留，费用未知。

这是 9 月 14 日的工程接入测试，不是模型能力排名，也不是当前 tools 2 / harness 3
的真实模型验证。Codex 包装、结构化输出和自报用量不等同于
直接模型 API；CLI 不能硬限制 `max_completion_tokens`。正式多模型实验在 v0.10
固定接入条件后开展。该题只有三个工程判例，不构成正式数据准入或完整域覆盖。

早期 `codex-development-r1` 因默认 WebSocket 重试耗时，在已有模型动作后由控制器
主动中止，保留 `cancelled`、冻结补丁和原始轨迹，没有评分，也未改写为成功尝试。
r2 是修正传输后新建的 Docker 计划。r2 启动前曾因临时数据清单使用绝对路径被拒绝，
没有调用模型；修正为项目内可移植清单后运行。失败记录均保留在本地证据目录。

## 2026-09-14 行为回归

Python 3.12.13、实际 Docker 启用条件下，**216 项测试通过，耗时 80.962 秒，无跳过**。覆盖：

- 工作区路径、符号链接、`.git` 和私有资产访问限制；任务进程不继承控制器模型凭据。
- 源码新增、修改、删除、初始 Git 忽略规则、完整框架规模的补丁捕获与回放。
- 固定模型循环、MCP 调用、预算截止、工具会话关闭、服务失败及中断恢复。
- 真正构建和加载候选产物、正确/替代/错误补丁、数值边界和全部必需变体评分。
- 固定计划分母、缺失与基础设施错误、只读报告与独立证据复核。
- 模型输出非法工具请求时停止求解，已有补丁仍被评分；服务故障保持 incomplete。

此次还修复了工具参数中的 `{workspace}` 等字面量被错误替换，以及较长编译输出
过早触发会话关闭的问题。工具参数通过 stdin 单独传递；执行日志和模型反馈分别
限制为 16 MiB 和 32 Ki 字符。普通公开测试失败可返回模型继续处理。

最终测试数量、耗时和干净安装结果记录于[结构化结果](validation_controlled_20260914.json)。
干净安装在新建虚拟环境中完成，仓库外运行并清除 `PYTHONPATH`。数值四组控制实测为：
基线和语义错误变体失败，参考解和替代解成功；同时逐项执行离线 verify。
开发验收依据行为，不以 hash、文件数量或固定文档字节为标准。

## 既有完整框架与 GPU 证据

下列为 9 月 9 日前后已保存的真实工程控制，保留原任务、评分和运行身份。本轮
没有重跑完整 PyTorch/CUDA 构建，也没有把早期自主 Agent 轨迹算作新循环的证据。

| 任务 | 已有验证 |
| --- | --- |
| CPU LayerNorm，环境 r3 | 完整 PyTorch 基线构建 1925.059 秒；4 个控制符合预期且 verify 通过 |
| CUDA LogSoftmax，任务/评分 r2 | 完整 CUDA PyTorch 基线构建 2113.620 秒；5 个控制符合预期且 verify 通过 |
| CPU VectorNorm compile | 4 个控制符合预期且 verify 通过；实际生成并加载 Inductor 产物 |

见[远程验证](validation_remote_20260909.json)、
[CPU 控制](../../datasets/v0.8_candidates/layer_norm_cpu/validation/runtime_controls_r3.json)、
[CUDA 控制](../../datasets/v0.8_candidates/log_softmax_cuda/validation/runtime_controls_r2.json)及
[compile 控制](../../datasets/v0.8_candidates/vector_norm_compile_cpu/validation/runtime_controls_r1.json)。
13 个控制没有清理错误。compile 数学复算覆盖 16 个操作、115 个标量输出 / 230
个编码条目；属于 AI 辅助审查，批次构建见证不等于逐操作对抗性执行证明。

旧 CUDA 脚本化尝试保留两个 OOM 与一个成功 case；
[同冻结补丁复评](../../datasets/v0.8_candidates/log_softmax_cuda/validation/native_replay_r4.json)
三个 case / 30,464 个比较值全部成功。复评不重跑求解、不覆盖原失败、不增加独立
重复。共享 GPU 的后续观测不能证明原 OOM 根因，也不提供性能或独占运行证据。

## 清理和后续范围

本轮退役任意 Agent argv 执行、原生参考 Agent、模型网关/relay、多次重复择优和复杂
比较统计，并删除专用示例与测试。版本设计、实施与实验文档已恢复，见[版本历史](../history/README.md)。保留数据读取、执行环境、独立评分和报告中实际
需要的功能。历史 v0.7 实现在
[Git 历史](https://github.com/yangyi-nju/op_bench/tree/55c8ef2bd7d0bf78fc26bfd25498d9252e70781f)
可查，不增设第二套兼容执行流程。

v0.9 扩充分类数据，每类至少十道独立题，并维护选题与准入流程；v0.10 做固定工具
框架的多模型实验。Agent 框架对比、行为分析和统计扩展后续处理。代码许可仍待
学校确认。当前软件包仍为未正式发布的 `0.8.0.dev1`。

```bash
OPBENCH_TEST_DOCKER_IMAGE=opbench/cpp-fixture:cpu OPBENCH_TEST_DOCKER=1 \
  python -m unittest discover -s tests/core -v
python scripts/check_markdown_links.py
```

上面的测试命令对应 9 月 14 日环境；当前开发验证命令见[贡献说明](../../CONTRIBUTING.md)。
9 月 14 日原始证据位于 `runs/v0.8/validation/20260914-controlled/`，默认忽略；摘要随项目
保存。未提交工作区的版本号和 dirty 标志不能唯一表示源码，该轮保存控制器源码
副本用于追溯，不做文件 hash 级开发校验。

更早轮次保留原文：[初始轮次](validation_summary.json)、
[早期核心](validation_20260909.json)、[完整框架](validation_remote_20260909.json)、
[旧原生 Codex 与历史清理](validation_codex_20260909.json)。
