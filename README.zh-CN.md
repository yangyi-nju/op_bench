# OpBench

**OpBench 是面向算子修复任务的模型评测系统。** 给模型一份有缺陷的代码和问题描述，让它使用统一工具完成修复，再由独立评分器判断是否解决问题。任务关注算子数值、行为，以及直接影响算子的编译和运行链路。

第一阶段要回答的是：**在相同题目、工具和预算下，哪个模型更擅长修复算子问题？** 当前 v0.8 提供这条评测主流程；数据集扩充与准入流程安排在 v0.9，多模型实验安排在 v0.10。

[English](README.md) · [架构导读](docs/architecture.md) · [使用文档](docs/README.zh-CN.md) · [历史版本记录](docs/history/README.md)

## 一道题如何完成评测

以 softmax 数值溢出为例，题目提供待修复源码、输入输出约定、环境和公开测试。模型可以查看和修改源码、构建、运行公开测试；私有测试和参考答案不提供给模型。

1. **准备题目**：读取任务和实验配置，准备源码与运行环境。
2. **模型修复**：OpBench 反复调用模型，执行它选择的固定 MCP 工具，反馈公开结果。
3. **独立评分**：停止求解，保存代码补丁，在独立工作区构建并运行评分测试。
4. **汇总结果**：保存调用轨迹、补丁、测试结果，计算各模型的 `resolved_rate`。

`resolved_rate = 已解决题数 / 计划题数`。公开测试通过不等于最终解决；构建和规定的评分测试需要全部满足要求。缺失结果与运行故障会单独显示，未完成的实验报告只表示当前进度。

## 运行评测

需要 Python 3.12+、Git、Docker 和 Linux/macOS。在仓库根目录安装并检查数据清单：

```bash
python -m pip install .
opbench doctor
opbench datasets validate --dataset datasets/v0.8_candidates/dataset.json
```

当前[候选数据集](datasets/v0.8_candidates/README.md)包含来自上游的算子任务，尚未完成正式准入。上述校验检查清单结构；实际执行还需准备题目声明的源码、依赖镜像及 CPU/GPU 资源，步骤见[运行环境指南](docs/v0.8/remote_execution.md)。

准备完成后，编写实验配置，选择数据集、模型和共同预算。[模型接入指南](docs/v0.8/agent_integration.md#实验配置与运行)提供配置及运行、恢复、报告命令。兼容 Chat Completions 的服务可为每个模型设置 Base URL 和密钥环境变量，也可使用本地 Codex CLI 做模型接入检查。受控评测使用断网 Docker 工作区；控制器调用模型，并执行模型选择的 OpBench 工具。

只想快速检查安装或评测引擎，可以按[贡献说明](CONTRIBUTING.md)运行小型测试夹具，无需完整框架构建。这些夹具属于内部工程测试，不计入评测数据集。

## 项目放了什么

| 目录 | 内容 |
| --- | --- |
| [`src/op_bench/`](src/op_bench/) | 按数据、运行、环境、评分、结果拆分的实现；从[架构导读](docs/architecture.md)开始阅读 |
| [`tasks/`](tasks/README.md) | 旧 schema 的历史任务素材，待迁移和准入审查 |
| [`datasets/`](datasets/) | 数据集清单、候选任务和数据说明 |
| [`tests/`](tests/) | `core/` 保存行为测试，[fixtures/](tests/fixtures/README.md) 保存集成测试使用的小型任务包 |
| [`runs/`](runs/README.md) | 按版本和批次保存的本地运行证据，区分工程验证、正式实验、缓存和历史归档 |
| [`docs/`](docs/README.zh-CN.md) | 当前使用与设计说明，以及[历次迭代记录](docs/history/README.md) |

新增题目主要编写任务包；更换模型主要修改实验配置；修改评分规则进入 `evaluation` 模块。各模块的职责、依赖与阅读顺序见[架构导读](docs/architecture.md)。

## 当前进度

v0.8 的工程实现和本地验收已经完成，软件包仍为未正式发布的开发版本 `0.8.0.dev1`。当前候选任务与保留的历史数据仍待新版准入审查；v0.9 将完善分类与准入，每类至少 10 道独立题。v0.10 开展固定工具下的多模型实验，开始前按选定条件对目标服务逐个做真实 API 小样本联调。外部 Agent 框架对比和行为分析放在后续阶段。

具体变更见 [CHANGELOG](CHANGELOG.md)，实现状态和实测证据见[当前状态](docs/v0.8/development_status.md)与[验证记录](docs/v0.8/validation_report.md)。旧版本文档保留原文，用于追溯当时的设计与实验，不作为当前使用说明。

开发环境与测试方法见[贡献说明](CONTRIBUTING.md)。代码许可仍待作者与学校确认，上游代码和补丁保留各自归属与许可要求。
