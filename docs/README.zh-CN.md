# 文档导航

第一次接触项目，先读[项目说明](../README.zh-CN.md)，再按需要选择下面的入口。

## 理解和使用当前项目

| 你想做什么 | 从这里开始 |
| --- | --- |
| 理解代码结构、调用流程和模块职责 | [架构导读](architecture.md) |
| 运行第一个模型，查看结果或恢复实验 | [实验配置与模型接入](v0.8/agent_integration.md#实验配置与运行) |
| 编写一道题，检查正确与错误补丁 | [任务格式](v0.8/task_format.md)、[候选任务包](../datasets/v0.8_candidates/README.md) |
| 理解模型能调用哪些工具 | [模型与工具接入](v0.8/agent_integration.md) |
| 理解源码、公开测试、私有评分之间的边界 | [信息边界](v0.8/information_boundary.md) |
| 理解 resolved_rate 与未完成实验 | [计分规则](v0.8/statistical_protocol.md) |
| 在 CPU/GPU 主机部署环境 | [运行环境指南](v0.8/remote_execution.md) |
| 修改代码和运行测试 | [贡献说明](../CONTRIBUTING.md) |
| 保存实验、区分批次、查找旧结果 | [运行记录](../runs/README.md) |

## 查看当前版本的设计与进度

v0.8 工程实现与本地验收已完成，尚未正式发布；v0.9 更新数据集与准入；v0.10 在目标模型服务联调通过后开展多模型实验。

- [v0.8 设计](v0.8/design.md)：本轮范围和设计取舍。
- [实现状态](v0.8/development_status.md)、[实施计划](v0.8/implementation_plan.md)、[验收清单](v0.8/acceptance_matrix.md)：已经完成和仍需处理的事项。
- [验证记录](v0.8/validation_report.md)：实际测试、运行结果和已知限制。
- [数据候选](../datasets/v0.8_candidates/README.md)、[历史任务初审](v0.8/task_disposition.md)：后续数据集整理的输入。

## 回顾历次迭代

[历史版本记录](history/README.md)保留早期版本文档与实验报告的原文。查阅时按版本理解其口径；当前安装、命令和模块结构以上面的入口为准。各版本变化摘要见 [CHANGELOG](../CHANGELOG.md)。
