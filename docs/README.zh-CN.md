# OpBench 文档

语言：[English](README.md) | 中文

文档按版本归档。根目录只保留索引，新增版本时使用 `docs/vX.Y/` 目录，文件名保持小写 snake_case，例如 `design.md`、`developer_guide.md`、`experiment_report.md`、`implementation_plan.md`。

建议按以下顺序阅读：

1. [v0.3 设计方案](v0.3/design.md)：10-task PyTorch 数据扩展、public/hidden test 分层、multi-file overlay 和 CUDA 试点方案。
2. [v0.2 设计文档](v0.2/design.md)：环境管理与数据准入版本已确认的需求和架构。
3. [v0.2 开发者指南](v0.2/developer_guide.md)：registry、admission、curation、资产检查和容器管理流程。
4. [v0.2 实验报告](v0.2/experiment_report.md)：3-task verified 数据集、admission、gold 闭环和真实 Codex action bridge 评测结果。
5. [v0.2 实施计划](v0.2/implementation_plan.md)：当前开发里程碑和验证命令。
6. [v0.1 开发者指南](v0.1/developer_guide.md)：v0.1 架构、模块职责、实验流程和扩展规则。
7. [v0.1 手动验证流程](v0.1/manual_validation.md)：v0.1 将 task 从 `draft` 晋升为 `verified` 的操作命令。
8. [v0.1 实验报告](v0.1/experiment_report.md)：第一次真实 Codex action bridge 实验的证据和分析。
9. [v0.1 数据构建流程](v0.1/builder_workflow.md)：如何从 GitHub PR 初始化 draft task。
10. [v0.1 PRD](v0.1/product_requirements.md)：v0.1 规划阶段的产品需求记录。

版本迭代记录见仓库根目录的 `CHANGELOG.md`。
