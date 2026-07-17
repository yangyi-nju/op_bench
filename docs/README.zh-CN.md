# OpBench 文档

语言：[English](README.md) | 中文

文档按版本归档。根目录只保留索引，新增版本时使用 `docs/vX.Y/` 目录，文件名保持小写 snake_case，例如 `design.md`、`developer_guide.md`、`experiment_report.md`、`implementation_plan.md`。

建议按以下顺序阅读：

1. [全局项目方案](project_plan.md)：使命、研究问题、原则、路线图、发布门和 v1.0 目标。
2. [当前项目状态](project_state.md)：事实基线、当前版本、已确认决策、开放项和下一动作。
3. [v0.6 平台设计](v0.6/design.md)：统一的 Demo→Platform 架构和完成定义。
4. [v0.6 实施计划](v0.6/implementation_plan.md)：M1–M7 依赖、实现范围和验证策略。
5. [v0.6 开发者指南](v0.6/developer_guide.md)：协议选择、Runtime 支持、身份、Artifact、Resume、失败归因和已知限制。
6. [v0.6 验收矩阵](v0.6/acceptance_matrix.md)：可度量的发布要求和证据状态。
7. [v0.6 M6 验证记录](v0.6/m6_verification.md)：Runtime Conformance、Replay 清单、真实 Codex canary 和精确 Remote 阻塞证据。
8. [v0.6 发布说明](v0.6/release_notes.md)：平台交付范围、迁移合同、证据、阻塞门和明确不作出的结论。
9. [v0.6 M7 验证记录](v0.6/m7_verification.md)：可执行 Demo、文档、干净环境验证和最终发布判定。
10. [v0.7 Dataset Factory 与 Boundary 设计](v0.7/design.md)：Boundary 分类、Admission Factory、matched-runtime 恢复和数据集合同。
11. [v0.5 实验报告](v0.5/experiment_report.md)：verified 17-task 数据集、51-attempt 全量结果、precision 拆解和 8 维指标。
12. [v0.5 设计方案](v0.5/design.md)：问题维度分类、precision 子类、候选策略和扩展指标。
13. [v0.5 候选检索](v0.5/candidate_search.md)：兼容 ghstack 的 PyTorch PR 检索和 precision 筛选规则。
14. [v0.5 远程 agent 配置](v0.5/setup_remote_agent.md)：远程镜像、主机配置和 admission 执行方法。
15. [v0.5 admission prompt](v0.5/admission_prompt.md)：precision task 批量 admission 说明。
16. [v0.4 设计方案](v0.4/design.md)：CUDA runtime tier、远程 GPU Docker SSH 执行器、`inplace_build` 源码加载和 public test ablation 设计。
17. [v0.4 实验报告](v0.4/experiment_report.md)：13-task 3-repeat Codex CLI 评测，84.6% resolved rate，按 tier 拆解。
18. [v0.4 CUDA 候选 task](v0.4/candidate_tasks_cuda.md)：CUDA task 筛选标准和 PR 候选池。
19. [v0.3 设计方案](v0.3/design.md)：10-task PyTorch 数据扩展、public/hidden test 分层、multi-file overlay 和 CUDA 试点方案。
20. [v0.3 实验报告](v0.3/experiment_report.md)：10-task 评测结果，76.7% resolved rate，稳定性分析。
21. [v0.2 设计文档](v0.2/design.md)：环境管理与数据准入版本的需求和架构。
22. [v0.2 开发者指南](v0.2/developer_guide.md)：registry、admission、curation、资产检查和容器管理流程。
23. [v0.2 实验报告](v0.2/experiment_report.md)：3-task verified 数据集、admission、gold 闭环和真实 Codex action bridge 评测结果。
24. [v0.2 实施计划](v0.2/implementation_plan.md)：开发里程碑和验证命令。
25. [v0.1 开发者指南](v0.1/developer_guide.md)：v0.1 架构、模块职责、实验流程和扩展规则。
26. [v0.1 手动验证流程](v0.1/manual_validation.md)：v0.1 将 task 从 `draft` 晋升为 `verified` 的操作命令。
27. [v0.1 实验报告](v0.1/experiment_report.md)：第一次真实 Codex action bridge 实验的证据和分析。
28. [v0.1 数据构建流程](v0.1/builder_workflow.md)：如何从 GitHub PR 初始化 draft task。
29. [v0.1 PRD](v0.1/product_requirements.md)：v0.1 规划阶段的产品需求记录。

版本迭代记录见仓库根目录的 `CHANGELOG.md`。
