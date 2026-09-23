# 版本历史

这里保存 OpBench v0.1–v0.7 的设计、实施与实验记录，便于回顾毕业设计的演进过程。当前系统的说明从[文档首页](../README.zh-CN.md)进入。

这些历史文档从 v0.7 阶段的提交 `82c8e064bf30fead2d941d11cf94053ce3fc7e14` 恢复，保留原文；设计文档路径不变，四份原位于 `runs/` 的实验摘要现按版本归入 `runs/archive/`。文中“当前”“已完成”、版本路线、评测结果和验收结论，都指当时的状态；旧版命令、模块名和数据准入规则不代表当前 v0.8 的接口与约定。

## 按版本阅读

| 版本 | 设计与目标 | 实施、数据与使用记录 | 实验与验证记录 |
| --- | --- | --- | --- |
| v0.1 | [产品需求](../v0.1/product_requirements.md) | [开发者指南](../v0.1/developer_guide.md)、[任务构建流程](../v0.1/builder_workflow.md) | [实验报告](../v0.1/experiment_report.md)、[人工验证流程](../v0.1/manual_validation.md) |
| v0.2 | [设计](../v0.2/design.md) | [实施计划](../v0.2/implementation_plan.md)、[开发者指南](../v0.2/developer_guide.md)、[首批 PyTorch 数据集](../../datasets/pytorch_mini/README.zh-CN.md)（[English](../../datasets/pytorch_mini/README.md)） | [实验报告](../v0.2/experiment_report.md) |
| v0.3 | [设计](../v0.3/design.md) | [候选任务](../v0.3/candidate_tasks.md) | [实验报告](../v0.3/experiment_report.md) |
| v0.4 | [设计](../v0.4/design.md) | [CUDA 候选任务](../v0.4/candidate_tasks_cuda.md) | [实验报告](../v0.4/experiment_report.md) |
| v0.5 | [设计](../v0.5/design.md) | [候选检索](../v0.5/candidate_search.md)、[准入提示模板](../v0.5/admission_prompt.md)、[远程执行配置](../v0.5/setup_remote_agent.md) | [实验报告](../v0.5/experiment_report.md) |
| v0.6 | [设计](../v0.6/design.md) | [实施计划](../v0.6/implementation_plan.md)、[开发者指南](../v0.6/developer_guide.md)、[发布说明](../v0.6/release_notes.md) | [实验报告](../v0.6/experiment_report.md)、[验收矩阵](../v0.6/acceptance_matrix.md)、[M6 验证](../v0.6/m6_verification.md)、[M7 验证](../v0.6/m7_verification.md)、[MCP 实验](../v0.6/mcp_agent_experiment.md)、[MCP 实验验证](../v0.6/mcp_agent_experiment_verification.md) |
| v0.7 | [设计](../v0.7/design.md)、[质量扩展约定](../v0.7/quality_expansion.md) | [候选检索](../v0.7/candidate_search.md)、[运行时匹配](../v0.7/setup_matched_runtime.md)、[边界任务](../v0.7/boundary_tasks.md)、[数据集说明](../v0.7/dataset_card.md) | [验证报告](../v0.7/validation_report.md) |

## 跨版本记录

- [截至 v0.7 的完整 Changelog](CHANGELOG-through-v0.7.md)：原根目录 `CHANGELOG.md` 的完整快照。
- [当时的全局项目方案](../project_plan.md)：保留 2026-08-11 时的方向与版本规划。
- [当时的项目状态](../project_state.md)：保留 v0.7 完成时的状态、测试与实验结论。

当前路线以当前版本的设计文档为准。历史中曾计划的 v0.8 或后续阶段名称，不应与后来实际采用的版本范围混用。

## v0.7 分阶段设计与实施

`docs/superpowers/` 保存当时的细分工作记录，适合追溯某项决策如何落实；初次阅读可以先看上面的版本设计与验证报告。

| 阶段 | 设计记录 | 实施计划 |
| --- | --- | --- |
| P1：数据工厂约定 | [设计](../superpowers/specs/2026-07-26-v0.7-p1-factory-contract-design.md) | [计划](../superpowers/plans/2026-07-26-v0.7-p1-factory-contract.md) |
| P2：匹配运行时 | [设计](../superpowers/specs/2026-07-26-v0.7-p2-matched-runtime-design.md) | [计划](../superpowers/plans/2026-07-26-v0.7-p2-matched-runtime.md) |
| P3：边界任务 | [设计](../superpowers/specs/2026-07-27-v0.7-p3-boundary-tasks-design.md) | [计划](../superpowers/plans/2026-07-27-v0.7-p3-boundary-tasks.md) |
| P4：数据集验证 | [设计](../superpowers/specs/2026-07-27-v0.7-p4-dataset-validation-design.md) | [计划](../superpowers/plans/2026-07-27-v0.7-p4-dataset-validation.md) |
| P5：发布 | 见 [v0.7 设计](../v0.7/design.md) | [计划](../superpowers/plans/2026-07-27-v0.7-p5-release.md) |
| 50 条任务质量扩展 | [设计](../superpowers/specs/2026-07-29-v0.7-quality-taxonomy-expansion-design.md) | [计划](../superpowers/plans/2026-07-29-v0.7-quality-first-50-task-release.md) |

## 原始实验摘要

以下 Markdown 报告随历史文档一起恢复，整理时原文未改。现存本地原始记录按版本归档，位置见[实验归档索引](../../runs/archive/README.md)；先前已经退役的资产仍从 Git 历史查阅，不因本次整理恢复。

- [v0.5 精度 PR 筛选摘要](../../runs/archive/v0.5/pr_screening/summary.md)
- [v0.6 MCP 完整实验报告](../../runs/archive/v0.6/mcp_full_20260722_event_redaction_r5_report/experiment_report.md)
- [v0.7 边界任务实验报告](../../runs/archive/v0.7/validation_report/experiment_report.md)
- [v0.7 50 条任务质量实验报告](../../runs/archive/v0.7/quality_validation_report/experiment_report.md)

## 查看历史资产

恢复文档不等于恢复旧实现。历史文档引用的旧代码、数据集、环境 Dockerfile 和运行产物，部分已不在当前工作树中；其中的本地路径与相对链接按原提交上下文保留。需要核对资产或重现旧命令时，应查看上述提交中的对应文件，或使用独立的历史检出目录。

例如，可以只读取历史文件而不改变当前分支：

```bash
git show 82c8e064bf30fead2d941d11cf94053ce3fc7e14:datasets/pytorch_mini/dataset.json
```

历史文档中包含的本机绝对路径，是当时的执行位置记录，通常无法在其他机器直接打开。历史报告的实验配置和评价口径也不能直接替代当前版本的结果。
