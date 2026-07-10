# OpBench Docs

Language: English | [中文](README.zh-CN.md)

Documents are archived by version. The docs root keeps only indexes; new versions should use a `docs/vX.Y/` directory with lower snake_case filenames such as `design.md`, `developer_guide.md`, `experiment_report.md`, and `implementation_plan.md`.

Use these documents in this order:

1. [v0.5 precision experiment report](v0.5/experiment_report.md): 17-task cumulative draft manifest and 6-task, 18-attempt precision-phase results with eight-dimensional metrics.
2. [v0.5 design](v0.5/design.md): problem-dimension taxonomy, precision subclasses, candidate policy, and extended metrics.
3. [v0.5 candidate search](v0.5/candidate_search.md): ghstack-aware PyTorch PR discovery and precision screening rules.
4. [v0.5 remote agent setup](v0.5/setup_remote_agent.md): remote images, host configuration, and admission execution.
5. [v0.5 admission prompt](v0.5/admission_prompt.md): batch admission handoff for precision tasks.
6. [v0.4 design](v0.4/design.md): CUDA runtime tiers, remote GPU Docker executor over SSH, `inplace_build` source loading, and planned public test ablation.
7. [v0.4 experiment report](v0.4/experiment_report.md): 13-task 3-repeat Codex CLI evaluation, 84.6% resolved rate, tier breakdown.
8. [v0.4 CUDA task candidates](v0.4/candidate_tasks_cuda.md): screening criteria and PR pool for CUDA tasks.
9. [v0.4 public test ablation notes](v0.4/public_test_ablation.md): design notes; experiment deferred to v0.5.
10. [v0.3 design](v0.3/design.md): 10-task PyTorch expansion, public/hidden test split, multi-file overlay, and CUDA pilot plan.
11. [v0.3 experiment report](v0.3/experiment_report.md): 10-task evaluation results, 76.7% resolved rate, stability analysis.
12. [v0.2 design](v0.2/design.md): approved requirements and architecture for the environment-management and dataset-admission release.
13. [v0.2 developer guide](v0.2/developer_guide.md): registry, admission, curation, asset inspection, and container management workflow.
14. [v0.2 experiment report](v0.2/experiment_report.md): 3-task verified dataset, admission, gold-loop, and real Codex action-bridge evaluation results.
15. [v0.2 implementation plan](v0.2/implementation_plan.md): active development milestones and verification commands.
16. [v0.1 developer guide](v0.1/developer_guide.md): v0.1 architecture, module responsibilities, experiment flow, and extension rules.
17. [v0.1 manual validation workflow](v0.1/manual_validation.md): v0.1 commands for promoting a task from `draft` to `verified`.
18. [v0.1 experiment report](v0.1/experiment_report.md): evidence from the first real Codex action-bridge experiment.
19. [v0.1 dataset builder workflow](v0.1/builder_workflow.md): how to bootstrap draft tasks from GitHub PRs.
20. [v0.1 PRD](v0.1/product_requirements.md): product requirements record for v0.1 planning.

See the repository-level `CHANGELOG.md` for the version history.
