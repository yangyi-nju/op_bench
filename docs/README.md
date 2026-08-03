# OpBench Docs

Language: English | [中文](README.zh-CN.md)

Documents are archived by version. The docs root keeps only indexes; new versions should use a `docs/vX.Y/` directory with lower snake_case filenames such as `design.md`, `developer_guide.md`, `experiment_report.md`, and `implementation_plan.md`.

The current stable historical freeze is `opbench-v0.7.0`: [25-task cumulative](../datasets/pytorch_v0.7/dataset.json), [6-task Boundary](../datasets/pytorch_v0.7_boundary/dataset.json), and [8-task Precision](../datasets/pytorch_v0.7_precision/dataset.json). Its validation evidence is descriptive and **non-leaderboard**. The active [v0.7 50-task quality expansion](v0.7/quality_expansion.md) remains in development. The historical [v0.6 experiment report](v0.6/experiment_report.md) remains available.

Use these documents in this order:

1. [Global project plan](project_plan.md): mission, research questions, principles, roadmap, release gates, and v1.0 target.
2. [Current project state](project_state.md): factual baseline, active release, decisions, open items, and next actions.
3. [v0.7 Dataset Factory and Boundary design](v0.7/design.md): boundary taxonomy, admission factory, matched-runtime recovery, and dataset contract.
4. [v0.7 50-task quality expansion](v0.7/quality_expansion.md): current quality gates, multi-axis taxonomy, Admission, and 122-attempt contract.
5. [v0.7 candidate search](v0.7/candidate_search.md): source window, automated candidate funnel, and human-review dispositions.
6. [v0.7 matched-runtime setup](v0.7/setup_matched_runtime.md): reproducible source and wheel Runtime preparation.
7. [v0.7 Boundary task report](v0.7/boundary_tasks.md): admitted Task evidence and B1–B5 coverage.
8. [v0.7 Dataset Card](v0.7/dataset_card.md): release identity, Dataset hashes, provenance, taxonomy, contamination risks, and limitations.
9. [v0.7 validation report](v0.7/validation_report.md): the 18-Attempt real Codex validation cohort and integrity evidence.
9. [v0.6 platform design](v0.6/design.md): the unified Demo-to-Platform architecture and completion definition.
10. [v0.6 implementation plan](v0.6/implementation_plan.md): M1–M7 dependencies, implementation scope, and verification strategy.
11. [v0.6 developer guide](v0.6/developer_guide.md): protocol selection, Runtime support, identity, artifacts, resume, failure attribution, and known limits.
12. [v0.6 acceptance matrix](v0.6/acceptance_matrix.md): measurable release requirements and evidence states.
13. [v0.6 release notes](v0.6/release_notes.md): completed platform scope, migration contract, closed gates, frozen evidence, and explicit non-claims.
14. [v0.6 experiment report](v0.6/experiment_report.md): the human-readable 51-Attempt real MCP results, analysis, evidence, and limits.
15. [v0.6 real MCP Agent experiment guide](v0.6/mcp_agent_experiment.md): frozen four-cohort procedure, resume rules, and publication boundary.
16. [v0.6 real MCP experiment verification](v0.6/mcp_agent_experiment_verification.md): Integrity, trace, cleanup, deterministic-report, and privacy gates.
17. [v0.6 M6 verification](v0.6/m6_verification.md): Runtime conformance, replay inventory, real Codex canaries, the original exact-Remote block, and its 85/85 closure addendum.
18. [v0.6 M7 verification](v0.6/m7_verification.md): executable Demo, documentation, clean-environment verification, and final release decision.
19. [v0.5 experiment report](v0.5/experiment_report.md): verified 17-task dataset, 51-attempt full results, precision breakdown, and eight-dimensional metrics.
20. [v0.5 design](v0.5/design.md): problem-dimension taxonomy, precision subclasses, candidate policy, and extended metrics.
21. [v0.5 candidate search](v0.5/candidate_search.md): ghstack-aware PyTorch PR discovery and precision screening rules.
22. [v0.5 remote agent setup](v0.5/setup_remote_agent.md): remote images, host configuration, and admission execution.
23. [v0.5 admission prompt](v0.5/admission_prompt.md): batch admission instructions for precision tasks.
24. [v0.4 design](v0.4/design.md): CUDA runtime tiers, remote GPU Docker executor over SSH, `inplace_build` source loading, and planned public test ablation.
25. [v0.4 experiment report](v0.4/experiment_report.md): 13-task 3-repeat Codex CLI evaluation, 84.6% resolved rate, tier breakdown.
26. [v0.4 CUDA task candidates](v0.4/candidate_tasks_cuda.md): screening criteria and PR pool for CUDA tasks.
27. [v0.3 design](v0.3/design.md): 10-task PyTorch expansion, public/hidden test split, multi-file overlay, and CUDA pilot plan.
28. [v0.3 experiment report](v0.3/experiment_report.md): 10-task evaluation results, 76.7% resolved rate, stability analysis.
29. [v0.2 design](v0.2/design.md): approved requirements and architecture for environment management and dataset admission.
30. [v0.2 developer guide](v0.2/developer_guide.md): registry, admission, curation, asset inspection, and container management workflow.
31. [v0.2 experiment report](v0.2/experiment_report.md): 3-task verified dataset, admission, gold-loop, and real Codex action-bridge evaluation results.
32. [v0.2 implementation plan](v0.2/implementation_plan.md): development milestones and verification commands.
33. [v0.1 developer guide](v0.1/developer_guide.md): v0.1 architecture, module responsibilities, experiment flow, and extension rules.
34. [v0.1 manual validation workflow](v0.1/manual_validation.md): v0.1 commands for promoting a task from `draft` to `verified`.
35. [v0.1 experiment report](v0.1/experiment_report.md): evidence from the first real Codex action-bridge experiment.
36. [v0.1 dataset builder workflow](v0.1/builder_workflow.md): how to bootstrap draft tasks from GitHub PRs.
37. [v0.1 PRD](v0.1/product_requirements.md): product requirements record for v0.1 planning.

See the repository-level `CHANGELOG.md` for the version history.
