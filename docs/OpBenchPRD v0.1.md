# OpBench PRD

版本：v0.1  
状态：已评审 
日期：2026-05-22

## 1. 背景

OpBench 是一个面向深度学习框架算子问题的 agent benchmark。它参考 SWE-bench 的基本思想：从真实社区 issue/PR 中构建任务，让 agent 在指定仓库快照上修复问题，并通过自动化测试评测修复效果。

但 OpBench 不能简单复制 SWE-bench。SWE-bench 的很多任务主要依赖代码仓库、依赖安装和测试命令，复现成本相对可控。OpBench 面向 PyTorch、TensorFlow 等框架中的算子问题，问题可能与 **dtype、shape、数值精度、backend、kernel、构建产物、硬件、驱动和运行环境**强相关。对这类问题而言，环境不是附属说明，而是数据集样本的一部分。

因此，OpBench 的核心目标不是只收集 issue 和 patch，而是构建一套能够稳定保存、复现、修复和评测真实算子问题的系统。

## 2. 产品目标

第一版目标是完成一个可审核、可扩展的 MVP 系统，证明 OpBench 的方法论成立。

MVP 必须达成：

- 构建一个小型真实算子问题数据集。
- 每条数据绑定明确、可执行、可验证的运行环境。
- 支持 full repository 任务，即任务来自真实框架完整仓库，而不是只使用玩具 repo。
- 支持 source snapshot 或等价源码资产，避免评测时依赖不稳定的在线 clone。
- 支持 Docker 环境中的 baseline replay、gold replay 和 agent evaluation。
- 至少接入一个真实 agent。
- Host 侧负责 agent 控制、模型调用和认证，容器侧负责仓库操作和测试执行。
- 评测结果和复现证据以结构化 JSON 保存。
- 系统架构支持后续扩充更多数据、更多环境和更多 agent。

## 3. 非目标

第一版不追求：

- 大规模数据集。
- 覆盖 PyTorch、TensorFlow、JAX 等多个框架。
- 覆盖 CPU、CUDA、ROCm、TPU 等所有硬件 tier。
- 一次性解决所有 full source build 任务。
- 建设线上排行榜或 Web UI。
- 支持复杂多 agent 调度。
- 自动化挖掘并验证所有候选 PR。

第一版重点是固定标准流程，并让一个小规模真实闭环跑通。

## 4. 用户与场景

### 4.1 Benchmark 构建者

Benchmark 构建者从真实 issue/PR 中筛选算子问题，构建 task manifest，准备环境、源码快照、test patch、gold patch，并通过 replay 验证任务可纳入数据集。

### 4.2 Agent 评测者

Agent 评测者接入一个或多个 agent，在固定数据集上运行评测，获得每个任务的 resolved 状态、失败原因、命令日志和 patch diff。

### 4.3 系统维护者

系统维护者管理 Docker image、container 生命周期、source snapshot、运行缓存、replay evidence 和数据集版本。

## 5. 核心原则

### 5.1 环境是一等公民

每条任务必须声明并绑定可执行环境。环境信息不能只停留在 README 或依赖版本描述中，而应包含 Docker image、Dockerfile、build context、image digest、preflight 命令、硬件要求和运行限制。

### 5.2 复现优先于数量

数据集条目只有在 replay 通过后才能进入 verified 状态。一个任务必须证明：

- baseline 能稳定复现原始失败。
- gold patch 能稳定修复失败。
- pass-to-pass 测试没有回归。
- 测试确实运行在声明的环境中。
- agent 修改的源码确实是被测试加载和执行的源码。

### 5.3 Host 与 Container 职责分离

Host 侧负责：

- benchmark orchestration；
- agent 控制逻辑；
- 模型调用；
- 身份认证；
- 结果收集；
- 数据集与环境资产管理。

Container 侧负责：

- 仓库文件读取；
- 代码修改；
- 命令执行；
- 测试运行；
- diff 生成；
- 环境内 setup 状态维护。

### 5.4 Agent 必须通过标准 action interface 操作任务

Docker-backed task 中，真实 agent 不能直接在 host workspace 中运行命令或修改文件。agent 必须通过标准 action interface 操作容器内任务环境。否则评测会绕过环境约束，失去可复现性。

### 5.5 数据集采用分层策略

OpBench 第一版接受两层数据集：

- Layer A：MVP verified tasks，用于快速跑通完整评测闭环。
- Layer B：Full framework source-build tasks，用于体现 OpBench 区别于 SWE-bench 的高保真环境能力。

两层数据都必须来自真实 full repo 问题，但可以在环境构建成本和源码加载方式上分阶段推进。

## 6. 数据集分层

### 6.1 Layer A: MVP Verified Tasks

Layer A 用于第一版闭环验证。它应优先选择真实 full repo 中成本较低、可稳定复现的任务。

要求：

- 任务来自真实社区 issue/PR。
- base commit、merge commit、gold patch 可追溯。
- 使用完整仓库快照作为任务来源。
- 环境使用 Docker 固化。
- 测试命令能稳定运行。
- agent 修改的代码必须被测试实际使用。
- replay 通过后可标记为 verified。

允许：

- 选择 Python-level operator / nn / dtype / shape / lazy logic 问题。
- 使用预构建 wheel 作为底层运行环境，但必须保证被评测代码路径可被 agent 修改并被测试加载。
- 对 full repo 任务做最小可控的 setup，以避免每次从零构建大型框架。

不允许：

- 使用纯玩具 repo 替代真实 full repo。
- 用与真实 PR 无关的自造问题冒充真实任务。
- 让测试只覆盖镜像内安装包而不覆盖 agent 修改的 workspace 源码。

### 6.2 Layer B: Full Framework Source-Build Tasks

Layer B 用于后续扩展高保真任务。它面向需要完整源码构建、native extension、kernel、backend 或硬件环境的任务。

要求：

- 每条任务有可复用环境资产。
- 环境资产可以是预构建 Docker image，包含源码、依赖和构建产物。
- image 必须记录 digest 或等价不可变标识。
- replay 必须验证测试加载的是任务对应的源码或构建产物。
- 任务应记录构建成本、硬件要求和资源预算。

Layer B 可以比 Layer A 更慢、更重，但必须更接近真实框架开发和算子调试环境。

## 7. Full Repo 技术实现要求

第一版要求任务来自 full repo，因此系统必须解决以下问题。

### 7.1 Source Snapshot

系统需要支持将指定 base commit 的完整仓库快照准备为数据资产。对于 PyTorch 这类大仓库，允许从本地已 clone 仓库导出 snapshot，避免每次评测依赖网络 clone。

source snapshot 不需要进入 git，但必须记录：

- repo；
- base commit；
- snapshot path；
- snapshot hash 或生成证据；
- 生成脚本；
- 生成时间；
- 是否 clean；
- 是否包含 `.git` 元数据。

### 7.2 Workspace Preparation

评测开始时，系统应从 source snapshot 准备干净 workspace。每次 baseline、gold、agent attempt 都应使用独立 workspace。

对于 Docker task，workspace 必须放在 Docker 可稳定挂载的位置。macOS 上应避免使用 Docker 不可见的系统临时目录。

### 7.3 Source Loading

对 full repo 任务，系统必须明确测试加载的代码来源。

可选实现路线：

1. **Editable / source overlay 路线**
   - 使用预构建 wheel 提供 native 依赖。
   - 将 workspace 中的 Python 层源码覆盖到 import path。
   - 适合 Python-level 修复。

2. **Prebuilt source image 路线**
   - 每条任务或每组任务提前构建 Docker image。
   - image 内包含对应 base commit 的构建产物。
   - agent 修改源码后，必要时执行局部 rebuild 或可控 setup。

3. **Full source build 路线**
   - 评测时从 workspace 执行完整 build。
   - 真实性最高，但成本最高。
   - 不作为第一版默认路径。

第一版需要对 PyTorch full repo 任务做技术分析并选择可落地路线。无论选择哪条路线，都必须通过 replay evidence 证明 agent 修改的文件会影响测试结果。

### 7.4 Image Digest

环境不能只依赖 mutable tag。任务进入 verified 前必须记录 Docker image digest 或等价不可变标识。

Manifest 中至少应能记录：

- image tag；
- image digest；
- Dockerfile path；
- build context；
- build args；
- platform；
- preflight evidence。

## 8. 数据模型需求

### 8.1 Dataset Manifest

Dataset manifest 应包含：

- `dataset_id`
- `version`
- `status`
- `description`
- `environment_policy`
- `tasks`

每个 dataset task entry 应包含：

- `task_id`
- `task_path`
- `pr_url`
- `issue_url`
- `environment_status`
- `source_status`
- `replay_status`
- `admission_status`
- `notes`

### 8.2 Task Manifest

Task manifest 应包含：

- `task_id`
- `version`
- `source`
- `statement`
- `operator`
- `environment`
- `agent_visible`
- `evaluation`
- `artifacts`
- `metadata`

`source` 至少包含：

- `repo`
- `repo_url`
- `issue_url`
- `pr_url`
- `issue_number`
- `pr_number`
- `base_commit`
- `merge_commit`
- `checkout_mode`
- `snapshot_path`

`environment` 至少包含：

- `backend`
- `tier`
- `image`
- `image_digest`
- `dockerfile`
- `build_context`
- `workspace_dir`
- `preflight_workdir`
- `python_executable`
- `python_version`
- `os`
- `platform`
- `build_mode`
- `hardware`
- `resource_requirements`
- `dependencies`
- `preflight_commands`

`evaluation` 至少包含：

- `setup_commands`
- `fail_to_pass`
- `pass_to_pass`
- `test_command`
- `timeout_sec`

`artifacts` 至少包含：

- `gold_patch`
- `test_patch`

`metadata` 至少包含：

- `difficulty`
- `curation_status`
- `deterministic`
- `estimated_runtime_min`
- `notes`

## 9. 任务状态机

任务应有明确 admission pipeline。

状态建议：

- `candidate`：候选任务，尚未完整构建。
- `environment_ready`：环境 image 可用，preflight 通过。
- `source_ready`：source snapshot 可用。
- `baseline_reproduced`：baseline 能稳定复现失败。
- `gold_verified`：gold patch 能修复失败且无回归。
- `verified`：可进入正式评测集。
- `blocked`：因环境、源码、测试或任务定义问题阻塞。
- `deprecated`：任务不再使用。

Dataset manifest 中的 `environment_status`、`source_status`、`replay_status`、`admission_status` 应能表达这些状态或其简化映射。

## 10. Replay 需求

系统必须提供标准 replay 流程。

### 10.1 Baseline Replay

Baseline replay 步骤：

1. 准备 workspace。
2. 准备 Docker 环境。
3. 运行 preflight。
4. 执行 setup commands。
5. 应用 test patch。
6. 不应用 gold patch。
7. 运行 fail-to-pass 测试。
8. 运行 pass-to-pass 测试。
9. 输出 replay evidence。

通过条件：

- fail-to-pass 至少按任务定义失败。
- pass-to-pass 全部通过。
- 无 environment error。
- 无 runner error。

### 10.2 Gold Replay

Gold replay 步骤：

1. 准备 workspace。
2. 准备 Docker 环境。
3. 运行 preflight。
4. 执行 setup commands。
5. 应用 test patch。
6. 应用 gold patch。
7. 运行 fail-to-pass 测试。
8. 运行 pass-to-pass 测试。
9. 输出 replay evidence。

通过条件：

- fail-to-pass 全部通过。
- pass-to-pass 全部通过。
- 无 environment error。
- 无 runner error。

## 11. Agent 评测需求

第一版必须至少接入一个真实 agent。允许只接入一个真实 agent，但系统设计应支持后续扩展多个 agent。

### 11.1 Agent 类型

第一版至少支持：

- `noop`：不修改代码，用于负例 sanity check。
- `gold`：应用 gold patch，用于上界 sanity check。
- 一个真实 agent：例如 Codex CLI 或等价 agent。

### 11.2 Docker Task 的 Agent 边界

对于 Docker-backed task：

- agent 控制逻辑运行在 host。
- agent 的文件读取、文件修改、命令执行和测试运行必须通过 action interface。
- action interface 的命令必须在任务容器内执行。
- agent 最终提交 `git diff` 作为 patch。
- 如果某个真实 agent 尚不能通过 action interface 操作容器，系统必须返回 `agent_runtime_unsupported`，不能退化为 host 执行。

### 11.3 Agent 输入

Agent 可见输入包括：

- issue statement；
- task metadata 中允许公开的字段；
- workspace 文件；
- allowed commands；
- 环境说明；
- 测试反馈。

Agent 不应直接看到：

- gold patch；
- hidden evaluator-only metadata；
- replay 结论；
- 数据集 admission 标注。

## 12. Action Interface 需求

第一版 action interface 至少支持：

- `read_file`
- `write_file`
- `apply_patch`
- `run_command`
- `run_test`
- `git_diff`

所有 action 都必须：

- 限制在 workspace 内；
- 记录输入；
- 记录输出；
- 记录 exit code；
- 记录 stdout / stderr；
- 支持 timeout；
- 能关联到当前 task attempt。

后续可扩展：

- `list_files`
- `search`
- `read_file_range`
- `structured_test_result`
- `resource_usage`
- `environment_probe`

## 13. 环境管理需求

系统需要支持：

- Docker 可用性检查。
- image inspect。
- image build。
- image digest 记录。
- task-scoped container 创建。
- workspace mount。
- preflight 执行。
- setup state 保持。
- container cleanup。
- timeout。
- 环境不可用时返回 `environment_unavailable`。

环境不可用属于调度或数据集环境问题，不应计为 agent 修复失败。

## 14. 结果与证据

每次 evaluation 输出结构化 JSON。

单次结果至少包含：

- `task_id`
- `mode`
- `agent`
- `status`
- `fail_to_pass_total`
- `fail_to_pass_passed`
- `pass_to_pass_total`
- `pass_to_pass_passed`
- `duration_sec`
- `environment`
- `commands`
- `patch`
- `error`

命令日志至少包含：

- command；
- cwd；
- exit_code；
- stdout；
- stderr；
- duration_sec；
- timed_out。

Dataset 级结果至少包含：

- 每个 agent 的 resolved rate；
- fail-to-pass pass rate；
- pass-to-pass pass rate；
- timeout rate；
- environment unavailable rate；
- agent runtime unsupported rate；
- 平均或中位 runtime。

## 15. 缓存与资产管理

允许大型资产不进入 git。

不进入 git 的资产包括：

- source snapshots；
- Docker build cache；
- replay runs；
- agent run outputs；
- large logs；
- temporary workspaces。

但 manifest 或 evidence 必须记录足够信息，使得资产可以被重建或校验。

推荐目录：

- `.op_bench_cache/sources/`
- `.op_bench_cache/workspaces/`
- `.op_bench_cache/images/`
- `runs/env/`
- `runs/sources/`
- `runs/replay/`
- `runs/experiments/`

## 16. MVP 验收标准

MVP 完成需要满足：

- 至少 2 条真实 full repo PyTorch task 进入数据集 draft。
- 至少 1 条 task 进入 verified。
- verified task 必须有 replay evidence。
- 环境使用 Docker，并记录 image digest。
- source snapshot 准备流程可复用。
- baseline replay 和 gold replay 命令可重复执行。
- 至少接入一个真实 agent。
- agent 对 Docker task 的操作遵守 action interface。
- `noop`、`gold`、真实 agent 均能通过同一 runner 执行。
- 数据集和任务 manifest 校验通过。
- 单元测试通过。
- README 或 manual validation 文档能指导人工跑通。

## 17. 当前项目状态

截至 v0.1 PRD 编写时，项目已经具备：

- task manifest 基础结构；
- dataset manifest 初版；
- Docker environment manager；
- Docker executor；
- source snapshot 准备脚本；
- replay 脚本；
- `noop`、`gold`、Codex agent 基础适配；
- PyTorch mini dataset draft；
- 两条 PyTorch PR draft task；
- 本地 source snapshot 准备能力；
- 部分验证脚本和单元测试。

当前未完成：

- 至少一条 PyTorch full repo task 的 verified replay。
- Docker-backed real agent 的完整 action-interface bridge。
- image digest 固化。
- full repo source loading / source build 技术路线定稿。
- 数据集 admission 状态自动更新。
- verified task 的 evidence 标准化归档。

## 18. 关键技术风险

### 18.1 Full Repo 源码加载风险

PyTorch 这类仓库不能简单挂载源码后直接运行测试。测试可能导入镜像内已安装的 torch，而不是 workspace 中 agent 修改后的源码。如果强制从 source tree import，又可能缺少构建生成文件、动态库或 native extension。

需要在技术设计中明确：

- 哪些任务允许使用 source overlay。
- 哪些任务必须使用 prebuilt source image。
- 哪些任务必须 full source build。
- 如何用 replay evidence 证明测试覆盖了 agent 修改。

### 18.2 环境构建成本风险

Full source build 可能耗时很长，且对机器资源要求高。第一版应避免把完整 source build 放在每次 agent attempt 的热路径上。

### 18.3 Agent 隔离风险

如果真实 agent 绕过 action interface 在 host 上执行命令，评测结果无效。系统必须在 Docker-backed task 上 fail closed。

### 18.4 数据集质量风险

真实 PR 中的测试可能不稳定，或者原始 issue 不能在当前环境复现。任务必须经过 replay admission，不能只凭 PR 合并就纳入 verified。

## 19. 开发流程

后续开发按以下流程推进：

1. PRD 审核确认。
2. 编写 Technical Design v0.1。
3. 明确 full repo 技术路线。
4. 切分 milestone。
5. 每个 milestone 独立实现、测试、提交。
6. 每条 verified task 都保存 replay evidence。
7. MVP 跑通后再扩展数据集和 agent。

建议 milestone：

- M1：PRD 与技术设计定稿。
- M2：full repo source loading 路线验证。
- M3：一条 PyTorch task verified。
- M4：Docker action interface bridge 接入一个真实 agent。
- M5：mini dataset 实验跑通。
- M6：整理文档、证据和初始 commit。
