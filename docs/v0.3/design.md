# OpBench v0.3 设计方案

日期：2026-06-05

## 1. 背景

OpBench v0.2 已经完成小规模真实闭环：

1. `pytorch_mini` 数据集包含 3 条 verified PyTorch CPU operator task。
2. 每条 task 都绑定 environment/source registry 和 task-local admission evidence。
3. baseline/gold admission 可以稳定验证 fail-to-pass 和 pass-to-pass。
4. 真实 Codex CLI 可以通过 `codex_action_bridge` 在 action interface 边界内完成修复。
5. 最终评分在 fresh workspace 和 task-scoped Docker runtime 中独立执行。

v0.2 同时暴露了下一阶段必须解决的问题：

1. 当前数据集规模太小，只有 3 条任务。
2. 当前 task 都是单文件 `torch/nn/modules/linear.py` overlay，不能代表更广泛的算子问题。
3. hidden tests 在 agent 修复阶段不可见，但 prompt 中仍容易被描述成 visible tests。
4. 容器基础工具不完整，例如缺少 `rg`。
5. 当前 source snapshot 未初始化 submodule，不能支撑 C++/CUDA/source build 类任务。
6. 复杂 runtime tier 还只是设计概念，没有进入实际 admission 试验。

因此，v0.3 的主题应从“平台最小闭环”推进到“可持续扩展的数据集协议”。v0.3 不应只简单增加 task 数量，而应把新增 task 过程中必然遇到的测试可见性、multi-file 修改、patch scope、环境能力和复杂任务分层固化为系统规则。

## 2. v0.3 目标

v0.3 的目标是构建一个更接近真实算子 benchmark 的 PyTorch-only 数据集和评测协议。

核心目标：

1. 将 PyTorch verified task 扩展到 10 条。
2. 只做 PyTorch，暂不引入 TensorFlow/JAX。
3. 引入 public/hidden test 分层，明确 agent 可见测试和 scorer-only 测试。
4. 支持 multi-file Python overlay，不能继续局限于单文件 overlay。
5. 建立 patch scope 检查，明确 agent 修改哪些文件会进入评分。
6. 标准化 Docker runtime 基础工具，减少 agent 因工具缺失失败。
7. 尝试构建 1 条 CUDA task 的 verified 试点；如果验证复杂度过高，则保留为 candidate/block evidence，不阻塞 v0.3 主线。
8. 继续使用 `codex_action_bridge` 作为参考真实 agent，不把多 agent 对比作为 v0.3 主目标。

## 3. 非目标

v0.3 不承诺以下能力：

1. 不做多 agent leaderboard。
2. 不引入 PyTorch 以外的框架。
3. 不默认支持完整 PyTorch source build。
4. 不把 CUDA/GPU task 作为必须完成项。
5. 不要求 C++/CUDA kernel 修改进入默认 verified set。
6. 不追求 20+ task 或大规模评测。
7. 不把环境不可用、硬件不可用或 source snapshot 不完整计为 agent failure。

## 4. 范围

### 4.1 数据集范围

v0.3 的正式数据集建议命名为：

```text
datasets/pytorch_v0.3/dataset.json
```

目标规模：

```text
10 verified PyTorch tasks
```

任务类型优先级：

1. Python-level operator/module behavior bug。
2. 涉及多个 Python 文件的 operator behavior 或 dispatch/ref 实现问题。
3. CPU package runtime 可复现的问题。
4. 需要额外依赖、环境变量或测试参数的 CPU task。
5. CUDA task 作为试点，优先选择最小可复现、环境可控、无需 full source build 的问题。

不进入 v0.3 默认 verified set 的任务：

1. 必须完整编译 PyTorch 的任务。
2. 必须修改 C++/CUDA kernel 才能修复的任务。
3. 必须依赖特定 GPU 型号、driver、cuDNN/cuBLAS 版本的任务。
4. admission replay 不稳定的 flaky task。

### 4.2 Agent 范围

v0.3 继续使用一个真实 agent：

```text
codex_action_bridge
```

原因：

1. 当前阶段重点是数据集扩展和评测协议，不是 leaderboard。
2. 单 agent 能更清楚暴露 action interface、prompt、测试可见性和环境问题。
3. 多 agent 引入会放大工程变量，容易掩盖数据集和环境问题。

v0.3 可以保留 agent adapter 扩展点，但不要求接入 Claude Code、OpenHands、Aider 等。

## 5. 设计原则

### 5.1 数据质量优先于数量

10 条 verified task 必须全部有 admission evidence。不能为了凑数量，把 `draft`、`blocked_environment` 或 `not_reproduced` task 混入正式 benchmark。

### 5.2 环境和测试可见性必须显式化

每条 task 必须明确：

1. 哪些测试 agent 可以看到和运行。
2. 哪些测试只用于最终 scoring。
3. 哪些文件允许 agent 修改并进入 runtime overlay。
4. 当前 runtime tier 能证明什么，不能证明什么。

### 5.3 Multi-file overlay 是 v0.3 核心能力

v0.3 的数据扩展不能继续只依赖 `torch/nn/modules/linear.py`。即使新增 task 中仍有单文件任务，系统也必须支持多文件 overlay，并通过至少 2-3 条 verified task 证明该能力。

### 5.4 CUDA 试点不阻塞主线

CUDA task 可以作为 v0.3 的探索项，但必须满足两个约束：

1. 如果当前机器和 Docker runtime 能稳定复现，则可以作为 1 条 verified CUDA task。
2. 如果需要大量 GPU 调度、driver 管理、source build 或硬件专用处理，则保留为 `blocked_environment` 或 `cuda_candidate`，并把完整分析记录到 v0.3 报告中。

## 6. Task Schema 调整

v0.3 允许修改 task schema。建议新增或明确以下字段。

### 6.1 测试可见性

建议结构：

```json
{
  "tests": {
    "public": [
      {
        "name": "test_visible_behavior",
        "command": ["python", "-m", "pytest", "test/foo.py::test_visible_behavior"],
        "description": "Agent-visible sanity test"
      }
    ],
    "fail_to_pass": [
      {
        "name": "test_hidden_regression",
        "command": ["python", "test/op_bench_hidden.py", "TestCase.test_bug"],
        "visibility": "hidden"
      }
    ],
    "pass_to_pass": [
      {
        "name": "test_existing_behavior",
        "command": ["python", "test/op_bench_hidden.py", "TestCase.test_existing"],
        "visibility": "hidden"
      }
    ]
  }
}
```

语义：

- `public`: agent 修复阶段可以运行；可以来自原仓库已有测试，也可以来自 task 提供的 public test patch。
- `fail_to_pass`: scorer-only 测试，用于判断问题是否修复。
- `pass_to_pass`: scorer-only 回归测试，用于判断是否破坏已有行为。
- hidden scoring tests 不应在 prompt 中被描述为 visible tests。

### 6.2 Test Patch 分层

当前 v0.2 只有一个 `test.patch`。v0.3 建议拆分为：

```text
artifacts/public_test.patch
artifacts/hidden_test.patch
```

规则：

1. `public_test.patch` 可在 agent workspace 中应用。
2. `hidden_test.patch` 只在 admission 和 final scoring 阶段应用。
3. 如果没有 public test，可以省略 `public_test.patch`，但 prompt 必须说明没有额外 public tests。
4. admission 必须检查 hidden tests 实际运行了测试用例，避免 v0.2 中缺少 `unittest.main()` 的问题再次出现。

### 6.3 Patch Scope

建议在 task manifest 中声明：

```json
{
  "patch_scope": {
    "allowed_paths": [
      "torch/nn/modules/linear.py",
      "torch/nn/modules/lazy.py"
    ],
    "mode": "enforced"
  }
}
```

规则：

- `allowed_paths`: agent patch 中允许进入评分的文件。
- `mode=enforced`: 如果 patch 修改了不在范围内的文件，task 结果标记为 `patch_out_of_scope`。
- `mode=filtered`: 只提取 allowed paths 的 diff 进入评分，其余修改记录到 metadata。
- v0.3 默认建议使用 `enforced`，避免 agent 以为修改生效但 scorer 忽略。

### 6.4 Source Loading

建议扩展 `source_loading`：

```json
{
  "source_loading": {
    "mode": "python_overlay",
    "overlay_paths": [
      "torch/nn/modules/linear.py",
      "torch/nn/modules/lazy.py"
    ],
    "sync_policy": "all_declared_paths",
    "allow_new_files": false
  }
}
```

v0.3 重点是 `overlay_paths` 从单文件扩展为多文件。

## 7. Runtime Tier 设计

v0.3 建议调整 runtime tier 定义，使其更贴合实际可执行范围。

| Tier | 名称 | 含义 | v0.3 状态 |
| --- | --- | --- | --- |
| T0 | `cpu_python_overlay_single` | v0.2 已验证的单文件 Python overlay | 兼容保留 |
| T1 | `cpu_python_overlay_multi` | 多文件 Python overlay，使用 installed wheel + runtime overlay | v0.3 核心 |
| T2 | `cpu_package_runtime` | installed package，可有额外依赖、环境变量、pytest 参数 | v0.3 支持 |
| T3 | `cuda_declared` | 声明 CUDA/GPU 需求，支持 preflight 和 skip reason | v0.3 试点 |
| T4 | `source_build_required` | 需要完整 source build 或 C++/CUDA 编译 | v0.3 只记录 candidate/block |

### 7.1 `cpu_python_overlay_multi`

这是 v0.3 的核心 runtime。

流程：

1. agent 读取 full source snapshot。
2. agent 修改 allowed paths。
3. final scoring 将所有 declared overlay paths 同步到 installed wheel 的 runtime overlay。
4. scorer 执行 hidden tests。

关键要求：

- overlay 文件必须存在于 source snapshot。
- overlay 文件必须能映射到 installed package 中的相同路径。
- admission evidence 必须记录每个 overlay 文件的 hash。
- patch scope 和 overlay paths 必须一致或有明确解释。

### 7.2 `cpu_package_runtime`

适用于无需修改 package source，或只需要执行 package 行为验证的 task candidate。

v0.3 中可支持：

- 额外 pip dependency。
- 环境变量。
- pytest 参数。
- import path 设置。
- preflight command。

但如果 task 需要修改 package 源码，仍应进入 overlay tier。

### 7.3 `cuda_declared`

CUDA task 试点的最低要求：

1. environment manifest 明确 CUDA base image、driver/runtime 要求。
2. preflight 能区分：
   - Docker 不可用。
   - GPU 不可用。
   - CUDA runtime 不可用。
   - PyTorch CUDA import 可用但具体 op 不可用。
3. admission 失败必须归因到环境或测试，而不是 agent。

如果本地无法稳定验证 CUDA task，v0.3 仍可交付：

```text
1 条 cuda_candidate task + blocked_environment evidence + 设计说明
```

但不计入 10 条 verified CPU/Python 主数据集。

## 8. Environment 设计

### 8.1 CPU Docker Image 标准化

v0.3 CPU image 应包含基础工具：

```text
git
ripgrep
grep
sed
diffutils
patch
python
pip
pytest
```

v0.2 中 Codex 尝试运行 `rg` 失败，说明 agent 真实使用的工具集合需要被环境显式支持或在 prompt 中明确限制。v0.3 建议优先在 Docker image 中补齐常用工具。

### 8.2 Environment Registry 扩展

建议 environment registry 增加：

```json
{
  "tools": {
    "git": "required",
    "rg": "required",
    "pytest": "optional"
  },
  "capabilities": {
    "python_overlay_single": true,
    "python_overlay_multi": true,
    "cuda": false
  }
}
```

preflight 应检查 required tools 是否存在。

### 8.3 容器生命周期

v0.3 继续沿用 v0.2 的 managed container label：

```text
op-bench.managed=true
```

新增建议：

- 在 experiment summary 中记录 task container 数量。
- 在异常退出时尽量清理容器。
- `manage_containers.py list` 输出关联 task id、attempt id、runtime tier。

## 9. Source Snapshot 设计

v0.3 继续允许本地 source snapshot，但要强化完整性记录。

建议 source registry 增加：

```json
{
  "file_manifest": {
    "mode": "overlay_paths",
    "paths": {
      "torch/nn/modules/linear.py": "sha256:...",
      "torch/nn/modules/lazy.py": "sha256:..."
    }
  }
}
```

对于 multi-file overlay，admission evidence 必须记录：

- base commit。
- source snapshot id。
- overlay paths。
- 每个 overlay path 的 baseline hash。
- gold patch 应用后的 overlay path hash。

这样后续可以判断 task 是否因为 source snapshot 变化导致 evidence 失效。

## 10. Admission Runner 调整

v0.3 admission 需要在 v0.2 基础上增加以下 gate。

### 10.1 Public Test Gate

如果 task 声明了 public tests：

1. baseline 阶段可运行 public tests。
2. gold 阶段也必须运行 public tests。
3. public tests 失败不能直接说明 bug 复现，但可以说明 public test 或环境不稳定。

建议新增状态：

```text
blocked_public_test
```

### 10.2 Hidden Test Execution Gate

admission 必须证明 hidden tests 实际运行了测试用例。

可记录：

- exit code。
- stdout/stderr。
- unittest/pytest collected count，如果可解析。
- 是否出现 `Ran 0 tests`。

如果 hidden test 没有实际执行测试，应标记：

```text
blocked_test
```

### 10.3 Patch Scope Gate

在 gold 和 agent scoring 中都检查 patch scope：

- gold patch 如果超出 scope，task manifest 必须更新 scope，否则 admission 失败。
- agent patch 如果超出 scope，结果为 `patch_out_of_scope`。

### 10.4 Multi-file Overlay Gate

admission 需要验证：

1. 所有 overlay paths 能同步到 runtime package。
2. 所有 overlay paths 的 hash 进入 evidence。
3. hidden tests 实际使用 overlay 后的 runtime package，而不是误 import source tree。

### 10.5 Flaky Replay Gate

对于新增 task，建议 admission 支持有限重复：

```text
--repeat 2 或 --repeat 3
```

规则：

- baseline/gold 结果必须一致。
- 非预期 signal exit，如 139，应记录为 `flaky_replay` 或 `unstable_environment`。
- retry 不能掩盖真实失败；报告必须记录 retry 次数和失败样本。

## 11. Agent Prompt 和 Action Interface

### 11.1 Prompt 调整

prompt 中必须明确：

1. Public tests 是 agent 可运行的测试。
2. Hidden tests 不可见，也不应尝试读取。
3. Final score 由 hidden fail-to-pass/pass-to-pass 决定。
4. 允许修改的文件范围。
5. 容器内可用工具集合。

### 11.2 Search Action

v0.3 可以考虑新增受控 search action：

```json
{"action": "search", "pattern": "LazyLinear", "paths": ["torch/nn"]}
```

好处：

- 降低对容器内 `rg`/`grep` 差异的依赖。
- 便于记录 agent 搜索行为。
- 可以限制搜索路径，避免 agent 读取 hidden test。

但这不是 v0.3 必须项。如果 Docker image 已包含 `rg`，可以先不新增 action。

### 11.3 Patch Export

`git_diff` 应继续只导出 workspace diff，但 v0.3 scorer 需要在导出后执行 patch scope validation。

## 12. Dataset Builder 和 Curation

v0.3 的数据扩展建议分成三层。

### 12.1 Candidate Pool

候选来源：

- PyTorch PR/issue。
- 关键词：`operator`, `nn`, `torch.ops`, `torch.compile`, `dtype`, `precision`, `cuda`, `cpu`, `dispatch`, `lazy`, `meta`, `decomposition`, `refs`。
- 优先 merged PR，避免修复方向不明确。

Candidate metadata 至少包含：

- PR URL。
- Issue URL。
- base commit。
- merge commit 或 patch source。
- suspected runtime tier。
- suspected overlay paths。
- hardware requirement。
- candidate status。

### 12.2 Draft Task Bundle

每个 candidate 进入 draft 时生成：

```text
tasks/pytorch/<task_name>/task.json
tasks/pytorch/<task_name>/issue.md
tasks/pytorch/<task_name>/artifacts/gold.patch
tasks/pytorch/<task_name>/artifacts/public_test.patch
tasks/pytorch/<task_name>/artifacts/hidden_test.patch
```

如果没有 public tests，可省略 `public_test.patch`。

### 12.3 Verified Dataset Slice

只有 admission 通过的 task 进入：

```text
datasets/pytorch_v0.3/dataset.json
```

建议数据集记录：

```json
{
  "dataset_id": "pytorch_v0.3",
  "version": "v0.3",
  "status": "verified",
  "target_task_count": 10,
  "tasks": []
}
```

## 13. 实验报告要求

v0.3 实验报告应包含：

1. 10 条 task 的来源、类型、runtime tier 和 admission 状态。
2. public/hidden test 分层统计。
3. single-file vs multi-file overlay 统计。
4. CUDA candidate 或 CUDA verified 试点结果。
5. gold agent 10-task 闭环结果。
6. Codex action bridge 10-task 评测结果。
7. patch scope failure 统计。
8. environment/source/tooling 问题总结。
9. 与 v0.2 的对比。
10. 明确哪些问题留到 v0.4。

## 14. 开发阶段计划

### Phase 1: Schema 和协议升级

目标：

- 扩展 task schema。
- 支持 public/hidden test patch。
- 支持 patch scope。
- 支持 multi-file overlay manifest。
- 更新 validator 和单元测试。

完成标准：

- v0.2 的 3 条 task 兼容通过。
- 新 schema 的 fixture task 可以通过 validate/admission。

### Phase 2: Runtime 和工具标准化

目标：

- 更新 CPU Docker image。
- preflight 检查 required tools。
- environment registry 记录 tool capabilities。
- action prompt 显示可用工具。

完成标准：

- `inspect_assets.py --check-docker` 能检查 tool availability。
- Codex 不再因为 `rg` 缺失失败。

### Phase 3: Multi-file Overlay

目标：

- source loading 支持多个 overlay paths。
- scorer 对多文件 patch 执行 scope validation。
- admission evidence 记录多文件 hash。

完成标准：

- 至少 2 条 multi-file overlay task verified。
- v0.2 单文件 task 继续通过。

### Phase 4: 数据集扩展到 10 条

目标：

- 构建 candidate pool。
- 将 task 从 draft 推进到 verified。
- 形成 `datasets/pytorch_v0.3/dataset.json`。

完成标准：

- 10 条 verified PyTorch task。
- 每条 task 有 stable admission evidence。
- 严格 dataset validation 通过。

### Phase 5: CUDA 试点

目标：

- 尝试 1 条 CUDA task。
- 如果环境能稳定复现，则进入 verified 或单独 CUDA slice。
- 如果复杂度过高，则记录 blocked evidence 和原因。

完成标准：

- 至少有一个 CUDA candidate 的完整分析。
- CUDA 不阻塞 CPU/Python 10-task 主线。

### Phase 6: 实验和报告

目标：

- 跑 gold 10-task 闭环。
- 跑 Codex action bridge 10-task 真实评测。
- 写 v0.3 实验报告。

完成标准：

- 产出 summary/results/patch/action logs。
- 报告说明 resolved rate、失败原因、环境问题和正确性边界。

## 15. 验收标准

v0.3 完成时必须满足：

1. `datasets/pytorch_v0.3/dataset.json` 包含 10 条 verified PyTorch task。
2. 每条 task 都有 task-local admission evidence。
3. 至少 2 条 task 使用 multi-file overlay。
4. public/hidden test 分层在 schema、prompt、runner、report 中生效。
5. patch scope validation 生效。
6. CPU Docker image 具备标准工具，preflight 可检查。
7. gold agent 在 10 条 task 上全部 resolved。
8. Codex action bridge 完成 10 条 task 评测。
9. CUDA task 至少完成 candidate 分析；若环境可控，尝试 verified。
10. 文档和实验报告完整记录问题、边界和后续计划。

## 16. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 找不到足够多适合 CPU/Python overlay 的真实 PR | 数据集无法达到 10 条 | 提前建立 candidate pool，允许 multi-file/ref/dispatch Python task |
| multi-file overlay 和 installed wheel 文件布局不一致 | admission 失败 | admission 增加 overlay path 映射检查 |
| hidden/public test 分层改动破坏 v0.2 task | 回归风险 | 保留 backward compatibility，先迁移 v0.2 三条 task |
| CUDA 环境复杂度过高 | 拖慢 v0.3 | CUDA 设为试点，不计入主线验收 |
| agent 修改 scope 外文件 | 分数解释混乱 | patch scope validation 返回 `patch_out_of_scope` |
| task replay flaky | verified 质量下降 | admission repeat 和 flaky classification |

## 17. 已确认决策

当前已确认：

1. v0.3 数据规模目标是 10 条 verified task。
2. v0.3 只做 PyTorch。
3. 允许修改 schema。
4. multi-file overlay 是核心目标。
5. CUDA task 可以尝试 1 条，但不阻塞主线。
6. `patch_scope.mode` 默认使用 `enforced`，agent patch 超出 scope 时返回 `patch_out_of_scope`。
7. CUDA task 如果 verified，先放入单独的 `pytorch_cuda_preview` slice，不放入主 `pytorch_v0.3` 10-task 验收集。
8. public tests 不强制每条 task 至少 1 条；如果没有 public tests，prompt 必须明确说明没有额外可见测试。
9. v0.3 先补齐 Docker 工具，不新增 `search` action；只有当 Codex 评测继续明显受搜索工具影响时，再引入受控 search action。
