# OpBench v0.2 开发者指南

本文说明 v0.2 新增的平台化能力。v0.1 的 action-interface 评测闭环仍然保留，详见 `../v0.1/developer_guide.md`；v0.2 在其上增加环境资产、源码资产、准入证据和数据集策展流程。

## 1. v0.2 边界

v0.2 的重点是扩大 verified task 前先把流程固定下来：

- 环境通过 `environments/registry.json` 统一登记。
- 源码快照通过 `sources/registry.json` 统一登记。
- task 可以使用 `environment_ref` 和 `source_ref` 引用资产。
- `scripts/run_admission.py` 负责 baseline/gold replay，并写入 admission evidence。
- verified dataset entry 必须引用 task-local stable evidence。
- `scripts/curate_dataset.py` 从混合数据集生成 verified-only slice。
- `scripts/inspect_assets.py` 和 `scripts/manage_containers.py` 用于环境与容器状态管理。

v0.2 仍不承诺大规模 GPU 调度、多 agent leaderboard 或 full PyTorch source build 任务默认通过。

## 2. 新增模块

| 模块 | 作用 |
| --- | --- |
| `src/op_bench/registry.py` | 加载 environment/source registry，解析 task refs，并把 registry defaults 与 task overrides 合并。 |
| `src/op_bench/admission.py` | 运行 baseline/gold admission，生成完整 evidence bundle 和 task-local stable evidence。 |
| `src/op_bench/assets.py` | 检查 registry 中环境镜像和源码快照的本地 cache 状态。 |
| `src/op_bench/containers.py` | 盘点和清理带 `op-bench.managed=true` label 的 Docker 容器。 |
| `src/op_bench/curation.py` | 生成 verified-only dataset slice 和 dataset summary。 |

## 3. Registry 规则

环境资产位于：

```text
environments/registry.json
```

源码资产位于：

```text
sources/registry.json
```

加载 task 时，registry 字段作为默认值，task 内联字段作为 override。也就是说：

- 公共 Docker image、digest、preflight 写在 environment registry。
- 某条 task 特有的 `source_loading`、测试命令、timeout 仍写在 task manifest。
- source registry 负责记录 snapshot 的本地路径、commit、submodule policy 和适用 task。

如果 source registry 的 commit 与 task `source.base_commit` 不一致，系统会拒绝解析，避免错用源码快照。

## 4. Admission 证据

完整运行证据写入：

```text
runs/admission/<task_id>/<run_id>/
```

其中包含完整命令日志，适合调试，不作为稳定数据集资产提交。

task-local stable evidence 写入：

```text
tasks/<framework>/<task>/admission/evidence.json
```

该文件只保留可审计摘要：task hash、environment/source ID、baseline/gold status、测试计数、decision。它不包含完整命令日志，避免把数据集膨胀成运行记录集合。

运行 admission：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_admission.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output-dir runs/admission/pytorch__149693__lazylinear_init/v0.2-migration \
  --write-task-evidence
```

只有 baseline 为 `baseline_reproduced` 且 gold 为 `resolved` 时，decision 才是 `verified`。

## 5. 数据集校验

普通校验允许混合 draft 和 verified task：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

严格校验要求 dataset 和每条 entry 都是 verified：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/<verified_slice>/dataset.json --require-verified
```

verified entry 必须满足：

- `admission_evidence` 文件存在。
- evidence 的 `task_id` 与 entry 一致。
- evidence 的 `task_manifest_hash` 与当前 `task.json` 一致。
- evidence decision 为 `verified`。
- baseline/gold 状态分别为 `baseline_reproduced` 和 `resolved`。
- `environment_ref` 和 `source_ref` 能从 registry 解析，且 runtime/source loading/commit 兼容。

## 6. Dataset Curator

从混合数据集生成 verified-only slice：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/curate_dataset.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --output-dataset datasets/pytorch_mini_v0.2/dataset.json \
  --output-summary datasets/pytorch_mini_v0.2/summary.json \
  --verified-only \
  --dataset-id pytorch_mini_v0.2 \
  --version v0.2
```

Curator 会先校验源数据集，再校验输出数据集。输出被标记为 `verified` 时，会自动使用严格 evidence 校验。

## 7. 环境与容器管理

检查 registry 资产状态：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py \
  --check-docker \
  --output runs/assets/v0.2-foundation.json
```

列出 OpBench 管理的容器：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/manage_containers.py list
```

预览可清理的已停止容器：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/manage_containers.py prune-stopped
```

实际清理必须显式加 `--execute`：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/manage_containers.py prune-stopped --execute
```

该命令只处理带 `op-bench.managed=true` label 且状态为 `created`、`exited` 或 `dead` 的容器，不会删除运行中的容器。

## 8. 当前状态

当前 v0.2 foundation 和第一轮数据扩展已完成：

- `pytorch__149693__lazylinear_init` 已迁移到 registry + stable admission evidence。
- `pytorch__160952__bilinear_lazy_check` 已修正 hidden test 构建问题，并通过 baseline/gold admission。
- `pytorch__147599__lazylinear_state_forward` 已从真实 PyTorch PR 构建 task bundle，并通过 baseline/gold admission。
- `datasets/pytorch_mini/dataset.json` 当前包含 3 条 verified PyTorch CPU tasks。
- 三条 task 均使用 `pytorch-cpu-torch2.6.0-py311` 环境和 `cpu_python_overlay` source loading。

当前 verified task 列表：

| Task | PR | Issue | 关注点 |
| --- | --- | --- | --- |
| `pytorch__149693__lazylinear_init` | https://github.com/pytorch/pytorch/pull/149693 | https://github.com/pytorch/pytorch/issues/149691 | `LazyLinear` lazy 初始化后 reset 参数语义 |
| `pytorch__160952__bilinear_lazy_check` | https://github.com/pytorch/pytorch/pull/160952 | https://github.com/pytorch/pytorch/issues/160407 | `Bilinear` 提前参数校验阻断 lazy 子类构造 |
| `pytorch__147599__lazylinear_state_forward` | https://github.com/pytorch/pytorch/pull/147599 | https://github.com/pytorch/pytorch/issues/147389 | `LazyLinear.load_state_dict()` 后首次 forward 未更新 `in_features` |

本轮验证命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json --require-verified
```

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --agent gold \
  --output runs/experiments/pytorch_mini_gold_v0.2.json
```

`gold` agent 闭环结果：3/3 resolved，median runtime 约 34.7 秒。该结果证明 dataset、source snapshot、Docker 环境、patch replay 和 scoring 流程可运行；它不是实际 agent 能力分数。真实 agent 评分仍应使用 `codex_action_bridge` 或后续新增的 action-interface agent adapter。

## 9. 本轮问题记录

本轮扩展暴露了两个数据集构建问题：

1. `pytorch__160952__bilinear_lazy_check` 的初始 test patch 缺少 `unittest.main()`，导致测试文件被执行但没有运行测试。修复后 admission 才能正确识别 baseline 是否复现。
2. 新增 patch 文件的 hunk 行数必须包含空行，否则 `git apply` 可能截断测试文件或报 `corrupt patch`。构建 task 时应先在临时 git 仓库中运行 `git apply` 检查。

另一个观察项：一次 gold experiment 中 `pytorch__160952__bilinear_lazy_check` 的 baseline fail-to-pass 出现过 exit 139；随后单独 admission stability check 恢复为预期 exit 1，并且 gold resolved。当前不把它标记为 blocked，但后续如果出现重复，应把 runner 增加“非预期信号分类”和重试策略。
