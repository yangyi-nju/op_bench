# OpBench v0.5 设计方案

日期：2026-07-08（2026-07-11 修订 release contract）

## 0. 修订决策

v0.5 的最终产物是一个**可发布的累计 PyTorch 数据集版本**，不是为分类表凑数的候选集合，也不包含 v0.6 之后的边界/API 兼容维度。release contract 固定为：

1. `datasets/pytorch_v0.5/dataset.json` 包含 v0.4 的 13 条和 4 条新 precision task，共 17 条 verified task。
2. 其中 6 条构成 precision slice（2 条 v0.4 锚点 + 4 条新增）；P4 无通过 admission 的样本时保持 coverage gap，不用版本错配或非精度 task 填充。
3. 同一 Codex 版本、同一 Linux remote execution policy 对 17 条运行 3-repeat，共 51 个有效 attempt；明确的 `environment_unavailable/environment_error` 不得计入 agent 分数，agent 补丁导致的 timeout 等结果仍按失败计分。
4. 同时发布累计总分和 precision slice 分数，保留 task/attempt 原始 patch、action log、结构化 summary 和 admission evidence。
5. v0.6/v0.7 分别扩边界和设备/API 兼容；这些不是 v0.5 的完成前置。

该修订把原先的“6-8 条新增、每个子类至少一条”从硬 release gate 调整为候选检索目标。OpBench 的核心原则仍是 evidence before verified；覆盖缺口必须透明，但不能驱动低质量 admission。

## 1. 背景

v0.4 完成了平台侧从"单 tier CPU"到"多 tier + 远程 GPU + kernel 级修改"的扩展，Codex CLI 在 13 条 task 上 3 次重复的 resolved rate 达到 84.6%（33/39）。但 v0.4 暴露了两个 benchmark 层面的结构性问题：

1. **task 池缺乏领域特色**。v0.3/v0.4 的候选 PR 来源是"翻 PyTorch 近期 merged PR"，落到 benchmark 上的问题类型是散的：既有 LazyLinear 初始化、autograd state、DataLoader dispatch，也有 CUDA kernel index underflow。这些问题共同点只有"都是 PyTorch bug"，没有算子领域的特殊性。SWE-bench 已经覆盖了"repo-level 通用 bug 修复"，OpBench 若不能在维度上区分开，就只是"SWE-bench 的 PyTorch 子集"。
2. **评测指标单一**。目前只有 resolved rate 一个数字，掩盖了大量信息：agent patch 是否简洁、是否引入 pass-to-pass 回归、不同 tier 的难度差异等。指标不细化，多 agent 对比结果就是"都 85% 左右"的糊状产出。

v0.5 不追求一版本覆盖所有问题维度，也不急于引入第二个 agent。核心是**先把 benchmark 能力建扎实**：确定算子领域的问题分类，从"精度问题"这一个维度开始建垂直数据集，同时把评测指标从单一 resolved rate 扩展到多维打分。当所有问题维度和评测维度都补齐后，再开展多 agent 对比才有意义。

## 2. v0.5 目标

1. **确定算子问题分类框架**（不实现全部，只落地"精度"维度）。产出一份"算子问题分类 × PR 检索策略"文档，为 v0.5 及后续版本的 task 筛选提供方法论。
2. **构建精度问题垂直切片**。围绕数值精度问题固化 6 条已通过 admission 的 task，并显式记录未覆盖子类。
3. **扩展评测指标**。在 resolved rate 之外补齐 patch 简洁度、pass-to-pass 单独统计、按 bug 类型分组通过率、按 tier 加权综合分等多维度指标。
4. **实验运行体验重构**。做两件事：把所有 tier 的执行统一到远程 Linux 服务器（消除 macOS QEMU 段错误 + 让 CPU / GPU 环境同源）；把 `run_experiment.py` 改造成幂等可续跑（中断后 resume 只重跑未完成 attempt）。
5. **平台清理**。删除未使用的 public test 机制（`--no-public-tests` flag 及相关代码路径）。

## 3. 非目标

1. **不做其他问题维度**（边界、性能、设备/API 兼容）。这些留给 v0.6+，逐版本推进。
2. **不做多 agent 对比**。Claude Code 端到端仍推迟。v0.5 只跑 Codex 一个 agent，用于验证新指标口径和精度类 task 的 admission 通过率。
3. **不做性能类 task**。性能问题评测对硬件、编译器、驱动版本极度敏感，需要单独的评测框架，v0.6+ 单开话题。
4. **不做横向扩框架**（TorchAudio / TorchVision / JAX）。v0.5 仍聚焦 PyTorch。

## 4. 算子问题分类框架

参考真实算子问题的分布，OpBench 的问题维度按以下 4 类展开，v0.5 只做第 1 类：

| 维度 | v0.5 状态 | 典型 bug 举例 | 评测特点 |
| --- | --- | --- | --- |
| **1. 精度问题** | 本版本落地 | float16 累积溢出、bfloat16 舍入误差、mixed precision 输出不一致、log-space 数值稳定性 | tolerance-aware 断言，可 deterministic 复现 |
| 2. 边界问题 | v0.6 | 空张量、shape 为 0/1 的退化情况、超过 int32 上限的 index、越界 grid | 逻辑断言，可精确复现 |
| 3. 设备/API 兼容 | v0.7 | `.to(device)` 漏转、CPU/CUDA 行为不一致、autocast 场景下 dtype 推断异常、跨 device 参数标量 | 双 device 对比断言 |
| 4. 性能问题 | v0.8+（需专门框架） | kernel 效率、内存分配次数、编译时间回归 | 需硬件基线、非确定性容忍、独立设计 |

**v0.5 的边界严格限定在第 1 类**。原因：
- 精度类问题在 PyTorch issue tracker 中有清晰的标签（`topic: numerical`、`module: precision`），检索策略最好落地
- 断言形式统一（`assertClose(rtol, atol)`），admission 稳定性高
- 与 SWE-bench 差异化最明显 —— 通用 SWE 数据集里几乎没有"数值精度"这一类

## 5. 精度问题的子分类

在"精度问题"这一顶层维度下，进一步细分为 5 个子类。这个分类同时决定了 v0.5 的 task 筛选策略和后续 tag 体系：

| 子类 | 内涵 | 典型 PR 特征 |
| --- | --- | --- |
| **P1. reduction 累积/提升精度** | sum / mean / norm 等 reduction 的 accumulator 或 promotion dtype 不正确 | 修改 accumulator dtype、修 reduction promotion、切换到稳定算法 |
| **P2. dtype 传播/转换损失** | 隐式 cast、decomposition、fake/meta 路径的 output dtype 与真实算子不一致 | 修 upcast/downcast、`dtype=` 传递、fake/decomposition dtype 推断 |
| **P3. 混合精度 (autocast) 不一致** | `torch.autocast` 场景下某算子 output dtype 与 gold reference 不一致 | 修 autocast wrap list、修算子的 `@autocast_custom_fwd` 装饰 |
| **P4. 数值不稳定** | log / exp / softmax / sigmoid 等操作在极端输入下 NaN / Inf | 加 log-sum-exp trick、加 clamp、加 stable variant |
| **P5. Kernel 数值正确性 bug** | CUDA kernel 内的累加、bounds 或 tail loop 令数值结果错误 | 修 kernel `.cu` 文件中的 size underflow、integer bounds、reduce tree |

v0.4 已有的 3 条 CUDA task 中，`132835_njt_sdpa_autocast` 属于 P3，`144009_softmax_ilpreduce_size` 属于 P5（size underflow 直接导致 log_softmax + exp 求和不为 1）。可作为 v0.5 分类的现成锚点。

候选检索目标仍是每个子类都有候选；正式 release 只纳入 admission verified task。当前 6 条覆盖 P1、P2、P3、P5，P4 作为已知 coverage gap 发布。

## 6. PR 检索策略

放弃"翻近期 merged PR"这种漫无目的方式。v0.5 采用三条并行的检索路径，每条都可脚本化：

### 6.1 Label + keyword 组合检索

用 GitHub CLI 直接查 PyTorch 仓库：

```bash
# P1 / P4 累积误差、数值不稳定
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search 'label:"topic: numerical" is:merged' \
  --json number,title,url,mergedAt,labels

# P2 / P3 dtype 精度、autocast
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(autocast OR "mixed precision" OR "dtype precision") is:merged' \
  --json number,title,url,mergedAt

# P5 kernel 精度 bug
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(numerical OR precision OR "log_softmax" OR "softmax") path:aten/src/ATen/native/cuda' \
  --json number,title,url,mergedAt,files
```

对每个子类维护一份 keyword pack（写在 `docs/v0.5/candidate_search.md`）。

### 6.2 测试文件反向检索

PyTorch 的测试命名规律显式暴露 bug 类型。用 `git log --all -p` 在测试文件里 grep bug 修复:

```bash
# 找测试文件里带 tolerance / precision 断言变化的 PR
git -C .op_bench_cache/sources/pytorch/pytorch grep -l "assertClose\|rtol=\|atol=\|torch.testing.assert_close" \
  test/test_reductions.py test/test_ops.py test/test_nn.py | \
  xargs git log --all --oneline --

# 找 test_*_precision.py 系列
find . -name "test_*precision*.py"
```

命中率高的原因：测试作者写 `assertClose(rtol=1e-3)` 而非 `assertEqual` 的时候，往往是在修一个已知精度问题。

### 6.3 Issue label 反向定位 PR

PyTorch 部分 issue 有 `topic: numerical stability`、`module: numerical`、`accuracy` 等 label。走 issue → 关联 PR 通路：

```bash
gh issue list --repo pytorch/pytorch --state closed --limit 200 \
  --search 'label:"topic: numerical" is:closed' \
  --json number,title,url,timelineItems  # 从 timeline 里找 "referenced by PR #N"
```

### 6.4 筛选 filter（v0.4 经验教训）

候选 PR 进入 admission 前先跑一遍 hard filter：

- **base commit 必须落在 2.6 release 附近**。post-2.6 nightly 会引入 wheel 不兼容符号（v0.4 反复踩坑）
- **PR 描述里不能是 "add support for"**。这类 PR 的 baseline 通常已通过 fallback 路径，无法产生 fail_to_pass
- **修改文件数 ≤ 3**。多文件 patch 对 agent 太难，且和 admission scope 校验冲突大
- **修改行数 20-200**。太短说明改动琐碎、baseline 语义不够明确；太长说明是重构而非 bug fix
- **测试改动必须存在**且是 tolerance 断言变化或新增用例

用 `scripts/screen_candidates.py`（v0.5 新增）自动过滤上述条件，输出候选清单。

## 7. 评测指标扩展

`summary.json` 从当前"每 agent resolved rate"扩展为多维打分。所有指标都可以从现有 `results.jsonl` 复算，不需要重跑历史实验。

### 7.1 新指标定义

| 指标 | 计算方式 | 意义 |
| --- | --- | --- |
| **resolved_rate** | `resolved / total` | 现有指标，主导评分保留 |
| **patch_conciseness** | `median(gold_patch_lines / agent_patch_lines)`，clamp 到 `[0, 1]` | agent patch 越简洁越接近 gold，值越高 |
| **pass_to_pass_kept_rate** | `pass_to_pass_passed / pass_to_pass_total` 单独统计 | 防止 agent 打了 hidden test 但破坏其他测试 |
| **fail_to_pass_only_rate** | `(fail_to_pass 全过 & pass_to_pass 全过) / total` | 严格 resolved，等于当前 resolved_rate 但显式区分 |
| **regression_rate** | `(fail_to_pass 全过但 pass_to_pass 挂) / total` | 只看引入回归的比例 |
| **tier_weighted_score** | `Σ (resolved × weight_tier) / Σ weight_tier`，weight: cpu=1, cuda_py=2, cuda_kernel=3 | 用 tier 权重平衡任务难度 |
| **per_problem_type_resolved_rate** | 按 `operator.problem_type` 分组的 resolved rate | 揭示 agent 在哪类问题上强/弱 |
| **median_runtime_sec** | 现有指标，保留 | 观察 agent 效率 |

### 7.2 实现位置

- `src/op_bench/reporter.py` 新增 `compute_extended_metrics(results, dataset)` 函数
- `scripts/run_experiment.py` 在写 `summary.json` 时调用
- `scripts/aggregate_experiments.py`（新增）：接受一个或多个 `runs/*/results.jsonl` 目录，聚合出跨批次的综合报告

### 7.3 task.json 补充字段

为支持"按 bug 类型分组"，把 v0.5 引入的分类信息写入 task.json：

```json
{
  "operator": {
    "framework": "pytorch",
    "component": "aten.native.cuda.softmax",
    "operator_name": "torch.nn.functional.log_softmax",
    "problem_type": "kernel-precision",
    "problem_dimension": "precision",
    "problem_subclass": "P5",
    "tags": ["cuda-kernel", "softmax", "tail-loop-underflow"]
  }
}
```

新增字段：
- `problem_dimension`：`precision | boundary | device_compat | performance`（v0.5 只用 `precision`）
- `problem_subclass`：`P1..P5`（精度维度下的子类）

现有 3 条 CUDA task 需要回填。CPU 老 task 允许 `problem_dimension` 为 `null`（表示尚未归类），聚合指标按 `null` 单独一组处理。

## 8. 实验运行体验重构

v0.4 的两个执行痛点：CPU 跑在 macOS Colima 时偶发 QEMU 段错误、Codex rate-limit sleep 5 小时期间人为中断就得全量重跑。v0.5 一次解决。

### 8.1 所有 tier 统一到远程服务器

不再区分"本地 CPU / 远程 GPU"。所有 tier 默认走 `remote_docker` backend，本地 Docker 只留作调试出口。

**改动**：

- `environments/registry.json` 每条环境都填 `host` 字段（默认指向服务器名）。CPU 环境（`pytorch-cpu`）之前无 host，现在默认为 `gpu-a10`（这台机器同时承载 CPU 和 GPU tier）。
- `src/op_bench/remote.py`：`RemoteDockerExecutor.gpus` 允许传 `None` 跳过 `--gpus` 参数。当前是硬编码 `"all"`，需要改成"环境声明需要 GPU 才传"。CPU 环境注册时 `hardware.requires_gpu=false`，remote executor 据此跳过 `--gpus`。
- Codex CLI 保持运行在本地控制端，通过 action bridge 操作远端容器；安装和账号配置写入 `docs/v0.5/setup_remote_agent.md`。
- 环境变量 `OP_BENCH_FORCE_LOCAL_DOCKER=1` 保留，强制走本地 backend（调试用）。

**并发**：远端 20 核机器上 CPU task 允许 task 级并发。`run_experiment.py` 新增 `--max-parallel N`（默认 1，保持向后兼容）。GPU tier 仍串行（`--gpus all` 且 kernel_build 吃满 CPU）。

**rsync 开销**：CPU task workspace 需要同步到远端，耗时取决于 snapshot 大小和链路。同步采用有界重试；并发度应同时考虑 Codex session、网络带宽和远端磁盘，而不是只看 CPU 核数。

### 8.2 断点续跑（Resume）

`run_experiment.py` 从"一次性 batch"改为"幂等续跑"。中断后重新执行同一命令即可继续。

**幂等 key**：`(task_id, agent, attempt)` 三元组。同一 `--output-dir` 下三元组唯一。粒度定在 attempt 级 —— attempt 内部失败直接整个重跑，代价可控（median 45s，最长 kernel_build ~90min），不做更细的 phase 级 checkpoint（复杂度不值）。

**Resume 语义**：

- 默认行为：如果 `--output-dir` 已存在且含 `results.jsonl`，读现有已完成的三元组集合，跳过它们。
- `--fresh`：显式覆盖，清空 output-dir 从头开始。
- `--only-tasks <task_id>...`：只跑指定 task_id（配合 resume 使用，用于精准回放某几条）。

**实现要点**：

- **增量写盘**：每完成一个 attempt 立即以行追加方式写入 `results.jsonl`，每次 write 后 `flush() + os.fsync()`。程序被 `kill -9` 时最近一次完成的记录也在盘上。
- **Baseline 也纳入 resume 范围**：baseline 和 agent record 统一增量写入 `results.jsonl`，baseline 幂等 key 为 `task_id`，resume 时直接读，不重跑。
- **run_state.json**：output-dir 根写一个 `run_state.json`，记录 task replay signature、agent 列表和 repeat 数。resume 时校验一致性，task 内容、agent 或 repeat 变化时拒绝错误续算。
- **summary.json 持续聚合**：summary 从 `results.jsonl` 的最新逻辑记录计算，在 baseline 和每个 attempt 落盘后覆写。transient 失败保留在原始 JSONL 中供审计，但不占用完成 key；补跑成功后只计同一 key 的最新记录。
- **workspace 目录**：每个 attempt 用独立子目录（`workspaces/<task>/<agent>/attempt_<N>/`），resume 重跑同一 attempt 时先清理该子目录。不会误删已完成 attempt 的 workspace。
- **损坏行容忍**：读 `results.jsonl` 时按 JSONL 行解析，解析失败的行跳过（可能是崩溃时半写入），宁可少算不 crash。

**边界情况**：

- Attempt 跑到一半崩溃 → 未落盘 → resume 时重跑。可接受。
- Rate-limit sleep 期间人为 Ctrl-C → 已完成 attempt 都在盘上 → resume 继续。核心收益场景。
- Codex CLI 二进制升级 → 应使用新 output-dir，避免混入同一正式实验；版本保存在 attempt metadata 中供审计。服务器环境或 task replay spec 变化会改变 task signature，禁止直接 resume。

**Baseline 缓存跨 run**：进一步优化 —— baseline 只依赖 `task.source_snapshot` + `hidden_test.patch`，与 agent / attempt 无关。可以做一个跨 run 的 baseline 缓存目录（`runs/_baseline_cache/<task_id>_<snapshot_hash>_<hidden_patch_hash>.json`），后续任何 run 遇到同样的 baseline signature 直接读缓存。v0.4 里 baseline 每次都重跑，浪费不少时间。这个改动放 v0.5 做（不大）。

### 8.3 删除 public test 机制

现在没有 task 携带 `public_test.patch`，机制未启用也不打算启用。清理内容：

- 从 `scripts/run_experiment.py` 删 `--no-public-tests` flag
- 从 `src/op_bench/evaluator.py` 和 agent prompt 里删 `public_test_patch` 相关分支
- `task.json` schema 里保留字段但标 `deprecated`，避免破坏历史 evidence
- 保留 `docs/v0.4/public_test_ablation.md` 作为历史设计记录，在 CHANGELOG 里记录运行机制不再参与 v0.5 评分

### 8.4 preflight 补充 problem_dimension 校验

`scripts/preflight_task.py` 增加检查：新提交的 task 若 `admission_status == "verified"` 则 `problem_dimension` 不能为空。历史 task 允许为空。

## 9. 数据集目标：pytorch_v0.5

v0.5 数据集 `datasets/pytorch_v0.5/dataset.json`：

- **精度维度新增 task**：4 条
- **保留 v0.4 全部 13 条**（10 cpu + 2 cuda_py + 1 cuda_kernel）
- **总计**：17 条 verified task

v0.4 的 3 条 CUDA task 中 `132835` 和 `144009` 归入精度维度（分别为 P3 和 P5）。加上 4 条新增 task，precision slice 共 **6 条**。coverage matrix 是发布资产的一部分，P4 缺失标记为 N/A。

## 10. 实施顺序

Phase 1 的两个基础设施改动（Resume + 跑机统一）必须先于 Phase 2 的 task 筛选，否则 Phase 2/3 期间还在受同样的运行痛点。Resume 又必须先于跑机统一 —— 跑机统一后单次 run 更长（rsync + 远端），中断代价更高，先补上 resume 再迁移。

```text
Phase 1a: 断点续跑 (run_experiment.py 幂等化)                [2-3 天]
  · results.jsonl 增量写 + fsync（baseline 与 agent record 同一日志）
  · run_state.json 一致性校验
  · summary.json 持续聚合
  · --fresh / --only-tasks 参数
  · baseline 跨 run 缓存

Phase 1b: 跑机统一到服务器                                    [1-2 天]
  · environments/registry.json 所有环境填 host
  · RemoteDockerExecutor.gpus 可选
  · 服务器安装 codex CLI + 账号配置
  · --max-parallel N 参数
  · OP_BENCH_FORCE_LOCAL_DOCKER 调试出口

Phase 1c (与 1a/1b 并行):                                      [并行 2 天]
  · 评测指标扩展 (reporter.py + aggregate_experiments.py)
  · candidate_search.md 文档 + screen_candidates.py 脚本
  · 平台清理 (删 public test、task.json schema 加 problem_dimension)
  · 回填 132835 / 144009 的 problem_dimension

Phase 2: task 筛选                                             [1-2 天]
  · 按 5 个子类跑 PR 检索
  · screen_candidates 硬过滤
  · 每子类 3-5 候选入池

Phase 3: task 制作 + admission                                 [2-3 周，主要瓶颈]
  · 逐条走 gold/hidden patch 提取 + admission
  · preflight_task 离线校验

Phase 4: 全量评测 + 报告                                       [1-2 天]
  · 同一 Codex/remote policy 对 17-task dataset 跑 3-repeat（51 attempts）
  · 验证新指标口径
  · 实验报告 + 文档同步
```

**关键风险点**：

1. **Phase 3 不可控**。候选配额不构成 admission 配额；只发布稳定 verified task，并在 coverage matrix 中保留空缺。
2. **Phase 1a 的 baseline 缓存** 有一个陷阱：如果 task 的 hidden_test.patch 变化了但缓存未失效，会用错的 baseline 打分。缓存 key 里必须包含 hidden_test.patch 的 hash，preflight 也要复查缓存一致性。
3. **控制端 Codex quota 与远端传输竞争**。提高 `--max-parallel` 会同时增加账号请求、SSH/rsync 带宽和远端磁盘压力，应从 1-3 并发实测，而不是只按 CPU 核数配置。

## 11. 验证方式

1. `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'` — 全部通过（v0.5 冻结时 169 个）
2. `PYTHONPATH=src python scripts/preflight_task.py --all` — v0.5 全部 task 离线预检通过
3. `PYTHONPATH=src python scripts/validate_dataset.py datasets/pytorch_v0.5/dataset.json` — dataset 校验通过
4. 每条新 task 的 admission evidence 存在且 hash 匹配
5. Codex 3-repeat 全量评测包含 17 tasks / 51 个有效 attempts，产出累计和 precision slice summary
6. 精度维度 resolved rate 若显著低于 v0.4 整体 84.6%，说明维度设计有效引入了新难度
7. **Resume 冒烟测试**：跑一次 5-task 实验，跑到第 3 条时 Ctrl-C，重新启动应从第 3 条继续，不重跑前 2 条。`results.jsonl` 行数与预期一致。
8. **跑机统一冒烟测试**：CPU task 在服务器上跑通 baseline + gold + agent 三阶段，`--gpus` 参数缺失时 remote executor 不报错。

## 12. 文件变更清单

### 新文件

| 文件 | 用途 |
| --- | --- |
| `docs/v0.5/design.md` | 本文档 |
| `docs/v0.5/candidate_search.md` | 5 个子类的 keyword pack 和检索命令 |
| `docs/v0.5/setup_remote_agent.md` | 本地 Codex 控制端与远程 Docker 配置 |
| `docs/v0.5/experiment_report.md` | 17-task / 51-attempt 正式实验报告 |
| `scripts/screen_candidates.py` | 自动过滤候选 PR（base commit、修改文件数、测试变更） |
| `scripts/aggregate_experiments.py` | 跨 batch 聚合报告 |
| `src/op_bench/resume.py` | Resume 状态管理：`ResultsStore`、`RunState`、baseline 缓存 |
| `tasks/pytorch/<新 4 条>/` | precision 新 task |
| `datasets/pytorch_v0.5/dataset.json` | v0.5 数据集 |
| `runs/_baseline_cache/` | 跨 run baseline 缓存目录（gitignore） |

### 修改文件

| 文件 | 变更 |
| --- | --- |
| `src/op_bench/reporter.py` | 新增 `compute_extended_metrics` 及多维指标 |
| `src/op_bench/task.py` | 补 `problem_dimension` / `problem_subclass` 属性 |
| `schemas/task_manifest.schema.json` | operator 下新字段 |
| `scripts/run_experiment.py` | 接入 Resume 机制、`--fresh`/`--only-tasks`/`--max-parallel`；删 `--no-public-tests`；接入扩展指标 |
| `scripts/preflight_task.py` | 补 `problem_dimension` 空校验 |
| `src/op_bench/evaluator.py` | 删 public test 相关分支 |
| `src/op_bench/remote.py` | `RemoteDockerExecutor.gpus` 允许为 `None`；CPU 环境跳过 `--gpus` 参数 |
| `src/op_bench/environment.py` | `prepare()` 根据环境 `hardware.requires_gpu` 决定 gpus 参数 |
| `environments/registry.json` | 所有条目填 `host` 字段，CPU 环境标记 `requires_gpu=false` |
| `tasks/pytorch/132835_njt_sdpa_autocast/task.json` | 回填 `problem_dimension=precision, subclass=P3` |
| `tasks/pytorch/144009_softmax_ilpreduce_size/task.json` | 回填 `problem_dimension=precision, subclass=P5` |
| `.gitignore` | 加 `runs/_baseline_cache/` |
| `CHANGELOG.md` | v0.5 条目 |
| `README.md` / `README.zh-CN.md` | 更新数据集、评测指标、执行方式说明 |
| `docs/README.md` / `docs/README.zh-CN.md` | 加 v0.5 索引 |

## 13. 后续版本方向

v0.5 落地后，OpBench 的能力路线图：

- **v0.6**：边界问题维度。子分类：空张量、shape 退化、int32 溢出、grid 越界等。同精度维度体量。
- **v0.7**：设备/API 兼容维度。子分类：CPU/CUDA 行为一致性、autocast dispatch、跨 device 参数标量、`.to(device)` 传递。
- **v0.8**：性能问题维度（需专门评测框架）。要在 admission 阶段解决硬件基线、非确定性容忍、跨硬件对比的问题。
- **v0.9**：**多 agent 对比首个正式版本**。所有问题维度和评测指标齐备后，Claude Code / Codex / 后续 agent 的对比数据才具备可比性。
- **v0.10+**：横向扩框架（TorchAudio、TorchVision、JAX 或其他），验证 OpBench 方法论不 lock-in 到 PyTorch。
