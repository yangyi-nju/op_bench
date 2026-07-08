# v0.5 精度类 PR 检索策略

本文档给外部 agent 用于筛选精度维度 PR 候选池。目标：每个子类（P1–P5）产出 3-5 条候选 PR，最终 6-8 条 admissible。

## 5 个子类的检索脚本

所有检索都限定：
- 仓库：`pytorch/pytorch`
- 状态：`is:merged`
- base commit 时间窗：**merged in 2024-01-01 到 2025-04-30**（避免 post-2.6 nightly wheel 不兼容，v0.4 已验证的坑）

### P1 数值累积误差

reduction 类（sum/mean/norm/logsumexp）在长序列或低精度下累积浮点误差。修复常见形式：改 accumulator dtype、加 Kahan 求和、切 log-space。

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(accumulator OR "Kahan" OR "logsumexp" OR "log_sum_exp" OR "sum precision") is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files
```

关键词：`accumulator`, `Kahan`, `pairwise sum`, `logsumexp`, `numerical stability`, `precision loss`, `float16 sum`, `bfloat16 mean`, `long reduction`.

### P2 dtype 转换损失

隐式 cast（half↔float / int↔float）在中间步骤丢失精度。修复：修 upcast 逻辑、修 `dtype=` 参数传递、修 output dtype 推断。

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(upcast OR downcast OR "output dtype" OR "dtype promotion" OR "intermediate dtype") is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files
```

关键词：`upcast`, `downcast`, `promotion`, `intermediate dtype`, `output dtype`, `type_as`, `to(dtype=)`, `promote_types`.

### P3 混合精度 (autocast) 不一致

`torch.autocast` 场景下某算子 output dtype 与 reference 不一致。修复：autocast wrap list 或 `@autocast_custom_fwd` 装饰。

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(autocast OR "mixed precision" OR autocast_custom_fwd) is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files
```

关键词：`autocast`, `mixed precision`, `amp`, `custom_fwd`, `custom_bwd`, `autocast_dtype`.

### P4 数值不稳定

log/exp/softmax/sigmoid 在极端输入下 NaN/Inf。修复：log-sum-exp trick、clamp、stable variant。

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(NaN OR Inf OR "log_softmax" OR sigmoid OR "numerical instability" OR "log-sum-exp") is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files
```

关键词：`NaN`, `Inf`, `overflow`, `underflow`, `numerical instability`, `log-sum-exp`, `stable`, `clamp`.

### P5 Kernel 精度 bug

CUDA kernel 里的 shared memory 累加、warp reduce、tail loop 处理精度问题。修改 `.cu` / `.cuh` 文件。**注意：仅在 sm_70/sm_80 可复现**，避免 H100/FP8 专属特性。

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(numerical OR precision OR "warp reduce" OR "tail loop") path:aten/src/ATen/native/cuda is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files
```

补充搜索：`shared memory` + `float`、`reduce` + `precision`、`ilpReduce` / `warpReduce`。

### 通用 label 搜索

PyTorch 内部对精度类问题有几个高价值 label（各 label 通常关联 20-50 条 PR）：

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 300 \
  --search 'label:"topic: numerical" is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels
```

其他有用 label：`module: precision`, `module: numerical`, `topic: numerical stability`.

## 硬性过滤条件（hard filter）

候选 PR 进入 admission 前必须通过全部 6 条：

1. **状态**：`state=merged`，且**不能是 revert 或 reland**（PR title 含 `revert` / `reland` 直接排除）。
2. **base commit 时间**：merged 在 2024-01-01 到 2025-04-30（对应 torch 2.6 前后可稳定 checkout 的窗口）。
3. **修改文件数 ≤ 3**：agent 单次修复能力上限。可以是 1 py + 1 test，或 1 cu + 1 test。
4. **修改行数 20–200**：太短语义不明确，太长是重构不是 fix。计非空非注释行。
5. **测试改动必须存在**且是以下形式之一：
   - 新增测试文件或用例
   - 已有测试中 tolerance 变化（`rtol=` / `atol=` / `assertClose` 参数改变）
   - 已有测试 assert 类型变化（例如 `assertEqual` → `assertAlmostEqual`）
6. **PR 描述判断**：**排除** "Add support for X" / "Enable X on Y" 这类 feature-add PR；**保留** "Fix X" / "Correct X" / "Handle X" 这类 correctness fix。v0.4 已验证 add-support 类 PR 的 baseline 通常已通过（走隐式 fallback），无法产生 fail_to_pass。

## 输出格式（给下游 admission）

每条通过硬过滤的候选，输出如下 JSON：

```json
{
  "pr_url": "https://github.com/pytorch/pytorch/pull/XXXXX",
  "issue_url": "https://github.com/pytorch/pytorch/issues/XXXXX",
  "title": "Fix log_softmax accumulation for float16",
  "subclass": "P1",
  "problem_dimension": "precision",
  "component": "torch.nn.functional / aten/src/ATen/native/cuda",
  "files_changed": ["aten/src/ATen/native/cuda/SoftMax.cu", "test/test_nn.py"],
  "test_files_changed": ["test/test_nn.py"],
  "base_commit": "完整 SHA",
  "merge_commit": "完整 SHA",
  "patch_lines_source": 45,
  "patch_lines_test": 12,
  "requires_kernel_build": true,
  "min_gpu_arch": "sm_70",
  "bug_pattern": "one-line summary of the bug",
  "why_good": "为什么它是好的 admission 候选"
}
```

## 自动过滤脚本

`scripts/screen_candidates.py` 接受候选清单 JSON（数组），运行前 6 条硬过滤，输出通过者：

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 200 \
  --search '(autocast OR "mixed precision") is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,labels,files > candidates_p3.json

PYTHONPATH=src python3 scripts/screen_candidates.py \
  --input candidates_p3.json \
  --subclass P3 \
  --output candidates_p3_screened.json
```

脚本行为：
- 每条候选自动跑 `gh pr view --json files,commits,body`, 拉取 patch 信息
- 根据规则 1-6 打分/过滤
- 输出通过的候选列表 + 每条 rejected 的原因（便于人工复审）

## 优先级建议

- **优先做 P1 / P5**：这两类的 bug 断言最直接（`assertClose` 或 log_softmax 求和），admission 通过率最高。
- **P3 autocast** 需要 CUDA 环境，但 PR 密度高，值得投入。
- **P2 / P4** 稍难：dtype 转换和数值稳定的修复常常涉及多文件，先每类 3 候选凑数。

## 交付形式（给 Codex 或其他 agent）

- 每个子类一份 candidate 列表（JSON）
- 汇总一份跨子类的入池 PR 表（Markdown）
- Reject 名单 + 每条原因（人工复审用）

## 边界与例外

- 不选带 `topic: performance` 标签的 PR（v0.5 不做性能维度）
- 不选修改超过 5 个文件的 PR（agent 单次能力不够）
- 不选依赖 `nn.functional.scaled_dot_product_attention` FP8 / flash-attn 3 的 PR（H100-only）
- 如果 P5 kernel 类只找到 1 条通过 admission 的，允许放宽到 P1 补足（不强求每类都 ≥1）
