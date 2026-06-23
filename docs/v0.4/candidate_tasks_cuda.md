# OpBench v0.4 CUDA Task Candidate Pool

本文档记录 v0.4 CUDA task 的筛选策略和候选 PR。需要 Codex 协助补充 PR URL。

## 数据集组成（目标 8-10 条）

按难度分层：

| Tier | Count | 修复范围 | 评测时间 |
| --- | --- | --- | --- |
| `cuda_python_overlay` | 6-8 条 | Python only (.py) | ~5 分钟/task |
| `cuda_kernel_build` | 2 条 | 允许 .cpp/.cu/.h | ~30 分钟/task（含 build） |

## 筛选标准（cuda_python_overlay）

### 必须满足
1. PyTorch 仓库的 **merged PR**
2. 修复一个 **CUDA-related bug**（CPU 上不复现或行为不同）
3. **base commit 兼容 torch 2.6 + CUDA 12.4**
4. **gold patch 仅 Python 文件**（torch/ 下的 .py）
5. PR 中**包含对应的测试改动**
6. 修复代码量 **20-150 行**

### 优先级
- bug 类型多样化（dtype 精度 / device dispatch / CUDA autograd / stream / memory format）
- 修改 `torch/nn/functional.py` / `torch/nn/modules/` / `torch/autograd/` / `torch/cuda/`
- 2024-2025 年的 PR

## 筛选标准（cuda_kernel_build）

### 必须满足
1. PyTorch 仓库的 **merged PR**
2. 修复必须涉及 **C++/CUDA kernel** 文件（`.cpp` / `.cu` / `.h`）
3. **base commit 兼容 torch 2.6 + CUDA 12.4**
4. PR 包含对应的测试
5. 修复代码量 **30-200 行**（kernel 修改往往比 Python 多）
6. 必须能在 **A10/A100 (Ampere sm_80)** 上复现和验证（不依赖 H100/FP8）

### 优先级
- 修改 `aten/src/ATen/native/cuda/` 下的 kernel
- 修改 `torch/csrc/` 下的 binding
- bug 类型：kernel 边界条件、reduction 精度、index 越界、shared memory 错误
- 避免修改 build system / dispatcher 框架性代码

### 禁止
- 修改超过 1-2 个 kernel 文件（agent 解决不了）
- 涉及 cuDNN / cuBLAS 库版本相关的 bug
- 涉及多卡通信（NCCL）

## 给 Codex 的筛选提示词

见上一轮对话提供的提示词。Codex 应输出：
- 6-8 条 cuda_python_overlay 候选
- 2 条 cuda_kernel_build 候选

每条按以下格式：

```json
{
  "pr_url": "https://github.com/pytorch/pytorch/pull/XXXXX",
  "issue_url": "https://github.com/pytorch/pytorch/issues/XXXXX",
  "title": "...",
  "tier": "cuda_python_overlay | cuda_kernel_build",
  "component": "torch.nn.functional / aten/src/ATen/native/cuda 等",
  "files_changed": ["..."],
  "test_files_changed": ["..."],
  "base_commit": "完整 SHA",
  "merge_commit": "完整 SHA",
  "patch_lines": 数字,
  "kernel_files": ["..."]  // 仅 cuda_kernel_build 需要
  "bug_type": "...",
  "min_gpu_arch": "sm_80",
  "why_good": "..."
}
```

## 候选 PR 列表

### 已选定（7 条）

#### cuda_python_overlay (5 条)

| # | PR | Title | Component | Patch | Multi-file | Bug Type |
|---|----|----|----|------:|---|---|
| 1 | [#132835](https://github.com/pytorch/pytorch/pull/132835) | NJT SDPA manual autocast | torch.nested._internal.sdpa | 46 | No | dtype-precision |
| 2 | [#147786](https://github.com/pytorch/pytorch/pull/147786) | FakeTensor load with correct device | torch.serialization | 26 | Yes (3) | device-dispatch |
| 3 | [#131858](https://github.com/pytorch/pytorch/pull/131858) | pin_memory device API regression | torch.utils.data | 74 | Yes (6) | device-dispatch |
| 4 | [#133729](https://github.com/pytorch/pytorch/pull/133729) | DeviceContext bug | torch.utils._device | 38 | Yes (2) | device-dispatch |
| 5 | [#132616](https://github.com/pytorch/pytorch/pull/132616) | cuda mem_get_info accepts device str | torch.cuda.memory | 7 | No | parameter-validation |

#### cuda_kernel_build (2 条)

| # | PR | Title | Component | Patch | Multi-file | Bug Type |
|---|----|----|----|------:|---|---|
| 6 | [#141820](https://github.com/pytorch/pytorch/pull/141820) | torch.lerp CPU scalar + CUDA tensor | aten.native.cuda.lerp | 49 | Yes (2) | device-consistency |
| 7 | [#143264](https://github.com/pytorch/pytorch/pull/143264) | addcmul CPU scalar | aten.native.cuda.pointwise | 98 | Yes (2) | device-consistency |

### 备选

- [#139409](https://github.com/pytorch/pytorch/pull/139409) — torch.bool sort CUDA (25 行 kernel)，如果上面某条 admission 失败可替换
- [#135140](https://github.com/pytorch/pytorch/pull/135140) — cross-device scalar refs (6 行)，太小但语义清晰
- [#141065](https://github.com/pytorch/pytorch/pull/141065) — nested autocast HOP (12 行)，导出层逻辑复杂

### 选择说明

- 选 5 个 cuda_python_overlay 而不是 6 个：避免引入太多多文件 task 增加 admission 难度
- PR #132616 虽然只有 7 行，但覆盖了 `torch.cuda.*` 模块，组件多样性值得保留
- 预计达到 v0.4 总数：v0.3 (10) + v0.4 新增 (7) = **17 条 verified task**



## 构建流程

每条 task 的构建流程：

### cuda_python_overlay（同 v0.3 流程）
1. 提取 `gold.patch`（torch/ 下 Python 改动）
2. 提取 `hidden_test.patch`（test/ 下改动）
3. 创建 source snapshot（sparse checkout `torch/` + `test/`）
4. 生成 task.json（`environment_ref: pytorch-cuda-torch2.6.0-py311-cu124`）

### cuda_kernel_build（新流程）
1. 提取完整 `gold.patch`（包含 .py + .cpp + .cu + .h）
2. 提取 `hidden_test.patch`
3. 创建 source snapshot：
   - sparse checkout `torch/` + `aten/` + `c10/` + `test/` + `setup.py` + `tools/`
   - **必须 init submodules**（third_party/cutlass、third_party/eigen 等）
   - 完整 source snapshot 体积 ~2-3GB
4. 生成 task.json：
   - `environment_ref: pytorch-cuda-devel-torch2.6.0-py311-cu124`
   - `runtime_tier: cuda_kernel_build`
   - `source_loading.mode: inplace_build`
   - `patch_scope.allowed_paths` 包含 .cpp/.cu/.h 路径

## 测试稳定性注意

- 优先选 **Ampere (sm_80)** 都能跑的 task
- 使用 tolerance-aware 断言（`assertEqual(rtol=, atol=)`）
- 避免依赖 FP8 tensor core / TMA 等 Hopper-only 特性
- kernel-build task 要确认 `BUILD_TEST=0` 跳过 PyTorch 自带 C++ test 编译

## 收敛 PyTorch 版本

为减少维护成本，v0.4 维护两个 CUDA image：

```
op-bench/pytorch-cuda:torch2.6.0-cu124-py311           ← cuda_python_overlay
op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311     ← cuda_kernel_build
```

如果某个 PR 必须用其他 torch 版本才能复现，**放弃这个 PR**。

