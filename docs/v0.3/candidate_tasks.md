# OpBench v0.3 Task Candidate Pool

本文档记录由 Codex 筛选出的 10 条 PyTorch PR 候选任务。

## 候选列表

### 1. EmbeddingBag 2D offset bug (Multi-file)
- **PR**: https://github.com/pytorch/pytorch/pull/168159
- **Issue**: https://github.com/pytorch/pytorch/issues/167974
- **Component**: torch.nn.functional + torch.nn.modules.sparse
- **Files**: torch/nn/functional.py, torch/nn/modules/sparse.py
- **Base commit**: 6707dc8e444de405401f2e36e2868a19bb7ba43a
- **Patch size**: ~12 lines
- **Bug type**: shape-offset-inference
- **Test**: test/nn/test_embedding.py
- **Priority**: HIGH (multi-file, small patch, clear semantics)

### 2. nn argument length validation (Multi-file)
- **PR**: https://github.com/pytorch/pytorch/pull/162340
- **Issue**: https://github.com/pytorch/pytorch/issues/162327  
- **Component**: torch.nn.modules.conv + torch.nn.modules.utils
- **Files**: torch/nn/modules/conv.py, torch/nn/modules/utils.py
- **Base commit**: fefc406a3d0d90db0f808419fb88045f90b213cd
- **Patch size**: ~17 lines
- **Bug type**: parameter-validation
- **Test**: test/nn/test_pooling.py, test/test_nn.py
- **Priority**: HIGH (multi-file, validation logic, CPU pooling tests)

### 3. lr_scheduler deepcopy bug (Multi-file)
- **PR**: https://github.com/pytorch/pytorch/pull/127190
- **Issue**: https://github.com/pytorch/pytorch/issues/126854
- **Component**: torch.optim.lr_scheduler + torch.optim.swa_utils
- **Files**: torch/optim/lr_scheduler.py, torch/optim/swa_utils.py
- **Base commit**: d8d0bf264a736c7fb3cd17799a1c1aba4addf8d9
- **Patch size**: ~85 lines
- **Bug type**: state-management
- **Test**: test/optim/test_lrscheduler.py
- **Priority**: MEDIUM (multi-file, larger patch, covers optim)

### 4. DataLoader Subset dispatch bug
- **PR**: https://github.com/pytorch/pytorch/pull/163961
- **Issue**: https://github.com/pytorch/pytorch/issues/163184
- **Component**: torch.utils.data.dataset
- **Files**: torch/utils/data/dataset.py
- **Base commit**: 618efe837d71c179a96dc851ca08ebd9b10cae1f
- **Patch size**: ~30 lines
- **Bug type**: dispatch-logic
- **Test**: test/test_dataloader.py
- **Priority**: MEDIUM (utils.data coverage, single-file)

### 5. autograd unused gradient tracking bug
- **PR**: https://github.com/pytorch/pytorch/pull/168295
- **Issue**: https://github.com/pytorch/pytorch/issues/168059
- **Component**: torch.autograd
- **Files**: torch/autograd/__init__.py
- **Base commit**: d4de871adfac825e12bae9068e1c8433bd58455d
- **Patch size**: ~2 lines
- **Bug type**: autograd-state-management
- **Test**: test/test_autograd.py
- **Priority**: HIGH (tiny fix, clear semantics, autograd coverage)

### 6. LBFGS wolfe max iteration bug
- **PR**: https://github.com/pytorch/pytorch/pull/161488
- **Issue**: https://github.com/pytorch/pytorch/issues/91581
- **Component**: torch.optim.lbfgs
- **Files**: torch/optim/lbfgs.py
- **Base commit**: 6926710adf697e9d2160d43c4a96212dd27ceae0
- **Patch size**: ~9 lines
- **Bug type**: optimizer-control-flow
- **Test**: test/test_optim.py
- **Priority**: MEDIUM (small fix, optimizer coverage)

### 7. lr_scheduler last_epoch step bug
- **PR**: https://github.com/pytorch/pytorch/pull/149312
- **Issue**: https://github.com/pytorch/pytorch/issues/102261
- **Component**: torch.optim.lr_scheduler
- **Files**: torch/optim/lr_scheduler.py
- **Base commit**: 423fc671e9914f6a0ee567bbe47d56afb8cbe82b
- **Patch size**: ~28 lines
- **Bug type**: state-management
- **Test**: test/optim/test_lrscheduler.py
- **Priority**: MEDIUM (checkpoint/resume bug, deterministic)

### 8. autograd.backward inputs validation bug
- **PR**: https://github.com/pytorch/pytorch/pull/150975
- **Issue**: https://github.com/pytorch/pytorch/issues/150883
- **Component**: torch.autograd
- **Files**: torch/autograd/__init__.py
- **Base commit**: 6f9ffaa9916c02fa2aaae453db579a942b354708
- **Patch size**: ~21 lines
- **Bug type**: parameter-validation
- **Test**: test/test_autograd.py
- **Priority**: MEDIUM (autograd API validation)

### 9. nn.Module.set_submodule bug
- **PR**: https://github.com/pytorch/pytorch/pull/143455
- **Issue**: https://github.com/pytorch/pytorch/issues/143441
- **Component**: torch.nn.modules.module
- **Files**: torch/nn/modules/module.py
- **Base commit**: e3c4d1b7d6ea848f6b5558edf67e66dc28243641
- **Patch size**: ~76 lines
- **Bug type**: module-state-management
- **Test**: test/test_nn.py
- **Priority**: LOW (larger patch, edge case)

### 10. load_state_dict key prefix matching bug
- **PR**: https://github.com/pytorch/pytorch/pull/124385
- **Issue**: https://github.com/pytorch/pytorch/issues/123510
- **Component**: torch.nn.modules.module
- **Files**: torch/nn/modules/module.py
- **Base commit**: afa78ad08cc748c25e7e82cec02cec4c97c7d3af
- **Patch size**: ~11 lines
- **Bug type**: serialization-state-dict
- **Test**: test/nn/test_load_state_dict.py
- **Priority**: HIGH (serialization, small patch, dedicated test file)

## 推荐选择策略

为了达到 v0.3 的 10 条 verified task 目标，建议优先构建以下 7 条（加上现有 3 条 = 10 条）：

1. **#1 EmbeddingBag** — multi-file, 小 patch, 覆盖 sparse module
2. **#2 nn argument validation** — multi-file, 覆盖 conv/utils
3. **#5 autograd unused gradient** — 极小修复, autograd 覆盖
4. **#10 load_state_dict** — serialization 覆盖, 小 patch
5. **#4 DataLoader Subset** — utils.data 覆盖
6. **#6 LBFGS** — optimizer 覆盖
7. **#8 autograd.backward validation** — autograd API 覆盖

备选（如果上述有 admission 失败）：
- #3 lr_scheduler deepcopy (较大 patch 但 multi-file)
- #7 lr_scheduler last_epoch
- #9 set_submodule (patch 较大)

## 下一步操作

1. 为每个 base commit 准备 source snapshot
2. 从 PR diff 提取 gold.patch
3. 从 PR test 改动提取或手写 hidden_test.patch
4. 构建完整 task bundle (task.json, issue.md, patches)
5. 运行 admission pipeline
6. 筛选出 7-10 条 verified task 进入 datasets/pytorch_v0.3/
