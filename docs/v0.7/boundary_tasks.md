# OpBench v0.7 P3 Boundary Task 报告

日期：2026-07-27

状态：Passed

## 1. 结论

P3 从 10 条真实 PyTorch 候选中制作并验证了 6 条 Boundary Task，覆盖
B1–B5 全部子类。每条任务都完成了 Compatibility、Baseline Reproduced、
Gold Resolved、P2P Kept、人工复核和 8 阶段 Factory Admission，最终状态均为
`verified`。

本阶段没有为了达到数量目标降低 Admission 标准。自动筛选结果为
6 accepted、2 deferred、2 rejected；人工复核进一步把已被上游 revert 的
#147433 从 deferred 判为 rejected，#127448 保持 deferred，不把两者补入正式任务。

## 2. 候选漏斗

| PR | 自动结果 | 最终 P3 结果 | 原因或用途 |
| ---: | --- | --- | --- |
| 143792 | accepted | verified B1 | zero-size `addmv` decomposition |
| 117065 | accepted | verified B2 | 0-d index normalization |
| 126461 | accepted | verified B2 | rank-0 cumulative lowering |
| 147352 | accepted | verified B3 | checked storage-offset arithmetic |
| 118762 | accepted | verified B4 | default `dim` endpoint |
| 139751 | accepted | verified B5 | non-divisible Y-grid mask |
| 127448 | deferred | deferred | `change.large_diff_requires_review`，本阶段不扩范围 |
| 147433 | deferred | rejected | `review.upstream_reverted` |
| 143461 | rejected | rejected | `taxonomy.not_boundary` |
| 139502 | rejected | rejected | `test.missing_executable_delta` |

筛选输入、自动 Decision、人工拒绝和索引均保存在
`factory/v0.7/p3/`。自动筛选使用同一个固定时间戳重建时逐字节一致。

## 3. Verified Task 与 Admission

| Task | 子类 | Runtime 策略 | Baseline F2P/P2P | Gold F2P/P2P | 耗时 Baseline/Gold |
| --- | :---: | --- | --- | --- | ---: |
| `pytorch__143792__addmv_empty_matrix` | B1 | exact CPU full-source build | 0/1、1/1 | 1/1、1/1 | 780.653s / 768.724s |
| `pytorch__117065__index_copy_zero_dim` | B2 | torch 2.2.0+cpu overlay | 0/1、1/1 | 1/1、1/1 | 70.104s / 70.887s |
| `pytorch__126461__cummin_rank_zero` | B2 | torch 2.4.0+cpu overlay | 0/1、1/1 | 1/1、1/1 | 87.227s / 93.528s |
| `pytorch__147352__storage_offset_overflow` | B3 | exact CPU full-source build | 0/1、1/1 | 1/1、1/1 | 667.766s / 790.122s |
| `pytorch__118762__weight_norm_default_dim` | B4 | torch 2.3.0+cpu overlay | 0/1、1/1 | 1/1、1/1 | 80.870s / 80.164s |
| `pytorch__139751__triton_ygrid_mask` | B5 | torch 2.6.0+cu124 overlay | 0/1、1/1 | 1/1、1/1 | 88.690s / 90.966s |

六条任务合计 36/36 Compatibility checks 通过。Baseline 全部稳定复现目标
F2P 且保留 P2P；Gold 在完全相同 selector、Source 和 Environment identity 下
通过全部 F2P/P2P。Admission evidence 的 replay hash 均与当前 task bundle
重新计算结果一致。

每条任务都有独立的 `factory/review.json`、8 条不可变
`factory/chain/*.json` 和最终 `factory/admission.json`。链固定为：

```text
discovered → screened → bundled → preflight_passed
→ baseline_reproduced → gold_resolved → reviewed → verified
```

## 4. Runtime、Source 与 Digest

以下 SHA-256 都来自已提交的 Registry 或 Compatibility Evidence；私有主机路径
和原始命令日志不属于公开证据。

| Task | Source snapshot SHA-256 | Runtime artifact SHA-256 | Image ID |
| --- | --- | --- | --- |
| 143792 | `882123df39cc2b6dae98116a9c70f89540ae4069da3476156768fb6a29b2d810` | `0718d576a22aec3c4fa8254157eb69f5a8c66ad9a8e6c5f43d802c4853b56140` | `8288840b43b1f770a34cb1fcbc3429eadd0375bebe23baee18d7fcc10e6dadec` |
| 117065 | `86892fac0283784f8629258f659301291c7fa5b11788fe77414b11076c9a996e` | `2a8ff4440c1f024ad7982018c378470d2ae0a72f2bc269a22b1a677e09bdd3b1` | `9f529f436111a2b69dc1ad25df1f8f180023e030fb86f5c5808b04f3f020ba93` |
| 126461 | `3acabd5f692c8558338a2e4e698df0f541bff306f3b60cb81c422efeb863fa57` | `14a7a8b595347dddca594f9e448b93ce68ce4f871acbd32cf04bda7c03664c0c` | `e221898f5d6816e56297a8b7dc715b1fa307883b5c3861b4a5de94c13002ec49` |
| 147352 | `f82c557a0fd58629a1f093cd7fd1179747c5b24bf305b8e085ba96e83a1b0080` | `c4d4c2b7a1bbbd439dcf3e341ebb979f77152f6f2de4a4fafc14c18a90deb944` | `8288840b43b1f770a34cb1fcbc3429eadd0375bebe23baee18d7fcc10e6dadec` |
| 118762 | `cae5af9b60728275705bfd24df375c82acce658d02bf6186b07c12ffb57b8951` | `97a38b25ee0e3d020691e7846efbca62a3d8a57645c027dcb5ba0adfec36fe55` | `c1078f123ab2ea0e5ad89a35e456eacda203f98bfbbfd810a27a654d4a80a423` |
| 139751 | `712e36ae594e3724ae3465cc22fe93cef10418b0c4811b39fe7a2f6f71d0beeb` | `d4c3e9a8d31a7c0fcbb9da17c31a1917e1fac26c566a4cfbd8c9568ad7cade79` | `6ef9ad46672d4c4bbf565441b2fe0cdc4c4e1709a0f8c7179a825281a667010f` |

143792 与 147352 的 runtime artifact 是 exact Base Commit source-build
产物，不是 wheel。两者冻结了 gcc 11.4.0、cmake 3.22.1、ccache 4.5.1、
`BUILD_TEST=0`、`USE_CUDA=0` 和完整 build flag 列表。其余四条任务使用官方
wheel；139751 的 CUDA 12.4 runtime 在 sm_70 设备上完成兼容性与 Admission。

## 5. 低内存边界设计

147352 不申请超大 storage。F2P 使用小 tensor 和极端整数触发
`storage_offset * itemsize` 的 checked arithmetic，Gold 与真实
`Resize.h` 修复路径相同；P2P 覆盖正常 view 和 zero-size offset 语义。

139751 不构造上游回归中的超大 CUDA tensor。F2P 直接调用生产代码里的
`_has_constant_mask` 判定，构造 non-divisible Y-grid 的低内存 range-tree
surrogate；人工 review 明确确认它命中同一个 predicate 和同一个 Gold 修改路径。
P2P 覆盖 X grid、未超限 Y grid 和已有 Z grid。

## 6. 实施中发现并关闭的平台问题

- full-source build 成功后向隔离 Python 写入 workspace `.pth`，保证
  `python -I` 仍从刚构建的 exact source 加载 torch；
- full-source rsync 使用 partial directory，并把同步 timeout 与 build timeout
  对齐，避免 300 秒默认测试 timeout 截断大 snapshot；
- Factory promotion 通过 Environment/Source Registry 解析 task，避免把稀疏
  task override 错当成 `local` 空镜像；
- Admission 的浮点 `duration_sec` 在进入规范 Factory hash 前转换为整数
  `duration_ms`；
- probe 对 observation 数量显式 fail closed，并兼容 Python 3.9；
- draft task 的 promotion note 使用 `Verified by`，只有 deprecated 恢复继续使用
  `Restored by`。

## 7. 日志与公开证据策略

失败的 wheel ladder probe、完整构建输出、SSH/容器命令和 Admission 原始日志只
保存在 ignored `runs/` 树中。提交内容只包含任务定义、Registry identity、
Compatibility 摘要、稳定 Admission evidence、人工 review 和内容寻址 Factory
chain；不包含私有主机地址、密钥、绝对 source path 或 Agent 私有数据。

## 8. 验证

P3 聚焦验收通过 90 项核心/证据测试；最终 Factory/P3 收口组合另通过 53 项。
逐条校验了 6 个 Task manifest、Compatibility、Admission replay hash、Factory
chain、两个 Registry 和既有 17-task v0.5 verified Dataset。完整仓库回归通过
818/818 项，unittest 记录耗时 1174.098 秒（约 19 分 34 秒）。

P3 结论为 **Passed**。正式 `pytorch_v0.7`、Boundary/Precision Slice Freeze
以及真实 Codex 3-repeat Validation Cohort 属于 P4，本阶段没有提前发布 Dataset，
也没有形成跨 Agent 排名或反馈因果结论。
