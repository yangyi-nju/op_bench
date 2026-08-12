# OpBench v0.7 P1 Candidate Search

日期：2026-07-26

状态：P1 离线协议已冻结

## 范围

P1 只验证本地、确定性的 Candidate → Decision 工作流。输入为
`fixtures/factory/v0.7/candidates.json` 中的合成捕获，不访问 GitHub、
PyTorch mirror、Docker、SSH、CUDA、Agent 或远程 Runtime，也不生成正式
`pytorch_v0.7` Dataset。

候选捕获先规范化为 `CandidateRecord`，再由固定规则集生成不可变
`DecisionRecord`。公开 Artifact 只保存 provenance、规范化筛选事实和内容
哈希；不包含凭据、Hidden Test 内容、Gold Patch 内容、目标句柄或主机私有
路径。

## Boundary Keyword Packs

| Pack ID | 子类 | 根因 |
| --- | --- | --- |
| `boundary-b1-v1` | B1 | empty / zero-size |
| `boundary-b2-v1` | B2 | scalar / degenerate shape |
| `boundary-b3-v1` | B3 | integer / size overflow |
| `boundary-b4-v1` | B4 | parameter endpoints |
| `boundary-b5-v1` | B5 | kernel launch / grid bounds |

匹配只执行大小写无关的字面短语规则，并优先应用 exclusion phrase。它只提出
候选子类，不证明根因，因此自动接受的 Decision 仍携带
`review.root_cause_required`。

## 固定筛选规则

稳定 author-date 窗口为
`2024-01-01T00:00:00Z..2025-04-30T23:59:59Z`。普通自动筛选阈值为最多
3 个文件、最多 200 条改动行；无可执行测试增量时，少于 20 条改动行视为
低于行为修复阈值。

规则按以下顺序独立产生 finding，最终再按
`reject → defer → warning`、reason code、rule ID 排序：

| Rule ID | Reason code | Severity | 条件 |
| --- | --- | --- | --- |
| `v07-revert-title` | `metadata.revert_or_reland` | reject | title 以 revert 或 reland 开头 |
| `v07-required-dates` | `metadata.missing_date` | defer | author/merge date 缺失 |
| `v07-stable-author-window` | `window.outside_stable` | reject | author date 在窗口外且无 Environment Freeze |
| `v07-stable-author-window` | `window.environment_freeze_exception` | warning | 窗口外但有显式 Environment Freeze |
| `v07-change-kind` | `change.non_bug_change` | reject | 纯 refactor、cleanup 或 feature |
| `v07-normal-diff-size` | `change.large_diff_requires_review` | defer | 超过 3 文件或 200 改动行 |
| `v07-minimum-behavioral-change` | `change.below_behavioral_threshold` | reject | 少于 20 行且无 changed/external test |
| `v07-executable-test-delta` | `test.missing_executable_delta` | reject | 无 changed test 且无 external-test 引用 |
| `v07-commit-identities` | `source.missing_commit_identity` | defer | base/merge commit 缺失 |
| `v07-source-availability` | `source.unavailable` | defer | Source Snapshot 不可用 |
| `v07-runtime-support` | `runtime.unsupported` | defer | 硬件或 Runtime 要求当前不支持 |
| `v07-boundary-taxonomy` | `taxonomy.not_boundary` | reject | 输入不是 Boundary 候选 |
| `v07-root-cause-review` | `review.root_cause_required` | warning | 尚未完成人工根因复核 |

Disposition 只由 severity 派生：存在 reject 即 `rejected`；否则存在 defer 即
`deferred`；其余为 `accepted`。warning 不会单独阻止自动接受。

## 离线 Fixture

运行：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python \
  scripts/factory_screen_candidates.py \
  --input fixtures/factory/v0.7/candidates.json \
  --output-dir /tmp/opbench-v07-screen \
  --created-at 2026-07-26T00:00:00Z
```

9 条合成候选的冻结结果为：

| Disposition | 数量 | 覆盖 |
| --- | ---: | --- |
| accepted | 5 | B1、B2、B3、B4、B5 各 1 |
| deferred | 2 | 大 diff 人工例外、缺失日期/提交 |
| rejected | 2 | 非 bug change、稳定窗口外 |

期望索引保存在
`fixtures/factory/v0.7/expected_decisions.json`。相同输入和
`--created-at` 必须生成逐字节一致的 Candidate、Decision 与
`screening_index.json`。

## 解释边界

Fixture 的 `accepted` 只表示通过自动 P1 筛选，不等于人工根因复核、不等于
Task Bundle 完成、不等于 Admission verified，也不能进入 Dataset Freeze。
候选只有在后续完成 Preflight、Baseline Failure、Gold Success、Human Review
和 Integrity 证据门后，才可形成 verified `FactoryAdmissionRecord`。
