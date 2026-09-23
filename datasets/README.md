# 数据集入口

当前可由 v0.8 CLI 读取的上游任务清单是
[`v0.8_candidates/dataset.json`](v0.8_candidates/dataset.json)。三道题均为迁移候选，
不是正式发布的数据集；执行前须准备各题声明的源码和镜像，详见[候选说明](v0.8_candidates/README.md)。

```bash
opbench datasets validate --dataset datasets/v0.8_candidates/dataset.json
```

结构校验只检查任务声明，实际评分检查使用 `check-task`，结果写入 `controls.json`。
分类扩充及完整选题、准入流程在 v0.9 实施。

引擎测试的小型任务统一放在 [`tests/fixtures/`](../tests/fixtures/README.md)，不计入这里的数据集。

`pytorch_mini/` 仅保留 [v0.2 原始数据集说明](pytorch_mini/README.zh-CN.md)，没有当前可执行清单。
旧题素材位于 [`tasks/`](../tasks/README.md)，旧版本完整资产从[历史记录](../docs/history/README.md)查阅。
