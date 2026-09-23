# 运行记录

按 **版本 → 用途 → 批次 → 单次运行** 保存实验。这里存放本地原始证据；版本结论与可分享的摘要放在 `docs/v0.x/`，并注明对应批次。

```text
runs/
├── v0.8/
│   └── validation/
│       └── 20260922-internal-cleanup/
│           ├── README.md             # 本批目的、执行方式、结论与证据位置
│           ├── core.log              # 全套工程测试
│           └── installed-r2/         # 最终评分文件引用格式的安装联调
├── v0.10/
│   └── experiments/
│       └── YYYYMMDD-model-comparison/
│           ├── README.md
│           ├── run-r1/               # opbench run 的完整输出
│           └── replay-r1/            # 同补丁独立重评，保留原运行
├── cache/                            # 可复用构建等，不作为实验成绩
└── archive/                          # 旧版记录与迁移索引
```

图中 v0.10 是后续目录约定，目前没有正式多模型实验。已有真实 Codex 小任务也属于接入验证，不作为排名实验。

## 保存新一轮记录

1. 先选用途：`validation` 用于工程测试、评分控制、安装及模型联调；`experiments` 用于有固定数据、模型、工具与预算的正式评测。
2. 每个工作批次建一个目录，建议命名 `YYYYMMDD-topic`；批次内不同物理运行使用 `run-r1`、`run-r2`、`controls-r1` 等明确名称。文档中的 `quickstart-r1` 是可直接复制的演示批次名。
3. 将目录传给现有 `--output`。`run` 会保存计划、尝试、补丁和独立评分，不需要另造一套配置清单或实验注册系统。
4. 同一计划中断后用原目录 `--resume`，继续准备和未启动的任务；已启动的模型尝试不会自动重跑。改变配置或主动重新求解时新建 `run-r2`，在批次说明中写清原因。失败和中断记录保留，不覆盖后只留下成功结果。
5. 批次结束后写简短 `README.md`：目的、配置/命令位置、实际运行、结论、失败与限制、关键结果路径。已有 `plan.json` 或 `summary.json` 的详细字段无需再抄一份。

例如，先按[模型接入指南](../docs/v0.8/agent_integration.md#实验配置与运行)准备 `experiment.json`：

```bash
opbench run --experiment experiment.json \
  --output runs/v0.8/validation/quickstart-r1/model-run
opbench report --run runs/v0.8/validation/quickstart-r1/model-run
```

`report.json` 是派生汇总。评分详情只保存在对应评分目录的 `result.json`；变体父记录和 `controls.json` 通过 `evaluation_path` 关联子评分，不再保存完整副本。

## 已有批次

下面的位置相对仓库根目录，原始数据只在保存过它们的机器上存在。

| 批次 | 内容 | 位置 |
| --- | --- | --- |
| 9 月 7 日初始验证 | 早期数值、C++ 和原生框架检查，保留当时口径 | `runs/v0.8/validation/20260907-initial/` |
| 9 月 9 日验证 | 核心、完整框架/GPU、早期 Codex 接入 | `runs/v0.8/validation/20260909-{core,remote,codex}/` |
| 9 月 14 日固定循环联调 | Codex gpt-5.5，小任务与中断记录 | `runs/v0.8/validation/20260914-controlled/` |
| 9 月 21 日架构整理 | 核心回归、安装联调 | `runs/v0.8/validation/20260921-architecture/` |
| 9 月 22 日前两轮复核 | 简化、模块边界与输出路径检查 | `runs/v0.8/validation/20260922-{simplification,deep-audit}/` |
| 9 月 22 日内部精简 | 评分引用格式、完整回归、安装联调与目录整理 | `runs/v0.8/validation/20260922-internal-cleanup/` |
| 9 月 22 日测试夹具收敛 | 轻量数值镜像、三组控制、核心回归及安装验证 | `runs/v0.8/validation/20260922-fixture-consolidation/` |

当前验证结论见[验证报告](../docs/v0.8/validation_report.md)，更早记录见[实验归档索引](archive/README.md)和[版本历史](../docs/history/README.md)。

## 归档与版本管理

原始运行目录可能含私有测试、完整源码和模型轨迹，默认不进入 Git。仓库只保留本页、归档索引及四份已经恢复的历史报告；当前版本摘要继续随 `docs/v0.8/` 保存。需要迁移机器时，同步完整批次目录。

本次整理仅移动原始目录，未改写日志或评分。v0.8 的两个旧目录名通过本地 `.legacy-paths/` 桥接到新位置，供历史证据定位；新运行均使用上面的正式路径。旧记录中的构建路径和环境身份属于当时条件，归档不保证可直接继续运行或复用旧基线。
