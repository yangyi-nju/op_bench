# 固定模型循环与 MCP 工具

修订于 2026-09-23。v0.8 第一阶段固定工具、提示和运行流程，更换模型进行比较。
它不接受任意 Agent 启动命令，也不让本地 Codex 直接接管任务工作区。

## Benchmark、Agent 与模型的职责

Benchmark 定义题目、执行环境、提交规则和独立评分；Agent 负责如何求解，包括
对话、规划与工具调用；模型为 Agent 提供推理和行动决策。这是职责划分，不需要
把项目拆成多个服务。当前 OpBench 的固定循环与 MCP 工具共同构成内置 Agent。

| 评测目标 | 接入方向 | 保持共同的条件 |
| --- | --- | --- |
| 第一阶段比较模型，当前已实现 | OpBench 固定循环通过模型客户端调用不同模型 API | 同一题目、工具、提示、循环与预算 |
| 后续比较 Agent 框架，尚未实现 | 实验运行器通过适配器启动框架，框架管理自己的求解循环 | 固定模型、任务及约定的工具、预算和信息边界 |

HTTP 模型通过 `base_url`、`api_key_env` 和 `model` 接入；模型服务只需
理解 API 消息和工具调用格式，不需要理解 OpBench 的数据集或评分实现。
新增不兼容的模型协议只修改客户端转换，不为每个模型重写工具或评分器。
独立的 `evaluate --task ... --patch ...` 已能评价外部产生的补丁；这只完成评分，
不代表外部框架的求解预算、工具权限和运行轨迹已纳入统一实验。

后续框架接入时，适配器负责把公开任务和约定工具交给框架，再把提交交回 OpBench；
OpBench 继续负责环境准备、外层预算约束、提交冻结和独立评分。框架不得获得私有
评分资产或在同一次求解中反复查询最终分数。框架间如何固定工具和预算需作为实验
条件明确记录；当前先保留这个职责边界，不增加空的框架注册表或通用插件层。

## 共同执行流程

1. 控制器读取任务和模型配置，先保存完整模型 × 任务计划。
2. 准备固定源码、依赖与可信基线；为每次尝试建立独立工作区。
3. 控制器向模型提供相同公开输入和 MCP 工具说明，模型按相同协议提出工具调用。
4. 工具验证参数与路径，在指定工作区执行读取、搜索、修改、构建和公开测试等动作。
5. 终止或到达预算后停止工具执行，捕获并冻结补丁。
6. 独立评估应用该补丁并重新构建，执行私有评分，保存结果供报告读取。

MCP 只有一套工作区工具实现，模型适配器不另写一套编辑或评分逻辑。模型客户端
只负责请求和响应，不拥有候选工作区，不接触 grader。受控实验的任务代码在资源受限、
断网的 Docker 会话中运行；控制器与容器的[信息边界](information_boundary.md)一致。
显式 `development` 可用于 local 调试，不具备容器文件和网络隔离。

固定工具只有以下十项；所有模型共享名称、参数和行为：

| 工具 | 用途 |
| --- | --- |
| `list_files`、`read_file`、`search` | 列出、读取和搜索公开工作区文件 |
| `write_file`、`replace_text`、`delete_file` | 新建、修改或删除源码和公开测试文件 |
| `build`、`run_public_tests` | 执行任务声明的构建或公开测试命令 |
| `diff` | 查看源码差异 |
| `finish` | 结束求解并提交当前源码状态 |

没有任意 shell 或 Git 执行工具。MCP 使用 `initialize`、`tools/list` 和
`tools/call`；固定循环通过本地序列化 JSON-RPC 调用同一服务实现，不需要额外
部署进程，也没有独立 stdio 服务入口。

当前工具协议为 `controlled-tools-2`，循环协议为 `opbench-fixed-tools-3`。工具定义
与执行共用参数校验，字符串不允许 NUL，最长 1000000 字符；具体字段仍遵守工具
声明的更小上限。一次回复包含多个调用时，先验证全部参数，再执行任何动作，
避免前面的写入生效后才发现后面的参数非法；`finish` 必须是该回复的唯一调用。
当前仅支持实际工具使用的平面字段。
工具或循环协议升级后，未完成的旧计划不能继续启动新求解，需要建立新计划；
计划与尝试显式记录两种版本。已经完成的旧尝试仍可读取与重算报告，保持原身份。

## 模型接入

HTTP 模型客户端和本地 Codex model-only 客户端都进入同一控制器循环。服务 URL、
实际模型、必要的模型参数与连接配置显式记录；凭据从运行环境取得，不写入任务
包、日志或公开实验文件。切换模型不改变工具定义、预算、题目或评分。

本地 Codex 用作现有登录下的最小联调传输，模型只返回受控调用或最终说明，由
OpBench 执行工具。Codex 自身的 shell、编辑、MCP、浏览和自主工具链不参与求解。
CLI 能否使用指定模型需要实际调用确认；下面的配置选择 `gpt-5.5` / `high`，不使用宿主
默认模型名代替实验声明。模型不可用时如实记录错误，不静默换成其他模型。
每次调用用临时 provider 参数选择 HTTPS，并继续正常认证和端点选择；不修改
用户 `config.toml`，不复制登录凭据。使用前需安装并登录 Codex，确认本地模型缓存包含
指定模型；若不在 PATH 中，将 `codex_binary` 改为实际路径。CLI 的上下文包装和
自报用量是该接入的实验条件，`max_completion_tokens` 不由此接口强制执行；
求解时间、轮数和上下文限制仍由共同循环管理。正式比较需固定可比的接入条件。

实验配置使用 `schema_version: 2`，声明 `dataset`、`models` 和共同的 `harness`。
`information_profile: "controlled"` 要求 Docker；local 调试须显式声明
`"development"`。`reuse_baseline` 只共享可信未修改基线，`seed` 固定模型 × 任务
尝试单元的打乱顺序。
模型配置必需 `model_id`、`backend`、`model`，不能用任意 `argv` 代替模型接入。

| 配置 | 含义 |
| --- | --- |
| `backend: "chat_completions"` | 控制器调用明确的 `base_url`，凭据由 `api_key_env` 指定的环境变量提供 |
| `backend: "codex_cli"` | 使用指定 `codex_binary` 和现有登录做 model-only 联调；`model` 显式选择模型 |
| `reasoning_effort` | 显式推理配置；服务是否支持由真实调用确认 |
| `harness.budget_sec` | 一道题的求解总时间 |
| `harness.max_turns` | 固定模型循环的轮数上限 |
| `harness.max_context_chars` | 每轮请求前检查序列化对话消息的字符数；不含工具定义和 CLI 包装，不是 token 或费用预算 |
| `harness.per_tool_timeout_sec` | 单次工具执行上限，同时受剩余总时间限制 |

默认共同配置为 900 秒、60 轮、200000 个上下文字符、每工具 120 秒。

## 实验配置与运行

先按[环境指南](remote_execution.md)准备所选任务的源码、镜像和计算资源。以下配置
选择当前三道上游候选，其中包含 CUDA 任务；它们尚未正式准入，执行结果属于验证。
将配置保存为仓库根目录的 `experiment.json`，按实际模型与预算调整：

```json
{
  "schema_version": 2,
  "dataset": "datasets/v0.8_candidates/dataset.json",
  "information_profile": "controlled",
  "reuse_baseline": true,
  "seed": 42,
  "harness": {"budget_sec": 900, "max_turns": 60, "per_tool_timeout_sec": 120},
  "models": [
    {"model_id": "gpt-5.5", "backend": "codex_cli", "model": "gpt-5.5",
     "reasoning_effort": "high", "codex_binary": "codex"}
  ]
}
```

```bash
opbench run --experiment experiment.json --output runs/v0.8/validation/quickstart-r1/model-run
opbench run --experiment experiment.json --output runs/v0.8/validation/quickstart-r1/model-run --resume
opbench report --run runs/v0.8/validation/quickstart-r1/model-run
```

`--resume` 继续同一配置和计划，只启动尚无 `attempt.json` 的尝试。已有尝试即使
失败也不重跑；残留的 `running` 记录标为 `interrupted`。重新求解或修改配置应使用
新输出目录。任务路径相对于数据集，数据集路径相对于实验文件。只做单题
验证时，在实验顶层设置 `"task_ids": ["所选 task_id"]`；控制器只为所选题目建立计划，
无需复制数据清单。省略 `task_ids` 则运行清单中的全部任务。

直接 HTTP 接入只需替换 `models` 中的模型项，其余循环和工具不变：

```json
{
  "model_id": "declared-model",
  "backend": "chat_completions",
  "model": "actual-provider-model",
  "base_url": "https://your-provider.example/v1",
  "api_key_env": "OPBENCH_MODEL_API_KEY"
}
```

替换为实际服务地址和模型名，通过指定的控制器环境变量提供凭据。模型客户端
始终在控制器中运行；任务容器保持断网。本轮没有独立网关、relay 或分布式调度。
`model_id` 是报告中的唯一标识，`model` 是服务端模型名称；`models` 可包含多项，
各项可使用不同的地址与密钥环境变量。`base_url` 填 API 前缀，程序会追加
`/chat/completions`，并以 Bearer 方式发送环境变量中的密钥。当前 HTTP 客户端要求
兼容 Chat Completions 工具调用，并接受 `max_completion_tokens`、
`parallel_tool_calls` 等请求字段；特定服务是否兼容仍需实际联调。
仅检查模型接入和引擎时，可使用[内部测试夹具](../../tests/fixtures/README.md)，
无需完整 PyTorch 构建；夹具结果不用于衡量模型能力。

## 输入、求解与提交

公开输入包括题面、指定基线、声明的公开命令和提交规则，由最初的模型请求提供，
并在尝试目录保存为 `task_input.json`。工作区 `.opbench/` 只保存公开的
`submission-policy.json`，不再重复放置题面副本。公开输入不含私有评分资产、
上游未来历史、其他尝试、参考补丁或模型凭据。允许模型通过固定工具探索和修改
公开源码，权限不由 Gold 的文件路径决定。

最终提交是相对初始源码的补丁，支持合法新增、修改和删除。通过 `write_file` 或
`replace_text` 显式写入的源码会登记提交意图；这些工具写入的合法新文件不会因
初始 Git 忽略规则而静默丢失。其他自动发现的新文件仍按初始忽略规则筛选。
模型无需执行 Git 暂存。初始生成产物、控制器元数据和声明的 runtime 子树仍不属于
源码提交。
修改公开测试仅影响自测，不改变最终评分。工作区无法可靠同步或冻结时保留错误，
不得评分一份不确定是否为最终状态的旧副本。

任务的公开测试是固定命令，模型不得提供任意宿主执行命令。合法构建配置修改
可以进入补丁，独立构建负责使其生效。需要自建测试时，测试文件仍按公开提交
规则处理；它不会成为私有评分依据。

## 预算、错误与恢复

可信来源准备、依赖和基线构建在求解前完成。模型请求、工具调用、模型触发的
构建与公开测试计入固定求解预算，最终独立评分另行记录。任务容器的 CPU、内存、
PID 和 GPU 资源来自任务环境声明；模型调用没有无限免费重试。

当前计时修订为 `3`。求解容器启动计入预算；只扣除已经完成的初始工作区上传及
所有权初始化的实测时长。该准备 helper 断网且不运行模型或候选命令。缺少完成
计量时保留未知，不把未完成上传当作免费时间。原始耗时、扣除时长和净求解耗时
分别保存，旧实验仍保留旧计时身份。

普通工具操作、公开构建或测试失败返回反馈，模型可以在余下预算内继续。
服务响应正常，但模型未调用 `finish`、请求未知工具或给出非法参数时，固定循环
结束求解并冻结已有补丁，最终仍由独立评分决定 `resolved` 或 `unresolved`。
这类协议违规与预算耗尽均属于该次模型尝试的结果，不自动使实验 incomplete，
也不因违规直接否定一份已经正确的补丁。

请求超过剩余求解时间按预算耗尽处理；其他 HTTP/模型服务或响应封装故障、工具
执行器故障、无法可靠捕获补丁及不可用的独立评分分别保留错误，并使报告 incomplete。
服务或工具故障后捕获的补丁仍可获得诊断性评分，但不计为有效模型结果。
工具返回成功不代表最终修复成功。
已经结束的尝试不因恢复而再次调用模型，同补丁重评保留来源。
[计分协议](statistical_protocol.md)说明完整性与分母的处理。

运行轨迹用于复核和后续分析，不包含凭据。本轮只实现必要记录，不做行为分析、
框架对比、复杂费用推断或 token/费用总额硬上限。
