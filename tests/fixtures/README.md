# 内部测试夹具

这些小型任务包只验证 OpBench 的执行与评分行为，不属于评测数据集，也不作为模型
能力结果。正式实现不会导入这里的源码；CI 通过公开 CLI 执行它们。

| 目录 | 验证范围 | Docker 镜像 |
| --- | --- | --- |
| [numeric_softmax](numeric_softmax/) | 标准库数值协议、比较器隔离、固定工具模型循环 | `opbench/numeric-fixture:py3.12`，Python/Git |
| [cpp_reduction](cpp_reduction/) | C++ 补丁重新编译、实际共享库加载 | `opbench/cpp-fixture:cpu`，Python/C++ |
| [pytorch_softmax](pytorch_softmax/) | 张量维度、dtype、非连续输入与梯度 | `opbench/softmax-fixture:torch2.6.0`，PyTorch CPU |

每组保留待修复 `source/`、公开测试、私有 `grader/` 与正确、替代、错误补丁。
控制检查要求原始缺陷和错误补丁不通过、正确与替代解通过。numeric 的比较器在
候选进程外；另两组的 grader 导入候选模块或共享库，用来检查集成行为。

## 引擎与容器检查

在仓库根目录安装 OpBench 后执行，不需要模型或 GPU：

```bash
docker build -t opbench/numeric-fixture:py3.12 tests/fixtures/numeric_softmax
opbench check-task --task tests/fixtures/numeric_softmax/task.docker.json \
  --reference tests/fixtures/numeric_softmax/controls/reference.patch \
  --alternative tests/fixtures/numeric_softmax/controls/alternative.patch \
  --mutation tests/fixtures/numeric_softmax/controls/mutation.patch \
  --output runs/v0.8/validation/quickstart-r1/numeric-controls
```

查看输出中的 `controls.json`；它引用各组的独立 `result.json`。另两组使用各自
Dockerfile、`task.json`、`gold.patch`、`alternative.patch`、`wrong.patch`，完整命令
由 [CI](../../.github/workflows/ci.yml) 维护。每次重新执行使用新输出目录。

只检查本机安装时，可用 numeric 的 `task.json` 运行独立评分：

```bash
opbench evaluate --task tests/fixtures/numeric_softmax/task.json \
  --patch tests/fixtures/numeric_softmax/controls/reference.patch \
  --output runs/v0.8/validation/quickstart-r1/installed-evaluation
opbench verify --evaluation runs/v0.8/validation/quickstart-r1/installed-evaluation
```

预期状态为 `resolved`，核验为 `valid: true`。这个 local 变体只供开发检查；
受控求解和 Docker CI 使用 `task.docker.json`。numeric Docker 环境 revision 2
改为轻量镜像，夹具数据清单同步升为版本 2；先前使用 PyTorch 镜像的记录仍属于
环境 revision 1。

## 可选模型接入检查

完成上面的镜像构建，按[模型接入指南](../../docs/v0.8/agent_integration.md)准备
本地 Codex 后，使用唯一的 [experiment.json](numeric_softmax/experiment.json)
进行单题联调。它显式选择 `gpt-5.5` / `high`，使用同一受控 Docker 流程：

```bash
opbench run --experiment tests/fixtures/numeric_softmax/experiment.json \
  --output runs/v0.8/validation/quickstart-r1/model-smoke
opbench report --run runs/v0.8/validation/quickstart-r1/model-smoke
```

这会实际调用模型。CI 使用脚本化响应检查固定循环，不需要模型服务或凭据。
恢复与更换模型按接入指南执行，运行证据按 [runs 约定](../../runs/README.md)保存。
