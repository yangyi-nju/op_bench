# 任务合同与可移植任务包

> **范围修订于 2026-09-14，接口复核于 2026-09-23。** 本文以[总体设计](design.md)为准。
> v0.8 使用固定模型循环、受控 MCP 工具与独立评分；分类扩充和完整选题/准入流程留到 v0.9，正式多模型实验留到 v0.10。
> 新任务只接受 schema 2，必须声明任务、环境和评分身份；旧任务须先迁移。
> 历史结果仍可按原协议读取，不因此重新获得可执行任务身份。

## 任务边界与计数单位

主评测覆盖算子自身的数值、形状、类型、设备、梯度、别名和状态行为，以及直接
影响这些行为的编译、分解、自动微分和运行链路。是否修改 CUDA/C++ kernel 不是
唯一判据。通用模块管理等框架维护问题单独归类；不能仅因来自算子框架就计入
算子主分数。独立性能优化另定评分协议，必要的性能回归限制则可纳入修复合同。

一个逻辑任务描述一个清晰的问题。同一问题的 dtype、CPU/GPU 或其他必需环境
是验证变体，不自动变成多道独立题。多个独立问题出现在同一上游 PR 中也不要求
合并成一题。关联缺陷需分组，用于数据划分和统计，防止近似答案跨集合泄露或
同一问题被重复计权。逻辑任务必须完成其公开声明的全部必需变体才能通过。
一次 Attempt 在声明的求解环境中产生一份补丁，评分覆盖同一基线下预定的必需
变体。若比较不同求解硬件或环境，它们属于分别声明的实验条件，不混成同一次
重复运行；不同源码基线的关联问题不能默认共用一份补丁。

## 公开行为合同

正式题面必须足以支持合理修复，包括必需接口、可观察行为、输入范围、必须保留
的设备或编译路径，以及性能限制等相关约束。具体隐藏判例可以不公开，要求本身
不能依赖读过参考补丁才能得知。

每个任务应具有双向映射：每条公开要求由什么评分证据验证；每个评分断言依赖
哪条公开要求或既有接口约定。不能在题面没有依据时强制特定内部命名或参考补丁
结构。候选可修改源码和构建配置、添加测试，不使用参考补丁导出的文件白名单。
关闭要求保留的编译或设备路径不算满足合同；不同算法与合理特化可以被接受。

领域检查涵盖 dtype、shape、
stride、设备、极值、空输入、梯度、别名和状态等应按具体风险选择。数值容限要
说明依据，不能统一逐位比较或任意放宽。组合行为示例、边界输入、性质测试和
独立参考计算；随机输入数量与覆盖率不能单独证明测试充分。

## 任务作者审查

结构检查和控制组执行不能代替语义审查。每题按风险说明下列事项，不为填表编造不适用的设备、dtype 或性能要求：

1. **触发与预期。** 写明公开接口、最小复现、执行路径与可观察结果。数值题说明独立参考和容差；元数据、梯度、别名、状态或异常分别说明实际判据。必要的性能要求另给预热、同步、硬件、重复和阈值依据。
2. **题面充分性。** 根据公开输入列出必需要求，再核对隐藏断言；记录审查者是否已接触参考解。真实错误栈和公开接口名称可以保留，不能让 Agent 猜未公开的新名称或 Gold 内部结构。
3. **双向覆盖。** 每条要求对应 F2P、P2P 或已声明的公开测试，每个评分断言有公开合同依据；解释 dtype、shape、设备、模式、极值、空输入和非连续布局的覆盖或不适用理由。选择器、样本和数值元素数量不是覆盖率证明。
4. **真实加载。** 无补丁暴露目标行为，参考修复通过，相关回归保留；记录实际执行而非只检查补丁可应用。新增、修改、删除及合法构建选项必须进入最终加载产物，不能用旧 wheel 的未变代码代替。
5. **鉴别力与公平性。** 错误变体违反明确语义要求并被对应断言检出，不以语法、导入或环境失败充数。高风险接口和内部结构限制需验证合理替代解；其他题记录审查理由，不机械要求相同数量的替代补丁。
6. **公开反馈。** 复现或公开测试实际运行并提供相关行为信息；补丁格式检查和隐藏测试选择器不构成公开行为反馈。修改自己的测试不能改动固定评分合同。
7. **来源与暴露。** 保留上游提交、许可、归属、开发使用情况和有依据的关联缺陷分组。已经用于开发的题不能重新声明为未见保留集；来源审查、争议及剩余限制随修订记录。

审查说明可放在任务 `metadata` 和随包资产中，引用真实证据；不存在凭元数据字段或文件存在自动授予准入的机制。使用 `opbench datasets validate` 检查当前 bundle 结构，使用 `opbench check-task` 保存控制结果，再完成人工判断。旧 schema 专用静态审计工具已退役；历史处置依据见[逐题初审](task_disposition.md)。

## 评分证据与后续准入

任务 `tests` 中的测试全部参与评分，发布时固定；当前没有可选诊断测试的 schema。
额外诊断应在评分合同外记录，不能事后改变主分数。没有执行、评分异常和行为
失败要分别记录，缺少必需结果不能算通过；额外诊断通过也不能抵消必需测试失败。

任务准入需要确认：基线暴露声明的问题、合理修复通过、相关既有行为保留、真实
执行路径加载了修改后的实现、环境可重建，并完成人工审查。参考补丁是一个正确
实现，不是唯一答案。易过度约束的接口和结构需验证其他合理修复；错误变体应
表达真实的错误思路，语法错误被拒绝不等于已验证语义覆盖。

候选构建不能接触私有 grader 资产。公开开发工作区、冻结补丁、候选构建与可信
评分是不同阶段。适用时由独立进程计算参考值并比较候选输出；其他测试框架也
应明确候选能读取或影响的范围。只读挂载和同进程测试计数不能证明裁判隔离。

软件版本、数据集发布版本、逻辑任务修订、环境版本和评分规则身份分别标识。每次修订需说明
是否要求报告复算、补丁重评、环境重建或 Agent 重跑。开发集、正式集、保留集
及关联缺陷分组须明确。旧 50 题参与过开发，不因格式迁移自动成为新的保留集，
也不继承旧 `verified` 标签；逐题重新准入。来源、公开时间、上游许可与归属要
保留；OpBench 自有代码许可证仍待与学校确认。

## 当前目录和 schema

`task.json` 必须使用 `schema_version: 2`。数据集清单、执行计划与结果记录各有
自己的 schema；下面 `dataset.json` 的 `schema_version: 1` 不是旧任务格式兼容入口。

任务包可在不依赖作者工作目录和 OpBench 源码仓库的环境中读取：

```text
dataset.json
tasks/example/task.json
tasks/example/source/        # 公开基线，不含答案资产
tasks/example/grader/        # 评测方资产
```

```json
{
  "schema_version": 1,
  "dataset_id": "operator-challenge",
  "version": "0.8-candidate",
  "status": "candidate",
  "tasks": ["tasks/example/task.json"]
}
```

任务索引路径保持在包内，任务 ID 唯一；源码和 grader 路径相对于 `task.json`。
开发环境支持绝对源码路径，正式发布应替换为可移植配方。当前三道候选的
`source.path` 指向仓库 `.op_bench_cache/` 下的缓存，需要在执行主机准备；清单能
加载不代表这些源码或镜像已经存在。
[LayerNorm 候选任务](../../datasets/v0.8_candidates/layer_norm_cpu/task.json) 展示了现有 schema；
[CPU LayerNorm 候选](../../datasets/v0.8_candidates/layer_norm_cpu/migration_review.md)与
[CUDA LogSoftmax 候选](../../datasets/v0.8_candidates/log_softmax_cuda/migration_review.md)
提供公开合同、完整源码构建配方和控制组资产；实际构建与控制进度集中记录在
[实施状态](development_status.md)，任务包的存在不代表已经准入。

当前主要字段为：

| 字段 | 当前接口含义 |
| --- | --- |
| `task_id`、`statement` | 身份和公开题面 |
| `task_revision`、`scope`、`defect_group`、`scoring_revision` | schema 2 必需的任务修订、语义范围、关联缺陷组和评分规则修订；`scope` 为 `operator`、`operator_integration`、`framework_support` 或 `uncertain` |
| `source.path`、`source.revision` | 本地目录快照或选定 Git 树；`revision` 用于 Git 来源，缺失路径需同时提供 `repo_url` 与修订 |
| `environment` | 后端、Python、镜像、资源；`prepare` 生成补丁基线，`build` 构建未修改基线及应用补丁后的候选 |
| `environment.prepare_timeout_sec`、`environment.build_timeout_sec` | 分别限制该阶段的每条命令，不是整个构建阶段的总预算 |
| `environment.environment_id`、`environment.revision` | schema 2 必需的环境身份；实际镜像 ID 另行记录 |
| `variants` | 可选必需评分变体，每项含唯一 `variant_id`；省略 `environment` 或 `tests` 时继承顶层值，提供时完整替换；不改变源码和题面 |
| `grader_dir`、`tests` | 可选独立 grader 路径及必需测试；测试 ID 与选择器均唯一，分组为 `fail_to_pass` 或 `pass_to_pass`，至少一个 F2P；numeric 必须有 grader |
| `public_commands` | 公开开发命令；私有选择器和私有元数据不进入求解输入 |
| `metadata` | 来源和审查证据；不是自动准入标记 |

命令使用 argv 数组，不自动经过 shell；参数支持 `{workspace}`、`{python}`，
仅有 grader 的评分会话支持 `{grader}`。环境变量值可用
`{workspace}` 指定工作区路径，`{runtime}` 指向独立运行目录，不能包含凭据。grader 必须加载修改后的源码或
产物，不能意外使用未改变的安装包。

任务未声明 `variants` 或声明为空数组时采用一个 `default` 评分环境；非空时所有变体均必需，
顶层 `environment` 仍是求解环境。变体的显式环境须包含自己的环境身份和 Docker 镜像等必要字段，不能仅提供部分字段来合并顶层环境。
同一冻结补丁逐个在干净环境评分，报告仍按一道逻辑题计数。父评分使用
`schema_version: 2`，`variants` 中每项只保存 `evaluation_path`、`status` 和 `resolved`；
路径相对父评分目录，指向子评分的 `result.json`。测试详情与日志只在子评分中保存，
父记录不再展开或复制 `cases/groups`。任务快照同时固定求解和评分变体的镜像；
身份声明本身不是准入证明。未声明变体的单环境评分仍使用 schema 1。

`kind: "unittest"` 与 `expected_tests` 检查执行数量和失败等状态，空套件
不能通过；`kind: "command"` 仅检查进程结果，可信驱动需落实行为断言。二者都
不能证明任意候选与判定器共享进程时的隔离。

`kind: "numeric"` 使用候选 CLI 与控制器独立比较。`argv` 声明候选程序的执行命令，
不能引用 `{grader}`；`oracle` 是相对私有 grader 的 JSON 路径，其内容为：

```json
{"schema_version": 1, "input": {"x": [1, 2]}, "expected": {"shape": [2], "values": [0.2689414213699951, 0.7310585786300049]}, "atol": 1e-12, "rtol": 1e-10}
```

只有 `input` 通过 stdin 发送给候选；候选 stdout 返回 `shape` 和扁平 `values`，
stderr 单独记录。候选停止后，控制器检查输出完整性，并按
`abs(actual-expected) <= atol + rtol*abs(expected)` 判定。标量 shape 为 `[]`，
恰有一个值；含零维长度的空张量允许空值数组。首版只支持有限、可用 binary64
表示的实数，`expected_tests` 固定为一个协议 case，不能用自报字段证明内部设备、
dtype 或性能。非法参考文件为评价异常，非法/缺失输出和失败进程不能通过。
协议的工程验证使用[数值测试夹具](../../tests/fixtures/README.md)，不属于正式数据集。

候选构建在无 grader 会话内进行；仅工作区产物转移到后续会话，依赖应预装在镜像。
每个 numeric case 从同一构建产物复制出独立执行环境。Local 仅供开发，不提供私有
文件隔离；Docker 行为验证也不替代部署、镜像或题目语义审查。

## 分类与发布信息

任务 `metadata.category` 可保存分类描述。v0.9 再确定分类体系、每类至少 10 道独立
任务的规模，以及选题、筛选、审查和准入流程。本轮不在运行器中硬编码类别配额，
也不因缺少准入声明而阻止开发任务评分。

已有数据包的 `status`、来源和审查 metadata 可继续保存，但它们不是评分开关。
`resolved_rate` 只依据本轮预先声明的任务集合和执行结果；开发集合得到比例不
意味着任务已完成准入或可以作为正式研究发布。旧 `admission` 声明及历史
verified 不能代替 v0.9 的审查。具体计数见[计分协议](statistical_protocol.md)。

## 准备和检查

```bash
opbench prepare --task path/to/task.json --fetch --output runs/v0.8/validation/quickstart-r1/prepared
opbench datasets validate --dataset path/to/dataset.json
opbench check-task --task path/to/task.json --reference gold.patch \
  --alternative another-correct.patch --mutation targeted-error.patch \
  --output runs/v0.8/validation/quickstart-r1/task-controls
```

`--fetch` 在新受管目录准备声明的 Git 修订和子模块，不重置已有工作树；现有
源码只读检查。未准备的修订或子模块需显式处理。评分不应静默下载依赖、切换
任务修订或修改源码来源，离线运行前应完成准备。结构有效与环境已准备分别记录。
`prepare` 导出干净源码和公开任务输入，不执行任务的 `environment.prepare/build`；
它不是已经编译好的运行环境。

`run` 在复制源码、固定镜像与准备模型输入之前保存
完整 `plan.json`。逐项准备保存状态、阶段、耗时、错误与每次物理尝试；只发布
准备完成的输入。修复准备阶段缺失的源码或镜像后，用原命令加 `--resume` 继续，
已有输入不重复制作。恢复只启动尚无 `attempt.json` 的尝试；已有尝试即使发生
模型服务故障也不重跑，残留的 `running` 标为 `interrupted`。未完成准备须保持原计划声明和软件
条件；不能借恢复更换题目、补丁或实验配置。失败的阶段产物与记录保留用于诊断。
这一恢复入口只适用于 `run`，单独 `prepare` 和 `build-baseline` 没有 `--resume`。
`evaluate` 只接受单个任务和补丁（或 `--baseline`），用于评分开发与检查，不再实现
批量外部补丁导入、模型实验计划或恢复流程。

`evaluate` 退出码 0 表示取得了可判定的评分结果，可能是 `test_failed`、
`invalid_patch` 或 `build_failed`；是否修复要读取 `result.json` 的 `resolved`。
`verify --evaluation` 接收包含 `result.json` 的目录，只核验已有证据，不重新执行候选。

`check-task` 在源码与镜像准备前固定 `control_plan` 和各控制补丁。准备失败或中断
时，`controls.json` 保留阶段、错误、已执行结果引用以及完整矩阵中剩余的 `not_run`。
当前控制组文件为 schema 2，各控制项的 `evaluation_path` 指向其独立 `result.json`，
详情不再内嵌。控制项保存状态，完整计划保存对应补丁；通过数量可以从条目直接计算。
`not_run` 不是一次行为失败或成功；部分执行不能被当作完整控制组。该命令没有
`--resume`，再次检查使用新的输出目录，可显式复用已有 ready 基线。

这些命令为检查工具，不能替代语义审查。控制实验缺失、不可运行或仅有文件资产
时都不能完成准入。质量证据来自实际行为及审查，不来自文件哈希级改动验证。

## 显式复用基线构建

完整框架可先生成未应用候选补丁的基线构建，避免每个控制组从零开始编译：

```bash
opbench build-baseline --task path/to/task.json --output runs/v0.8/validation/quickstart-r1/baseline-build
opbench evaluate --task path/to/task.json --patch candidate.patch \
  --baseline-artifact runs/v0.8/validation/quickstart-r1/baseline-build --output runs/v0.8/validation/quickstart-r1/candidate
opbench check-task --task path/to/task.json --reference gold.patch \
  --alternative another-correct.patch --mutation targeted-error.patch \
  --baseline-artifact runs/v0.8/validation/quickstart-r1/baseline-build --output runs/v0.8/validation/quickstart-r1/reused-controls
```

基线创建时不接收候选补丁、私有 grader 或模型凭据；执行 `prepare/build`，停止
执行并捕获工作区后才可成为 `status: "ready"` 的产物。产物同时保留准备后的源码
和构建工作区。源码生成应写入 `prepare`；`build` 修改或删除已准备源码会使该
复用合同失败，需要调整配方。跨会话依赖仍须预装在镜像，其他持久产物位于工作区。

控制器为 HOME、临时文件、Python 用户安装与缓存分配独立的运行目录；这些目录
随构建产物保留，但不属于源码提交。可用 `{runtime}/...` 声明位置，或指定工作区
内尚不存在的新目录。运行目录不得覆盖现有源码、整个工作区或控制器元数据。
`artifact_contract.runtime_paths` 记录实际保留子树；公开 Agent 输入列出同一提交边界。

`evaluate --baseline-artifact` 为每次评价复制独立工作区，应用补丁并再次执行
正常构建。`check-task --baseline-artifact` 复用已有产物；不提供已有产物时，可以
选择 `--reuse-baseline` 为本次控制组构建一份，两参数互斥。有必需 variants 时分别
建立每个评分环境的基线，所有变体都必须 ready；CPU/CUDA 产物不能混用，同一补丁
仍须通过全部变体。

`run` 的实验配置使用 `reuse_baseline: true` 开启同题各模型尝试间的基线复用；没有直接接收
任意已有产物的 CLI 参数。存在 `variants` 时，计划分别建立求解环境与评分变体基线。
默认 `false` 也走同一可信构建和独立副本流程，只是每次 Attempt 自行准备临时基线，
保留构建记录与日志，评分结束后清理临时工作区。共享开关不改变源码或提交合同。
`--baseline-artifact` 当前用于单任务 `evaluate`、`check-task` 和 `replay`，不用于数据集评价。
`replay` 默认仍创建全新构建；显式选择产物时，在冻结当前任务快照后按原源码来源和
环境验证兼容性，再将原 Attempt 的冻结补丁应用到独立副本并执行正常构建。
记录保留所选产物来源；纯评分修订使用新 grader，不重新生成或替换原补丁。
重评要求原尝试明确记录 `submission.status: frozen`；缺少冻结声明的早期记录保留
为历史证据，不再通过猜测已有评分来补足提交状态。

所有固定框架尝试以 prepare 后、build 前的源码视图建立初始 Git 仓库，并公开
`.opbench/submission-policy.json`。已有源码的修改和删除被捕获；新增文件必须符合公开的源码提交边界。
保留 runtime 子树和初始构建产物不属于补丁。固定工具不提供通用 Git 操作，
通过写文件或文本替换工具显式写入的合法源码自动登记提交，包含原本被 Git
忽略的新源码；模型不需要使用 Git 工具。任务不得将合法修复必需的源码路径
误声明为运行或生成产物。完整合同见[模型接入协议](agent_integration.md#输入求解与提交)。

这是操作者选择的可信构建输入：兼容检查针对源码与完整构建环境，包括镜像、
`prepare/build`、环境变量和资源声明。原生产者的任务/评分身份保留在
`producer_task_identity`，纯题面或评分修订不因身份变化而使相同构建输入失效；
新评价仍记录当前任务身份。不做文件哈希认证，也不应接收 Agent 提交的所谓缓存。
始终执行 rebuild 不等于证明增量依赖完备；任务需验证源码新增、修改、删除和合法构建配置变更确实进入
最终产物，必要时由其构建配方清理过期产物。Local 绝对路径产物可能不可移植，
不能据此声称跨机器复现。失败或中断基线不能被当作 ready 复用。
