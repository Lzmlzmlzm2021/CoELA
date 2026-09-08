# PeerConsult V4：设计、实现与跨 Benchmark 复现规范

## 0. 文档定位

本文是 PeerConsult V4 的权威设计与复现文档，面向需要把该方法迁移到其他场景、任务、机器人形态或模拟器 benchmark 的实现者。

- 当前实现协议名：`PeerConsultV4`
- 当前实现位置：TDW-MAT / CoELA
- 文档快照：2026-09-08 当前工作区版本（按 `peer_consult.py`、`lm_agent.py`、`LLM.py`、`challenge.py` 的实际调用链复核）
- 当前状态：已实现，具备定向单元测试，并已产生 TDW-MAT 运行轨迹
- 历史版本：V3.3--V3.5 仅作为实验变体保留，不应被混入 V4
- 事实来源优先级：当前代码 > 本文 > 历史设计草案

全文使用以下标签区分可迁移边界：

- **[V4 Core]**：跨 benchmark 应保留的机制与不变量。
- **[Adapter]**：需要针对环境重写的感知、动作、身份和完成判定接口。
- **[TDW-MAT]**：当前 TDW-MAT/CoELA 的具体实现，不应直接硬搬到其他 benchmark。
- **[Research Option]**：可消融或后续实验的策略插件，不属于当前 V4 Core。

## 1. 方法摘要

PeerConsult V4 是一个由 Harness 支持的多智能体规划系统：

```text
多个局部 LLM Planner
        +
共享的事实 Dashboard / Blackboard
        +
原子资源协调与事实型 Validator
        +
各自原有的具身技能执行器
```

一句话定义：

> **决策去中心化，事实与资源协调中心化；Harness 强约束不变量，但不替 LLM 选择 benchmark 的具体最优答案。**

V4 不是以下系统：

- 不是一个中央 LLM 同时规划所有 agent；
- 不是让一个小模型充当另一个小模型的 peer critic；
- 不是把距离最近、容器最优、deadline 强制运输等 TDW-MAT 策略写死；
- 不是完全自由的自然语言协作；
- 也不是“零硬编码”。

## 2. 设计哲学

### 2.1 Mechanism-heavy, policy-light

Harness 应硬编码的是可跨任务表达的机制：

- 动作是否合法；
- 事实是否已由环境确认；
- 同一资源能否被两个 agent 同时占用；
- 任务、claim 和执行结果如何形成一致生命周期；
- 同一个无进展计划是否进入死循环；
- episode 之间的状态是否隔离；
- 消息是否来自允许公开的结构化事实。

Harness 不应默认硬编码的是策略答案：

- 应该抓哪个目标；
- 应该探索哪个房间；
- 应该走哪条路线；
- 容器是否值得拿；
- payload 何时运输；
- 失败后具体换哪个目标；
- 当前任务是否“聪明”。

判断一条硬规则是否应该进入 Core，可使用以下标准：

1. 它是否表达环境或资源的不变量，而不是某个 benchmark 的高分技巧？
2. 换成不同场景、任务和模拟器后，语义是否仍然成立？
3. 它是否只删除不合法/相互冲突的选择，而不是替模型决定最优选择？
4. 它是否能通过明确输入、输出和测试验证？
5. 如果该规则失效，是否会破坏事实一致性、安全性或活性，而不仅仅是降低分数？

若主要回答为“否”，应将其放入 Adapter 或可关闭的 Research Option。

### 2.2 小模型需要工作流约束，但仍需保留策略自主性

V4 接受 Harness 对小模型的重要性：小模型不应反复自行推导 ownership、任务生命周期、重复 claim、episode reset 等基础机制。但如果 Harness 继续替模型决定目标、路线、运输时机和替代任务，就会逐渐退化为面向单一 benchmark 的启发式系统。

因此边界是：

```text
Harness：提供正确、紧凑、可行动的思考环境
LLM：在这个环境内完成策略选择
```

### 2.3 中心化实现不等于中心化规划

当前 TDW-MAT 版本使用一个集中式 Coordinator 和 Blackboard，以便在同一个环境 step 原子处理两个 proposal。中心节点看见必要的符号事实，但不运行中央策略模型，也不替局部 planner 分配具体任务。

迁移到分布式系统时，可以把 Blackboard 替换成事务服务、共享数据库或共识协议，只要保持相同的原子语义。

## 3. 总体架构与职责

```text
Environment observations / evaluator truth
                    |
                    v
          Environment Adapter [Adapter]
          - legal candidates
          - stable identities
          - ownership/completion/progress facts
                    |
                    v
       Shared Blackboard + Harness [V4 Core]
       - goal ledger
       - persistent tasks
       - claims/reservations
       - public lifecycle events
       - one-boundary loop guards
            |                    |
            v                    v
   Agent 0 decision card   Agent 1 decision card
            |                    |
            v                    v
     Local Planner 0       Local Planner 1
     + private memory      + private memory
            | high-level plans   |
            v                    v
   Original skill/executor step [Adapter]
   - computes proposed low-level actions
   - does not mutate the environment yet
            |                    |
            +------ proposals ---+
                    |
                    v
     Canonical intent + task synchronization
                    |
                    v
       Atomic factual validation [V4 Core]
                    |
                    v
          Final joint action dictionary
                    |
                    v
              Environment step
                    |
                    v
             Outcome observation
                    |
                    +---- feedback loop
```

模块职责：

| 模块 | 接收 | 产生 | 不负责 |
|---|---|---|---|
| Environment Adapter | 原始 observation、环境 API、evaluator | 符号事实、合法候选、稳定身份、进展事件 | 团队策略选择 |
| Blackboard | 两名 agent 的公开事实与执行结果 | goal/task/claim/ownership/event ledger | 路线和目标效用计算 |
| Decision Card Compiler | Blackboard + 当前 agent 身份 | agent-specific Dashboard | 自动合并 peer 私有观测 |
| Local Planner | 私有记忆 + Dashboard + 合法候选 | 一个高层 plan | 修改环境事实 |
| Skill/Action Proposal Adapter | 高层 plan + 私有导航/技能状态 | 本轮 proposed low-level action | 真正提交环境动作、跨 agent 仲裁 |
| Intent Canonicalizer | 高层 plan + proposed action | task/stage/target/action identity | 策略排序 |
| Atomic Validator | 所有 intent/proposal + 权威事实 | 接受、拒绝或一次性暂停后的 final action | 完整路径规划、为 loser 指定替代任务 |
| Environment Executor | final joint actions | 环境状态变化与 terminal outcome | 高层规划和任务分配 |
| Completion Oracle | 环境/evaluator 状态 | authoritative completion | 接受 agent 自报完成 |

## 4. 信息边界与信任模型

### 4.1 私有信息

**[V4 Core]** 以下信息默认不得自动复制给 peer planner：

- 原始图像、深度图、segmentation；
- occupancy map、局部路径图；
- 精确 pose、目标距离和路线；
- 未公开的局部物体发现；
- hidden chain-of-thought、主观评分和模型推理；
- 未被环境验证的完成声明。

### 4.2 Harness 内部可见的事实

集中式实现可以为了验证而读取必要的符号事实，例如：

- 当前可见实体 ID 和类型；
- 实际 held objects；
- evaluator-confirmed completion；
- 当前 proposal 和底层 action 状态。

“Harness 内部可见”不等于“自动放入 peer prompt”。V4 的 Dashboard compiler 必须再次执行信息边界过滤。

### 4.3 可以公开给 peer planner 的信息

- goal requirement 与权威完成进度；
- 自己的 active/suspended task；
- peer 已公开承诺的任务；
- payload 的对象身份；
- object claim 和 room reservation；
- task commitment/suspension/release/failure/progress/completion；
- factual validator 对当前 agent 的最近反馈；
- benchmark 的原始 budget 事实。

### 4.4 权威性层级

```text
Completion oracle / evaluator truth
        > physical ownership reported by environment
        > atomic claim / reservation
        > task commitment
        > natural-language dialogue
        > agent-local planner bookkeeping
```

低层信息不得覆盖高层事实。例如，对方拿着物体不代表该物体已经送达；自然语言声称“完成”也不能修改 goal ledger。

## 5. Agent-specific Dashboard

### 5.1 当前 V4 卡片

**[V4 Core]** 当前实现会在每个 Coordinator step 为每个 agent 独立刷新紧凑卡片；只有到达真正的 planning boundary（TDW-MAT 中通常是 `plan is None`）时，LLM 才消费最新卡片。低层动作处于 `ongoing` 时不会重复调用 LLM。当前字段为：

```json
{
  "frame_budget": {
    "current": 100,
    "remaining": 2900
  },
  "goal": {
    "delivered_total": 1,
    "required_total": 4,
    "remaining": {"apple": 2, "book": 1}
  },
  "self": {
    "room": "kitchen",
    "payload": [{"id": 42, "name": "apple"}]
  },
  "peer": {
    "payload": [],
    "commitment": {
      "task_id": "room:living_room",
      "kind": "explore_room",
      "status": "in_progress",
      "priority": 20
    }
  },
  "active_task": {},
  "task_queue": [],
  "active_claims": {},
  "planning_loop_guard": null,
  "coordination_events": [],
  "recent_coordination_events": [],
  "recent_relevant_reviews": [],
  "recent_agent_dialogue": [],
  "coordination_guidance": []
}
```

当前卡片明确不包含：

- 距离排序；
- evidence confidence/age；
- container utility；
- coverage heuristic；
- deadline cost estimate；
- Harness 指定的替代目标、房间、路线或容器。

因此 V4 Dashboard 不是“底层导航信息板”，而是一个确定性编译的协调与工作记忆视图。默认情况下，它直接携带的空间信息主要只有 `self.room` 和任务中的语义 room；精确 pose、地图、路径、waypoint、目标距离和 navigation progress 仍保留在各 agent 的私有 memory/executor 中。默认关闭的 TDW-MAT `TDW_MAT_V4_SHARED_DELIVERY_TARGET=1` 扩展是例外：启用后可以公开经过适配器验证的 bed ID、position 和 room。

### 5.2 私有发现的处理

Blackboard 可以为验证维护 `known_entities`，但一个只由 peer 私下发现、尚未形成公开 commitment/claim 的任务，不会自动进入当前 agent 的 `task_queue`。

同时，V4 的 candidate compiler 不把共享 `task_queue` 当成局部目标白名单。因此 agent 自己看见的、物理可获取的对象，即使尚未在 Dashboard 出现，也仍然可以成为候选。

这两个规则共同实现：

```text
不泄漏 peer 私有发现
        +
不抹掉自己的局部发现
```

### 5.3 当前 TDW-MAT prompt 窗口

**[TDW-MAT]** `LLM.run()` 使用：

- 最近 10 条 `action_history`；
- 最近 6 条自然语言 dialogue；
- 最新 1 张 `[Memory Board]`；
- 当前局部 `object_list`、`object_per_room`、holding、`satisfied`、peer-held 等状态。

Dashboard 是 prompt 的一部分，不绕过 LLM 直接选择正常策略。

## 6. Persistent Task System

### 6.1 稳定任务身份

**[V4 Core]** 任务必须有跨 planning boundary 稳定的 ID。当前 TDW-MAT 映射为：

```text
获取具体实体： entity:<object_id>
运输当前 payload：delivery:<agent_id>
探索房间：       room:<room_identity>
```

通信不是任务切换。执行器内部短暂的 `plan=None` 也不是任务释放。只有真正 planning boundary 上的显式 idle/release 才清除 active task。

### 6.2 生命周期

通用状态：

```text
pending -> in_progress -> completed
                 |
                 +-> suspended -> in_progress
```

TDW-MAT 还会出现反映物理事实的状态：

```text
carried
surplus
in_use
```

V4 的关键语义是 **persistence without insistence**：

- active task 获得连续性提示；
- 切换时旧任务被 `suspended`，不会删除；
- suspended task 仍可重新选择；
- validator 拒绝任务时不施加长期时间 block；
- Harness 不强迫每轮恢复旧任务。

### 6.3 固定 priority

Priority 字段和非强制排序机制属于可迁移 Harness 接口；具体数值属于 benchmark 配置。

**[TDW-MAT] 当前数值：**

| Task kind | Priority |
|---|---:|
| `deliver_payload` | 130 |
| `deliver_goal_object` | 100 |
| `container_resource` | 40 |
| `explore_room` | 20 |

当前 `task_queue` 排序为：

```text
active task
-> suspended task
-> numeric priority descending
-> stable task ID
```

必须注意：当前实现不会按照 numeric priority 强制重排全部 CoELA action candidates。它只在 Dashboard 的 task queue 中展示 priority，并把 active task 对应的物理 plan 稳定地提前。Priority 是模型先验，不是合法性规则。

## 7. Legal Candidate Compilation

### 7.1 完整合法候选

**[V4 Core]** Adapter 应先生成当前状态下完整的合法高层动作集合。Harness 可以删除确实非法的候选，但不能仅因策略偏好删除合法动作。

允许硬过滤的通用例子：

- embodiment 不支持；
- action schema 非法；
- 目标已经权威完成；
- 目标已经被 peer 物理持有；
- 没有空闲操作资源却要继续 acquire；
- 本轮 atomic claim 失败；
- 消息引用了不存在或不可公开的事件。

不应由 Core 硬过滤的例子：

- 距离较远；
- priority 较低；
- payload 尚未满；
- 不是 active task；
- 探索的信息价值可能较低；
- deadline 还不紧迫；
- 最近失败但环境已经变化。

### 7.2 TDW-MAT 当前行为

**[TDW-MAT]** 合法候选包括：

- `go grasp target object ...`
- `go grasp container ...`
- `put ... into the container ...`
- `transport objects I'm holding to the bed`
- `go to <room>`
- `explore current room <room>`
- `send a message`
- 特定角色的 `wait`

V3.x 曾在已知目标存在时抑制 blind exploration；V4 恢复候选完整性。即使当前有目标任务，合法的 room navigation/exploration 仍可见，由 LLM 判断其信息价值。

角色能力仍是硬约束。例如 scout 不得获得 manipulation candidate。

### 7.3 Dashboard 对候选的直接影响边界

**[TDW-MAT]** `get_available_plans()` 的主要输入仍是当前 agent 自己的 `object_list`、`holding_objects`、`rooms/current_room/rooms_explored` 和 role。Dashboard 对候选编译的直接机械影响目前只有少量内容：

1. `active_task`：把匹配的物理候选稳定提前，但不增加、删除或强制选择候选；
2. `planning_loop_guard`：取消被 guard 任务的 continuity 排序奖励，但不删除该候选；
3. 可选 `delivery_target`：当本地不知道 bed 但正在持有 payload 时，可以启用 transport 候选，并把 bed room 导航候选提前。

V4 的 `task_queue`、peer commitment 和 `active_claims` 主要供 LLM 在 Memory Board 中进行语义判断，并供后置 Validator 仲裁。它们通常不会从 agent 的私有候选集合中直接删除冲突项。例如，对方已 claim 的物体仍可能因为本地 `object_list` 而出现在候选中；理想情况下 LLM 主动避开，若仍选择则由 Atomic Validator 拒绝。

## 8. Local LLM Planning

### 8.1 输入

每个 planner 接收：

```text
private local state
+ recent local action history
+ recent received dialogue
+ latest agent-specific Dashboard
+ complete legal candidate list
+ protocol-level guidance
```

### 8.2 输出

当前 TDW-MAT 保持 CoELA 的文本 plan 输出，例如：

```text
go grasp target object <apple> (42)
go to <Kitchen> (6000)
transport objects I'm holding to the bed
send a message
```

迁移到其他 benchmark 时可以改为 JSON/action ID，但必须能够规范化为：

```json
{
  "agent": 0,
  "stage": "acquire",
  "task_id": "entity:42",
  "target_id": 42,
  "action_identity": "stable-or-opaque-id",
  "planning_boundary": true
}
```

### 8.3 Planner 自主负责

- 选目标和房间；
- 是否继续 active task；
- 是否恢复 suspended task；
- 探索、收集、容器和运输之间的取舍；
- 失败后选择新的策略；
- 是否通信以及选择哪个公开事件。

当前 V4 不运行 peer LLM critic。所谓 `review` 是确定性 factual validator 的审计记录。

## 9. Atomic Object Claim

### 9.1 三个不同概念

```text
claim          = agent 正准备获取具体资源
physical owner = 环境确认 agent 已经持有资源
completed      = completion oracle 确认任务完成
```

三者不能相互替代。

### 9.2 原子流程

**[V4 Core]** Coordinator 必须先收集同一协调步中所有 proposal，再统一决定 claim：

```text
collect all intents
-> group acquire intents by resource ID
-> reject completed/owned resources
-> honor an existing valid claim
-> atomically grant at most one new claim
-> reject competing proposals
-> never let both conflicting proposals reach execution
```

当前 TDW-MAT 同帧没有已有 claim 时，使用由 `protocol_step + target_id` 产生的轮换式确定性 tie-break；它不比较距离，也不永久偏向 Agent 0。

### 9.3 Claim 生命周期

以下事件释放 claim：

- planner 切换到不同具体目标；
- task 被显式释放；
- physical ownership 建立；
- evaluator 确认完成；
- loop guard 要求释放；
- 有限 lease 超时。

Claim 是 Harness 的结构化协调原语，当前不要求先发送自然语言消息。在其他 benchmark 中，应明确 claim 是否免费、是否消耗通信带宽，并在实验中记录。

## 10. Room Exploration Reservation

**[V4 Core]** Room claim 是对 blind exploration scope 的临时 reservation，不是永久房间所有权，也不禁止 agent 为一个已知目标进入该房间。

当前 V4：

- 对 `go to` 和 `explore` proposal 建立 room reservation；
- 一个 agent 同时只保留一个 room reservation；
- agent 改选房间时释放旧 reservation；
- 同一 room scope 冲突时只批准一个；
- loser 被要求下一轮自行选择其他合法动作；
- 不用距离、room utility 或已知目标数量选择 winner。

在没有离散 room 的 benchmark 中，可替换为区域、搜索扇区、工作区、工具或其他可互斥探索 scope；如果不存在合适 scope，可以禁用该机制，但必须保留 object/resource claim。

## 11. Factual Validator

### 11.1 当前 V4 强制检查

Validator 只执行事实、能力和互斥约束：

1. pending one-boundary loop guard；
2. evaluator-confirmed completion；
3. physical ownership；
4. existing object claim；
5. duplicate same-object claim；
6. 无空闲手/操作资源时的 acquire；
7. room reservation conflict；
8. communication event ID 与公开 event type；
9. adapter 提供的角色/动作合法性（当前 TDW-MAT 在 `lm_agent` 内、进入 Governor 前检查，不属于 `_review_and_govern()` 本体）。

被拒绝时，当前 TDW-MAT 会：

- 清除该 agent 的高层 plan 和 target cache；
- 将本轮底层动作替换为安全转向动作；
- 记录 `verdict=revise`；
- 将相关任务保留为 suspended；
- 释放与该失败 proposal 相关的 claim/reservation；
- 下一 planning boundary 再由 LLM 自行选择。

“通过 Validator”仅表示 proposal 没有违反上述已实现不变量，不表示完整物理可行。当前 V4 不验证路径一定可达、运动一定无碰撞、抓取一定成功或模拟器一定接受该动作；通过的 final action 仍可能在 `env.step()` 后返回 invalid/failure。

### 11.2 当前 V4 明确不运行的旧策略

- distance/confidence ranking；
- goal-type quota governor；
- late acquisition/deadline governor；
- payload/container utility policy；
- navigation spatial-progress recovery；
- waypoint coverage exploration；
- 强制 delivery commit；
- Harness 自动分配替代 target/room；
- 两个 LLM 之间的 critic call。

### 11.3 Optional policy plugins

**[Research Option]** Strategy advisory、deadline safety kernel、quota control、container utility、spatial collision serialization 等可以作为独立、可关闭、可记录的插件研究，但不得静默混入 Core。每次插件覆盖 LLM proposal 都必须记录原 proposal、触发证据和最终动作，并单独报告 forced-action rate。

当前 V4 的 deadline advisory 明确为：

```json
{
  "force_delivery": false,
  "priority_delivery": false,
  "recommendation": "planner_decides"
}
```

也就是说，当前正常运输时机完全交给 planner。

## 12. Opaque Planning-loop Guard

### 12.1 监控 planning boundary，而不是底层动作步

一个高层 `go grasp` 或 `go to` 可能执行很多 move/turn action。V4 不把这些连续底层动作误认为多次 LLM 重试，只在真正调用 planner 时建立 boundary。

Adapter 必须为每个新 plan 提供：

- stable task ID；
- stable/opaque action identity；
- terminal execution outcome；
- task progress version。

### 12.2 当前算法

TDW-MAT 用以下字段生成 action hash：

```text
task_id + stage + target_id + normalized plan + action_type
```

如果同一 agent 连续两次在 planning boundary 上重复相同 task/action，并且 `task_progress_version` 没有增加：

1. arm 一个 guard；
2. suspend 当前任务；
3. 释放相关 claim/reservation；
4. 清除 plan；
5. 在下一次 planning boundary 只拒绝完全相同的 action identity；
6. planner 必须自行选择任意其他合法动作；
7. guard 立即被消费；
8. 原任务之后仍可重新选择。

它不是 360-frame cooldown，也不会长期封锁 target。

### 12.3 Progress heartbeat

**[V4 Core]** Core 只消费 adapter 报告的单调 progress version，不用距离和置信度推断“是否有进展”。

**[TDW-MAT]** 当前 progress 来源包括：

- 新 physical ownership；
- evaluator-confirmed delivery；
- room exploration task 完成。

迁移时必须谨慎定义 progress：过少会误伤正常的长动作，过多会掩盖死循环。应通过协议测试验证“真实进展重置计数、无进展重复触发一次性 guard”。

## 13. Decision-first Structured Communication

### 13.1 通信也是一个普通候选

Planner 先在物理动作和 `send a message` 之间作选择。只有选择通信后，才进行通信内容选择，因此不会在每轮无条件多调用一次生成模型。

### 13.2 当前允许公开的事件

```text
task_commitment
task_suspended
task_released
task_failure
task_progress
task_completed
claim_acquired
claim_released
```

模型只能输出：

```text
coordination_event:<event_id>
```

Harness 再渲染固定 JSON。模型自由文本、原始 observation 和 hidden reasoning 不进入 peer channel。无效 event ID 被替换为安全 wait，并记录 validator failure。

### 13.3 为什么不使用 peer critic

小模型拥有的信息本来就不完整，让它再评价另一个 agent 的策略容易放大信息不对称和模型误差。V4 只共享结构化生命周期事实；复杂策略仍由每个 agent 根据自己的局部信息决定。

## 14. 一次完整协调循环

当前执行顺序必须保持以下语义：

1. **Observe public truth**：读取双方 observation、held objects 和 completion oracle。
2. **Update Blackboard**：更新 goal ledger、ownership、tasks、claims、events 和 progress version。
3. **Build cards**：在收集本轮 proposal 之前，为每个 agent 生成独立 Dashboard。
4. **Inject cards**：将当前卡片提供给各自 planner。
5. **Local perception**：每个 agent 更新自己的 private memory 和 local candidates。
6. **Local planning**：只有 `plan is None` 时调用 LLM；低层 `ongoing` 不重复规划。
7. **Compute one local skill step**：skill executor 先产生本轮 proposed low-level action；此时尚未调用环境，不产生真实副作用。
8. **Canonicalize intents**：将 plan/action 转成 task、stage、target 和 boundary identity。
9. **Sync persistent tasks**：记录 commitment、switch、suspension 和 release。
10. **Atomic validation**：统一处理 guard、ownership、object claim、room reservation、hand capacity 和 message event。
11. **Finalize actions**：接受合法 proposal，拒绝冲突 proposal，但不指定替代任务。
12. **Environment step**：并发执行最终动作。
13. **Observe outcome**：读取 terminal evidence、ownership 和 evaluator truth。
14. **Update progress**：释放 claim、完成 task 或累积 no-progress boundary。
15. **Log**：保存 proposal、validator、final action 和 outcome。

伪代码：

```python
def coordination_step(states, completion_truth):
    board.observe(states, completion_truth)
    board.consume_execution_evidence()

    cards = {
        i: board.build_agent_card(i)
        for i in agents
    }

    proposed_actions = {}
    for i in agents:
        agents[i].set_dashboard(cards[i])
        proposed_actions[i] = agents[i].act(states[i])

    intents = canonicalize_all(agents, proposed_actions)
    board.sync_tasks(intents, genuine_planning_boundaries(intents))

    final_actions = factual_validator.atomic_review(
        intents=intents,
        actions=proposed_actions,
        public_truth=board,
    )

    return final_actions
```

虽然当前 Python 代码顺序调用两个 `agent.act()`，但 claim 在收集完两个 proposal 后才原子决定。因此第二个 planner 不会在同一轮免费看到第一个刚产生的隐藏 proposal；冲突由 atomic validator 处理。

## 15. 跨 Benchmark Adapter Contract

移植时不要复制 TDW plan string parser；应实现下列语义接口。

### 15.1 Observation adapter

最少输出：

```json
{
  "agent_id": 0,
  "step_or_time": 100,
  "terminal_action_evidence": [],
  "physical_resources_owned": [],
  "public_completion_facts": [],
  "local_observation_handle": "private"
}
```

原始局部 observation 可以留在 agent 内部，不必进入 Core。

### 15.2 Candidate adapter

必须提供当前 agent 的完整合法高层动作。每个候选至少具有：

```json
{
  "action_id": "grasp:42",
  "task_id": "entity:42",
  "stage": "acquire",
  "target_id": 42,
  "legal": true
}
```

不要把策略分数伪装成 legality。

### 15.3 Identity adapter

必须保证：

- 同一任务跨多次规划拥有稳定 task ID；
- 同一个语义 action 可以得到稳定 action identity；
- identity 不需要泄露 pose、路径或私有 reasoning；
- episode reset 后 identity ledger 不复用旧状态。

### 15.4 Progress adapter

显式发出：

```json
{
  "task_id": "entity:42",
  "event": "physical_ownership_acquired",
  "progress_version": 3
}
```

不要仅因 agent 移动了一小段距离就宣称 task progress，除非该 benchmark 明确定义这属于任务阶段进展。

### 15.5 Completion adapter

提供不可由 planner 对话任意修改的 authoritative completion。它可以来自：

- simulator evaluator；
- environment predicate；
- task API 返回值；
- 机器人传感器与受信状态机；
- 外部验证器。

### 15.6 Executor adapter

当前 TDW-MAT 的原有技能执行器先根据 high-level plan 计算“本轮 proposed low-level action”，但此时尚未调用 `env.step()`，所以还没有真实环境副作用。Coordinator 收齐双方 proposal、规范化 intent 并完成 Atomic Validator 后，才把 final joint actions 提交环境。

移植时推荐保留同样的两阶段语义：

```text
skill/executor dry step -> proposed action -> atomic validation -> environment execution
```

如果目标 benchmark 的 executor 无法在不产生副作用的情况下给出 proposal，则应把 Validator 前移到 high-level intent 上；无论采用哪种实现，都必须保证发生资源冲突的两个动作不会先后真实执行后才仲裁。V4 不要求替换导航、抓取或工具调用实现，优先保持原 executor，以便把协作增益与技能改动分离。

### 15.7 Budget adapter

统一抽象为：

```json
{
  "current": 100,
  "remaining": 900,
  "unit": "step|frame|second|turn|tool_call"
}
```

当前 Core 只公开原始 budget 事实，不据此强制策略。若加入 deadline safety，应作为可关闭插件。

## 16. TDW-MAT / CoELA Adapter

本节全部属于 **[TDW-MAT]**。

### 16.1 Planner 与 executor 仍然存在

V4 没有绕过 `lm_agent`：

- `lm_agent.act()` 更新局部视觉/语义记忆；
- `LLM.run()` 消费局部状态、历史和 Dashboard；
- `get_available_plans()` 编译 CoELA 候选；
- `gotoroom()`、`goexplore()`、`gograsp()`、`goput()` 执行技能。

`peer_consult.py` 是协作层，不是完整 agent 替代品。

### 16.2 原版执行器隔离

V4 明确使用 upstream CoELA：

- `gotoroom()`；
- `goexplore()`；
- 原有 `AgentMemory.move_to_pos()`；
- 原 grasp/put/transport。

V3.3--V3.5 waypoint exploration、增强导航和恢复策略不在 V4 路径中。

### 16.3 Planning-boundary marker

部分 CoELA executor 会在返回第一个底层动作时立刻清除高层 plan。`lm_agent` 因此暴露：

```text
_peer_consult_planning_boundary
_peer_consult_selected_plan
```

Coordinator 用它区分新的 LLM 选择与同一 plan 的后续低层动作。

### 16.4 TDW 权威事实

- `held_objects` / `oppo_held_objects` 用于 physical ownership；
- `get_delivered_objects()` 用于 evaluator-confirmed delivery；
- agent-local `satisfied` 不能把“peer 正拿着”误当成“已送达”；
- episode reset 清空 `object_per_room`、`object_list`、target cache 和 Dashboard；
- frame 被映射为 Dashboard budget；
- room name/ID 被映射为 exploration task/reservation。

### 16.5 当前协议入口

```text
--peer_consult
--peer_review_mode deterministic
--peer_consult_protocol PeerConsultV4
```

V4 与 V3.x 使用不同 protocol gate，避免历史实验行为被静默改变。

## 17. 日志、复现与可审计性

每次 run 至少记录：

- protocol 和 schema version；
- run instance ID；
- episode ID/epoch；
- model/backend ID；
- benchmark/dataset identity；
- frame/step limit；
- random seed；
- code commit 或工作区 patch identity。

当前 V4 JSONL 事件：

```text
reset
decision
outcome
final
```

每个 decision 应保存：

- Blackboard public view；
- 两份 agent-specific decision cards；
- 两个原始 proposal；
- planning boundary 与 task/action identity；
- validator trigger/verdict/reason；
- final actions；
- coordination events。

每个 outcome 应保存 terminal action evidence 和 evaluator progress。

日志必须能回答：

1. 模型原本选择了什么？
2. Harness 是否干预？
3. 干预依据是事实还是策略？
4. 最终执行了什么？
5. 之后是否产生权威任务进展？

## 18. 必须通过的协议测试

迁移实现至少应覆盖以下测试。

### 18.1 Candidate completeness

- 已知目标存在时，合法探索仍可见；
- shared task queue 缺少本地目标时，不得抹掉本地合法候选；
- 非法 embodiment 动作必须删除。

### 18.2 Privacy boundary

- peer 未 commitment/publish 的私有发现不进入当前 agent card；
- 原始 observation、pose、距离和 hidden reasoning 不进入公开事件。

### 18.3 Task persistence

- task switch 将旧任务 suspended，而不是删除；
- communication 不改变 active task；
- executor 内部 `plan=None` 不被误判成显式 release；
- suspended task 可以再次选择。

### 18.4 Atomic claims

- 同一资源的两个 proposal 最多批准一个；
- 已有有效 claim 保持所有权；
- tie-break 不永久偏向固定 agent；
- loser 不进入实际冲突执行；
- claim 在 ownership/completion/switch/lease expiry 时释放。

### 18.5 Loop guard

- 低层 ongoing steps 不计作 planner retry；
- 相同 task/action 无进展重复达到阈值时 arm guard；
- 下一 boundary 选择不同 action 会消费 guard 而不拒绝；
- 下一 boundary 重复相同 action 只拒绝一次；
- task progress version 增加会重置历史；
- 不产生长期 cooldown。

### 18.6 Communication

- planner 先决定是否通信；
- 只能选择允许公开的 event ID；
- 无效/过期 event ID 不得作为自由文本发送；
- renderer 不包含私有 observation 或 reasoning。

### 18.7 Completion and isolation

- agent 对话不能标记 completion；
- peer-held 不等于 delivered；
- episode reset 后旧实体、task、claim、guard 和 Dashboard 不可泄漏。

### 18.8 Executor isolation

- 切换 V4 不应改变 benchmark 原有 skill executor；
- 关闭 V4 后 baseline/V3.x 行为保持原样。

TDW-MAT 当前定向测试：

```text
python -m unittest \
  tests.test_v4_candidate_completeness \
  tests.test_v4_executor_isolation \
  tests.test_peer_consult_v4_loop_guard
```

## 19. 当前 TDW-MAT 经验结果（不是 Core 规范）

**[TDW-MAT empirical, 2026-08-26]** 最新已完成的 22 个 V4 episode 中：

- 23,389 个环境 decision steps；
- 1,211 次真正 planning boundaries；
- 366 次 acquire planning boundaries；
- 严格“双方同一 boundary 同时首次选择同一物体”：0 次；
- 一方正在追踪、另一方新选择同一目标：39 次，涉及 16/22 个 episode；
- 36 次由 `duplicate_claim` 拦截；
- 3 次先由 `planning_loop_guard` 改写；
- 44 次对 peer 已持有物体的 acquire proposal 被 `ownership` 拦截；
- 最终并发抓取同一 object ID：0 次；
- 268 个唯一 terminal grasp evidence 中记录失败：0 次；
- room assignment review：138 次。

这说明 object-level 原子互斥在当前样本中能够阻止实际重复抓取，但主要是 proposal 后的事后仲裁。它仍可能浪费 loser 的一个环境动作，并可能让模型之后再次提出相同冲突。

这些数字只用于诊断当前 TDW-MAT/Qwen3-VL-4B 运行，不是跨模型置信区间，也不应变成其他 benchmark 的固定阈值。

## 20. 已知限制

1. **事后而非完全事前分工。** Agent 可以先提出冲突 proposal，再被 Harness 否决。
2. **不同实例的同类型冗余未阻止。** 若 goal 只需一个 apple，但两个 agent 选择不同 apple ID，object claim 不认为冲突；V4 不运行 goal-type quota governor。
3. **Loser 会损失一个动作。** 当前 TDW-MAT 用安全转向替换被拒 proposal，不在同一环境 step 再调用 LLM。
4. **Room reservation 仍较频繁触发。** 需要在其他场景验证离散 room scope 是否合适。
5. **Priority 可能产生位置偏置。** 当前是提示而非强制效用函数，仍需 candidate-order 消融。
6. **Progress adapter 是关键风险。** 错误的 heartbeat 会误触发或掩盖 loop guard。
7. **集中式 Blackboard 是工程选择。** 分布式真实机器人需要事务和延迟处理。
8. **结构化通信表达能力有限。** 它提高可信度，但不能表达所有复杂协商。
9. **当前 V4 不保证 deadline safety。** 若 benchmark 需要硬安全截止，应实现独立 safety kernel 并做消融。

## 21. 推荐迁移步骤

另一个 Codex agent 应按以下顺序实现，不要从复制 TDW 启发式开始。

### Phase 1：识别 benchmark 不变量

1. 找到 agent 局部 observation 和私有 memory。
2. 找到高层 planning boundary。
3. 找到原有合法 candidate compiler。
4. 找到 skill executor 和 terminal outcome。
5. 找到 authoritative completion truth。
6. 定义可 claim 的具体资源和可 reservation 的探索 scope。

### Phase 2：实现最小 Core

1. 实现 episode-isolated Blackboard。
2. 实现 stable task/action identity。
3. 实现 persistent task lifecycle。
4. 实现 agent-specific Dashboard 和隐私过滤。
5. 保留完整合法候选。
6. 实现 atomic object/resource claim。
7. 实现 factual validator。
8. 实现 task progress version 和 one-boundary loop guard。
9. 接回原有 executor。

### Phase 3：实现通信与日志

1. 将 communication 放入普通候选。
2. 建立允许公开的 lifecycle event schema。
3. 实现 event-ID selection 和固定 renderer。
4. 记录 proposal、validator、final action 和 outcome。

### Phase 4：协议验证

1. 先写本节第 18 章的 deterministic tests。
2. 再做单 agent baseline parity。
3. 再做双 agent scripted collision tests。
4. 最后运行完整 LLM benchmark。

### Phase 5：策略实验

只有最小 Core 稳定后，才逐个加入并独立消融：

- priority 数值与候选顺序；
- room reservation；
- claim 成本；
- richer structured publish；
- deadline safety kernel；
- quota advisory；
- domain-specific navigation recovery。

任何插件都不得改变“代码已启用哪些策略”这一事实的可审计性。

## 22. 推荐跨 Benchmark 指标

除任务得分外，至少报告：

- completion rate / score；
- episode cost、step 或 wall-clock；
- LLM calls、tokens 和通信成本；
- acquire proposal 数；
- duplicate-claim proposal 数；
- ownership conflict 数；
- validator rejection rate；
- repeated conflict per unique resource；
- loop guard arm/apply 数；
- suspended task resume rate；
- claim lease duration；
- final physical conflict 数；
- forced-action rate；
- completion-oracle disagreement 数；
- episode state leakage 测试结果。

建议至少做以下消融：

```text
V4 Core vs no Harness
atomic claim on/off
task persistence on/off
loop guard on/off
structured communication on/off
canonical/reversed/randomized candidate order
room reservation on/off（若适用）
optional policy plugin on/off
```

## 23. 当前代码地图

### 协作层

`tdw_mat/tdw-gym/peer_consult.py`

- `_intent_from_plan()`：CoELA plan -> public intent
- `TDWSharedBlackboard.observe()`：事实、ownership、completion、progress
- `sync_tasks()`：persistent task lifecycle
- `task_queue()` / `_decision_view_v4()`：Dashboard
- `claim()` / `claim_room()`：resource coordination
- `_v4_*loop*`：one-boundary loop guard
- `_review_and_govern()`：factual validator
- `_reconcile_messages_v4()`：structured communication
- `TDWPeerConsultCoordinator.act()`：完整协调循环

### Planner

`tdw_mat/LLM/LLM.py`

- `get_available_plans()`：完整合法候选和 active continuity ordering
- `run()`：Dashboard/history/prompt/plan
- V4 communication event selection

### Agent 与执行器

`tdw_mat/tdw-gym/lm_agent.py`

- `LLM_plan()`：局部 planner 调用
- planning-boundary adapter marker
- V4 executor isolation
- original CoELA navigation/exploration/grasp/put/transport

### 环境入口

`tdw_mat/tdw-gym/challenge.py`

- protocol CLI
- Coordinator construction/reset/act/outcome/finalize

### 测试

```text
tdw_mat/tests/test_v4_candidate_completeness.py
tdw_mat/tests/test_v4_executor_isolation.py
tdw_mat/tests/test_peer_consult_v4_loop_guard.py
```

## 24. 最终复现检查表

在声称另一个 benchmark 已复现 PeerConsult V4 前，必须能够回答“是”：

- [ ] 每个 agent 是否仍由自己的局部 planner 决策？
- [ ] 是否没有中央策略模型替双方分配具体答案？
- [ ] Dashboard 是否 agent-specific 且不自动泄漏 peer 私有 observation？
- [ ] 合法候选是否完整，策略偏好是否没有伪装成 legality？
- [ ] task ID 是否跨 planning boundary 稳定？
- [ ] task switch 是否 suspend 而非删除？
- [ ] object/resource claim 是否在执行前原子决定？
- [ ] completion 是否只来自权威 oracle？
- [ ] factual validator 是否不替 loser 指定策略？
- [ ] loop guard 是否监控 planning boundary 而非低层 step？
- [ ] guard 是否只排除完全相同动作一次，而非长期 cooldown？
- [ ] task progress 是否由 adapter 显式、单调地报告？
- [ ] communication 是否 decision-first 且只发送允许公开的结构化事件？
- [ ] 原有 skill executor 是否保持隔离？
- [ ] episode reset 是否清空所有跨 episode 协作状态？
- [ ] 日志是否能重建 proposal -> validation -> action -> outcome？
- [ ] TDW-MAT 特有规则是否被放在 Adapter/插件而不是冒充 Core？

满足这些条件后，才可以在新的场景、任务或模拟器中把实现称为 PeerConsult V4，而不是仅仅借用了 Dashboard 或 claim 的局部思想。

## 附录 A：当前 V4 Blackboard 实现级字段清单

本附录描述当前 `TDWSharedBlackboard` 对象实际保存的内容。需要区分四类寿命：

```text
episode ledger：整个 episode 持续累积或更新
current snapshot：每次 observe 重建或每轮覆盖
leased state：带显式释放条件或 TTL
bounded history：只保留最近有限条记录
```

Blackboard 对象每个 episode 重新创建，不跨 episode 继承。它是 Coordinator 内部的中央事实对象，不等于直接交给 LLM 的 Dashboard，也不等于日志中的 `public_view()`。

### A.1 核心事实与物理状态

| 字段 | 内容与来源 | 寿命 | V4 用途/是否直接进入 Dashboard |
|---|---|---|---|
| `frame` | 两份 observation 中 `current_frames` 的最大值 | current snapshot | 生成 `frame_budget`、TTL 和事件时间 |
| `episode_epoch` | Coordinator reset 时递增 | episode | 隔离运行和日志；不作为普通策略输入 |
| `goal_ledger` | `required{name:count}`、evaluator-confirmed `delivered{id:name}`、派生 `remaining` | episode ledger，送达单调增加 | Dashboard 只公开计数，不公开伪造的 planner completion |
| `known_entities` | `id/name/type/category/seen_by/last_seen_by/last_seen_frame/position/room` | episode ledger，重复观测覆盖 | 中央 Harness 可验证；V4 默认不整体公开，只经 task/identity 投影 |
| `physical_owners` | `object_id -> agent/arm/carrier/container_id` | 每次 `observe()` 清空后按 `held_objects` 重建 | 生成双方 payload、拒绝重复抓取、确认物理进展 |
| `container_contents` | `container_id -> child object IDs` | 每次 `observe()` 重建 | 识别容器内 payload 和释放 ticket；不直接进入默认 Dashboard |
| `room_memory` | `visited_by/coverage_by_agent/last_visit_frame` | episode ledger | V4 只用 coverage=`all` 完成探索任务；不公开 coverage heuristic |
| `payload_states` | 当前 payload、loaded containers、started/updated frame | 有 payload 时更新，无 payload 时删除 | 当前 V4 Dashboard 改由 `physical_owners` 直接生成简化 payload；该字段主要服务旧策略 |
| `evaluator_progress` | `observe_outcome()` 写入的 evaluator tuple | current snapshot | 只用于审计；planner 相关完成事实必须先进入 `goal_ledger` |

`known_entities` 的当前来源包括：

1. `state.visible_objects` 的直接观察；
2. `state.held_objects` 的物理持有事实；
3. 两个 agent 已维护的 `object_per_room` 语义缓存。

因此中央 Harness 内部可能知道双方符号实体的并集，但 Dashboard compiler 必须再次执行隐私过滤。Harness 内部可见不等于 peer planner 可见。

### A.2 Persistent tasks 与 commitment

`tasks` 中每条任务的通用字段为：

```json
{
  "task_id": "entity:42",
  "kind": "deliver_goal_object",
  "status": "in_progress",
  "priority": 100,
  "owner": 0,
  "last_owner": 0,
  "object_id": 42,
  "name": "mouse",
  "room": "<Kitchen> (1000)",
  "created_frame": 420,
  "updated_frame": 600,
  "attempts": 1,
  "blocked_until": null,
  "reason": "selected_by_planner"
}
```

当前 TDW-MAT task kind 和 fixed priority：

| Kind | Stable ID | Priority | 事实来源 |
|---|---|---:|---|
| `deliver_payload` | `delivery:<agent_id>` | 130 | 当前 physical ownership 中存在 goal payload |
| `deliver_goal_object` | `entity:<object_id>` | 100 | 已知且 evaluator 仍需要的目标实例 |
| `container_resource` | `entity:<object_id>` | 40 | 已知容器实例 |
| `explore_room` | `room:<room_identity>` | 20 | 已知房间/探索 scope |

相关字段：

- `tasks`：episode 内持续存在；切换或拒绝通常变成 `suspended`，不删除；
- `active_tasks`：`agent_id -> task_id | None`，表示当前公开 commitment；
- `task_history`：记录 task switch，最多 64 条；当前 V4 card 不读取；
- `current_intents`：每轮收集 proposal 后整体覆盖，记录 `stage/target/plan/action/execution`；它不可能进入同轮已经生成的 Dashboard；
- `proposal_history`：每轮保存原 proposal、planning boundary、Validator 和 final action，最多 32 条，只用于审计。

### A.3 Claims 与 reservations

Object claim：

```json
{
  "42": {
    "agent": 0,
    "frame": 600,
    "reason": "accepted_acquire_intent",
    "kind": "goal_object"
  }
}
```

Room claim：

```json
{
  "1001": {
    "agent": 1,
    "frame": 600,
    "plan": "go to <Bedroom> (1001)"
  }
}
```

| 字段 | 建立 | 释放 |
|---|---|---|
| `claims` | Atomic Validator 接受 acquire intent 后 | ownership、completion、切换具体目标、task release、相关 review/guard、450-frame lease expiry |
| `room_claims` | V4 room-reservation Validator 接受探索 scope 后 | owner 放弃 scope、改选房间、相关 review/guard、450-frame lease expiry |

Claim 是“准备获取资源”的临时互斥承诺；physical owner 是环境确认已经持有；completed 是 evaluator 确认任务完成。三者不得混用。

### A.4 Progress、失败与 loop guard

| 字段 | 含义 | V4 消费方式 |
|---|---|---|
| `execution_evidence` | terminal action 的 `action_id/type/status/valid/completed_frame` | 最多 64 条；loop monitor 逐 ticket 消费 |
| `_seen_action_evidence` | 已记录 ticket 集合 | episode 内去重，不作为 prompt |
| `task_progress_versions` | `task_id -> monotonic version` | ownership、evaluator delivery、room task completion时递增 |
| `planning_loop_guards` | agent 对应的一次性 `{task_id, action_identity, reason, count, armed_frame}` | 只作用于下一个真正 planning boundary |
| `failure_counts` | 连续 terminal action 成功/失败 telemetry | V4 card 不显示，也不运行旧 navigation spatial-progress recovery |

当前 guard 的核心不是“失败三次后冷却目标”，而是：同一 agent 在真正 planning boundary 上连续两次提出相同 task/action、且 task progress version 没有增加时，arm 一次 guard。下一次 boundary 只排除完全相同的 action identity 一次，随后立即消费；原任务以后仍可重新选择。

### A.5 事件、消息和审计历史

| 字段 | 内容 | 上限/公开方式 |
|---|---|---|
| `coordination_events` | task/claim/failure/release/progress/completion/guard 生命周期 | 最多 64；Dashboard 放最近 12 |
| `reviews` | `reviewer/proposer/trigger/verdict/reason/replacement` | 最多 32；每个 Dashboard 只放与本人相关的最近 2 条摘要 |
| `dialogue_events` | 从 observation message vector 获取并按 sender 连续去重 | 最多 24；Dashboard 放最近 4 条、每条截断到 240 字符 |
| `proposal_history` | 原 proposal 到 final action 的完整链 | 最多 32；不进入 Dashboard |

V4 可公开通信事件类型为：

```text
task_commitment
task_suspended
task_released
task_failure
task_progress
task_completed
claim_acquired
claim_released
```

Planner 只能选择一个允许公开的 event ID，Harness 再渲染固定 JSON；自由文本不能修改 Blackboard 事实。

### A.6 当前纯 V4 正常路径中的遗留/预留字段

以下字段仍存在于共享类中，但不应误认为当前 V4 Core 已启用：

| 字段 | 当前状态 |
|---|---|
| `container_states` | V3.4/V3.5 的 parked/delivery-pending 生命周期；纯 V4 的 semantic flags 为 false，正常为空 |
| `target_cooldowns` | 旧版按 frame 冷却；V4 正常路径不创建，使用 one-boundary guard |
| `delivery_failure_counts` | 旧版 delivery retry 计数；V4 reconciliation 不增加 |
| `peer_support_requests` | 当前仅初始化，无实际读写 |

## 附录 B：Dashboard、Candidates、Memory Board 与 Governor 的完整例子

### B.1 Dashboard 从哪里来

V4 `_decision_view_v4()` 是确定性视图编译器，输入可分为三类：

```text
Blackboard facts
  goal/tasks/claims/ownership/events/reviews/dialogue/loop guard

当前 agent/adapter facts
  self.current_room
  current frame 与 max frame

固定协议 guidance
  候选合法性、局部自主决策、claim 语义等提示
```

默认输出字段：

```text
frame_budget
goal
self(room, payload)
peer(payload, commitment)
active_task
task_queue
active_claims（只列其他 agent 的 object claim）
planning_loop_guard
coordination_events
recent_coordination_events（当前与上一字段内容重复，均取最近 12 条）
recent_relevant_reviews
recent_agent_dialogue
coordination_guidance
```

它不是底层导航 Dashboard。虽然函数接收当前 state，但 V4 task queue 分支不计算距离；默认卡片不包含 pose、object position、地图、path、waypoint、碰撞、navigation progress、coverage score 或 deadline estimate。唯一位置例外是可选的 TDW-MAT shared-delivery-target adapter。

### B.2 Agent-specific 过滤

为 Agent A 生成 card 时：

- `self` 指 A，`peer` 指 B；
- `active_task` 只显示 A 当前任务；
- `peer.commitment` 只显示 B 已公开的当前任务；
- `active_claims` 只列其他 agent 的 claim；
- `task_queue` 排除 peer-owned/peer-claimed task；
- V4 还隐藏“从未由 A commitment、可能只来自 B 私有发现”的 unowned pending task；
- task queue 最多 10 条，按 active、suspended、fixed priority、stable task ID 排序。

两张 card 都在本轮任一 agent 产生 proposal 前编译。因此 Agent 1 不会在同轮 card 中看到 Agent 0 刚选择的新 plan；同轮冲突由后置 Atomic Validator 处理。

### B.3 Dashboard 对候选的影响

候选主来源仍是 agent 本地规则编译器：

```text
local object_list
+ holding state
+ rooms/current_room/rooms_explored
+ embodiment/role
-> complete high-level candidates
```

Dashboard 的直接机械影响限于：

1. active task continuity ordering；
2. guard 取消 continuity bonus，但不删除候选；
3. 可选 shared delivery target 启用 transport 或提前 bed-room 候选。

`task_queue`、peer claim 和 peer payload 的主要作用是给 LLM 语义判断，并给后置 Validator 提供事实。V4 不把共享 task queue 当本地目标白名单。

### B.4 具体场景

假设：

- goal 是 1 个 apple 和 1 个 mouse；
- Alice 在 Kitchen，active task 是 mouse 42；
- Bob 已 commitment 并 claim apple 55；
- Alice 的私有 `object_list` 同时包含 mouse 42、apple 55 和 bowl 50；
- 当前 frame=1200，剩余 1800 frames。

Alice 的简化 Decision Card 为：

```json
{
  "frame_budget": {"current": 1200, "remaining": 1800},
  "goal": {
    "delivered_total": 0,
    "required_total": 2,
    "remaining": {"apple": 1, "mouse": 1}
  },
  "self": {
    "room": "<Kitchen> (1000)",
    "payload": []
  },
  "peer": {
    "payload": [],
    "commitment": {
      "task_id": "entity:55",
      "kind": "deliver_goal_object",
      "status": "in_progress",
      "priority": 100,
      "owner": 1,
      "last_owner": 1,
      "object_id": 55,
      "name": "apple"
    }
  },
  "active_task": {
    "task_id": "entity:42",
    "kind": "deliver_goal_object",
    "status": "in_progress",
    "priority": 100,
    "owner": 0,
    "last_owner": 0,
    "object_id": 42,
    "name": "mouse",
    "room": "<Kitchen> (1000)",
    "reason": "selected_by_planner"
  },
  "task_queue": [
    {
      "task_id": "entity:42",
      "kind": "deliver_goal_object",
      "status": "in_progress",
      "priority": 100,
      "object_id": 42,
      "name": "mouse",
      "room": "<Kitchen> (1000)",
      "reason": "selected_by_planner",
      "blocked_until": null
    }
  ],
  "active_claims": {
    "55": {"agent": 1, "kind": "goal_object"}
  },
  "planning_loop_guard": null,
  "coordination_events": [
    {
      "event_id": 7,
      "frame": 1190,
      "event": "claim_acquired",
      "agent": 1,
      "task_id": "entity:55",
      "claim_scope": "object",
      "claim_id": 55
    }
  ],
  "recent_relevant_reviews": [],
  "recent_agent_dialogue": []
}
```

若当前 communication slot 可用，本地规则编译器仍可能生成：

```text
A. send a message
B. go grasp target object <mouse> (42)
C. go grasp target object <apple> (55)
D. go grasp container <bowl> (50)
E. go to <Bedroom> (1001)
F. explore current room <Kitchen> (1000)
```

这里体现了层次边界：

- active mouse task 把 B 放在物理候选前部；
- Bob 的 claim 没有直接删除 Alice 的本地候选 C；
- LLM 应根据 peer commitment/claim 选择 B，而不是 C；
- 如果 LLM 仍选择 C，Atomic Validator 以 `claim_conflict` 拒绝；
- Validator 不会替 Alice 自动选择 B、D、E 或 F。

### B.5 Memory Board 与 Prompt

`prompt_summary()` 会再次编译 agent-specific view，将它序列化为单行紧凑 JSON，并添加：

```text
[Memory Board] { ... }
```

Coordinator 会删除 `dialogue_history` 中旧的 Memory Board，只保留最新一张。与此同时，结构化 Decision Card 仍保存在 `llm.peer_decision_card`，所以存在两条使用路径：

```text
structured decision card
  -> candidate continuity / optional shared delivery target

text Memory Board
  -> LLM semantic decision
```

当前 TDW-MAT `LLM.run()` 的主要规划上下文为：

```text
goal
+ local progress（room/exploration/holding/satisfied/peer-held 等）
+ 最近 10 条 action history
+ 最近 6 条自然 dialogue
+ 最新 1 条 Memory Board
+ V4 protocol guidance
+ available high-level actions
```

Dashboard 每个 Coordinator step 刷新，但 LLM 只在真实 planning boundary 消费。低层 action 为 `ongoing` 或高层 plan 尚未结束时，agent 继续原计划。

### B.6 Governor 的“可行性”边界

V4 Governor 更准确地称为 Atomic Factual Validator。它检查：

1. one-boundary loop guard；
2. evaluator-confirmed completion；
3. physical ownership；
4. existing object claim；
5. same-step duplicate object claim；
6. acquire 时是否还有操作资源/空手；
7. room exploration reservation；
8. communication event ID 和 event type。

它不检查：

- 路径是否真的可达；
- 障碍是否能绕开；
- 导航是否卡住或两个 forward action 是否碰撞；
- 哪个目标最近或最有价值；
- deadline 是否要求运输；
- payload/container 策略是否最优；
- 模拟器是否一定接受动作。

通过时，proposal 通常保持不变并进入 final joint actions，随后由 `env.step()` 真正执行；执行仍可能失败。拒绝时，当前 TDW-MAT 会清除 plan/target、将任务保留为 suspended、按触发原因释放 claim，并把本轮动作替换为安全转向；下一 planning boundary 才由 LLM 重新选择。

同帧双方都 acquire mouse 42 时，Validator 先收齐两份 proposal，再通过轮换式 deterministic tie-break 只授予一个 claim。Winner 的 proposal 进入环境；loser 被 `duplicate_claim` 拒绝并在下一轮重规划。

## 附录 C：迁移到另一个 Benchmark 时的必读文件与逐项对照

本节面向接手迁移任务的另一个 Codex Agent。不要从复制整个 `peer_consult.py` 开始；先理解 V4 Core 与 TDW-MAT Adapter 的边界，再在目标 benchmark 中寻找等价接口。

### C.1 必读优先级

#### 第一组：必须完整理解

| 当前文件 | 必须阅读的部分 | 要理解的问题 |
|---|---|---|
| [`tdw-gym/peer_consult.py`](../tdw-gym/peer_consult.py) | `TDWGoalLedger`、`TDWSharedBlackboard`、`observe`、`sync_tasks`、`task_queue`、`_decision_view_v4`、claim/room claim、loop guard、`_review_and_govern`、`TDWPeerConsultCoordinator.act/observe_outcome` | V4 Core 的事实生命周期、原子顺序和隐私投影 |
| [`LLM/LLM.py`](../LLM/LLM.py) | role legality、`get_available_plans`、`_v4_continuity_task`、`_prefer_task_plan`、`shared_delivery_target`、`run`、V4 communication selection | 本地候选、Dashboard 的直接影响、Prompt 窗口、LLM 输出 |
| [`tdw-gym/lm_agent.py`](../tdw-gym/lm_agent.py) | `get_object_list`、`reset`、原导航/探索/抓取/运输方法、`LLM_plan`、`act`、planning-boundary markers | 私有 memory、何时真正重规划、高层 plan 如何变成 proposed low-level action |
| [`tdw-gym/challenge.py`](../tdw-gym/challenge.py) | episode reset、Coordinator 构造、`coordinator.act`、`env.step`、`observe_outcome`、protocol CLI | 环境闭环、真实副作用边界、episode isolation |

建议按上述顺序阅读，而不是只读 `peer_consult.py`。只读协作层容易错误地认为 V4 已经替换 `lm_agent`，或者误把 Governor 放在原执行器之前。

#### 第二组：理解 Adapter 与 baseline 隔离

| 当前文件 | 对照目的 |
|---|---|
| [`tdw-gym/agent_memory.py`](../tdw-gym/agent_memory.py) | 确认地图、路径、pose、`move_to_pos` 属于私有原执行器，不属于 Dashboard/Core |
| [`tdw-gym/tdw_gym.py`](../tdw-gym/tdw_gym.py) | observation schema、action type、`get_obs`、`step`、`check_goal`、`get_delivered_objects` |
| [`LLM/prompt_com.csv`](../LLM/prompt_com.csv) | 基础 CoELA prompt placeholder 和 communication 模式；不要把全部模板误认为 V4 Core |
| `D:/Desktop/AgentRob/CoELA-master_copy/CoELA-master/tdw_mat/LLM/LLM.py` | 原版 planner baseline，对照哪些候选/Prompt 行为是 V4 新增 |
| `D:/Desktop/AgentRob/CoELA-master_copy/CoELA-master/tdw_mat/tdw-gym/lm_agent.py` | 原版 agent/executor baseline，验证迁移没有把技能改动误当协作收益 |
| `D:/Desktop/AgentRob/CoELA-master_copy/CoELA-master/tdw_mat/tdw-gym/challenge.py` | 原始环境循环，对照 Coordinator 插入位置 |

Baseline copy 仅用于差异审计，不应该成为新 benchmark 的复制来源。迁移目标是保留 V4 语义，不是复制 TDW 字符串、frame 常数或 Replicant action type。

#### 第三组：协议测试

| 测试文件 | 必须迁移的断言 |
|---|---|
| [`tests/test_v4_candidate_completeness.py`](../tests/test_v4_candidate_completeness.py) | 私有合法候选不因 Dashboard 缺失而消失；active task 只排序；探索保持可见 |
| [`tests/test_v4_executor_isolation.py`](../tests/test_v4_executor_isolation.py) | V4 不静默修改原导航/探索/技能执行器 |
| [`tests/test_peer_consult_v4_loop_guard.py`](../tests/test_peer_consult_v4_loop_guard.py) | 真 planning boundary、progress heartbeat、一次性 guard、atomic conflict |
| [`tests/test_episode_isolation.py`](../tests/test_episode_isolation.py) | reset 后实体、任务、claim、guard、local cache 不跨 episode 泄漏 |
| [`tests/test_scout_capabilities.py`](../tests/test_scout_capabilities.py) | embodiment/role capability 硬约束（目标 benchmark 有异构 agent 时） |

### C.2 在目标 Benchmark 中必须找到的对应文件

对新 benchmark 先做只读代码地图。至少找到以下类别；实际文件名由目标仓库决定。

| 目标文件类别 | 要定位的函数/对象 | 与 V4 对照什么 |
|---|---|---|
| Episode runner / evaluation loop | reset、main loop、step、done、metrics | 对照 `challenge.py`：Coordinator 应插在 proposal 与真实环境 step 之间 |
| Observation builder | per-agent observation、visible/held/inventory、messages | 对照 `tdw_gym.py:get_obs` 和 Blackboard `observe`；明确 public/private 字段 |
| Local agent | `act/plan/replan/reset` | 对照 `lm_agent.py`：找出真实 planning boundary 和 local cache reset |
| Candidate compiler / action registry | legal skills、参数绑定、role checks | 对照 `LLM.get_available_plans`；保证完整合法候选，不混入策略过滤 |
| Planner prompt/client | prompt assembly、history window、parse | 对照 `LLM.run`；注入最新 agent-specific card，同时保留本地状态 |
| Skill executor/controller | navigate/manipulate/tool-call 的下一动作 | 对照 `lm_agent` executor：区分“计算 proposal”和“产生环境副作用” |
| Action schema/serializer | action ID、arguments、native payload | 为 stable action identity 和 Validator 提供规范化结构 |
| Ownership/inventory API | held objects、locks、tool ownership、capacity | 对照 `physical_owners`、object claim、hand/resource capacity |
| Completion evaluator | success predicate、delivered IDs、task API outcome | 对照 `goal_ledger`；不得相信 planner 自报完成 |
| Progress evidence | skill terminal status、predicate transition、stage result | 对照 `execution_evidence` 和 `task_progress_versions` |
| Communication channel | send/receive、带宽、消息成本 | 对照 decision-first event ID protocol；决定 claim 是否隐式免费 |
| Logging/tests/config | trace、seed、protocol flag、ablation | 对照 decision/outcome JSONL 和 V4 protocol gate |

### C.3 逐模块对照问题

#### Observation / Blackboard

- 什么事实是环境权威事实，什么只是 agent belief？
- ownership、capacity、completion 是否能通过稳定 API 获取？
- 原始图像、地图、pose 是否应继续私有？
- 哪些事实可以进入中央 Harness，但不能进入 peer Dashboard？
- snapshot 字段是每步重建，还是 episode ledger？
- reset 是否清空所有 task、claim、guard 和 local semantic cache？

#### Planning boundary / Candidate

- LLM 是每环境 step 调用，还是一个 plan 完成后调用？
- 一个高层 plan 会展开成多少低层动作？
- 如何产生 stable `task_id` 与 `action_identity`？
- Candidate compiler 是否提供完整合法动作，而不是只给启发式 top-1？
- role/capability/argument binding 在哪里检查？
- Dashboard 信息是否仅作 coordination hint，还是错误地成为私有候选白名单？

#### Proposal / Atomic Validator

- 能否在产生真实副作用前收齐所有 agent proposal？
- 如果本地 executor 先计算低层动作，这个计算是否无副作用？
- 哪些资源必须互斥：object、tool、workspace、API lock、search region？
- 同步冲突如何 deterministic tie-break，是否长期偏向一个 agent？
- loser 是暂停、wait、no-op，还是能在同一逻辑轮安全重规划？
- Validator 是否只检查不变量，而没有偷偷选择目标/路线？

#### Outcome / Progress / Completion

- 什么是 terminal action evidence？是否有稳定 ticket/action ID？
- 什么变化才算 task progress？移动距离是否真的代表语义进展？
- completion oracle 是否独立于 LLM 对话和本地 bookkeeping？
- 执行失败后如何回写 task、review、claim 和 guard？
- 环境异步时，旧 outcome 是否可能错误匹配新 proposal？

#### Dashboard / Communication

- 两个 agent 是否收到不同的 self/peer 投影？
- peer 未公开的私有发现是否被过滤？
- Dashboard 是否意外泄漏坐标、地图、hidden reasoning 或 evaluator label？
- 通信是否先由 planner 选择，再绑定到公开 event？
- 无效/过期 event ID 是否被确定性拒绝？
- 当前 benchmark 的通信是否消耗 action、时间、token 或真实带宽？

### C.4 推荐的目标代码结构

目标 benchmark 不必复用 TDW 类名，但建议把 Core 和 Adapter 分离：

```text
peerconsult_v4/
├── core/
│   ├── schemas.py              # Task, Claim, Intent, Evidence, Review
│   ├── blackboard.py           # episode ledger/snapshots/events
│   ├── dashboard.py            # agent-specific privacy projection
│   ├── task_lifecycle.py       # persistence/suspension/progress
│   ├── atomic_validator.py     # factual conflicts only
│   ├── loop_guard.py           # planning-boundary guard
│   └── communication.py        # event selection/rendering
├── adapter/
│   ├── observation.py          # native obs -> trusted facts
│   ├── candidates.py           # complete legal local candidates
│   ├── identity.py             # task/action/resource IDs
│   ├── execution.py            # proposal -> native final action
│   ├── completion.py           # authoritative oracle
│   └── budget.py               # frame/step/time/tool-call units
├── coordinator.py              # collect -> sync -> atomic validate
└── tests/
    ├── test_privacy.py
    ├── test_candidate_completeness.py
    ├── test_atomic_claim.py
    ├── test_loop_guard.py
    ├── test_episode_isolation.py
    └── test_executor_isolation.py
```

### C.5 迁移时哪些内容必须重写、哪些应保留

| 内容 | 迁移策略 |
|---|---|
| Blackboard task/claim/evidence schema | 保留语义，可重写数据结构 |
| agent-specific privacy projection | 保留不变量，按新 observation schema 重写 |
| atomic collect-before-execute | 必须保留 |
| persistent/suspended task lifecycle | 必须保留 |
| one-boundary loop guard | 保留机制，重新定义 progress evidence |
| structured lifecycle communication | 原则保留；按通信成本/能力适配 |
| TDW object type 0/1/2/3 | 必须重写 |
| `entity:<id>` / `room:<id>` 字符串 | 可替换，但 identity 必须稳定 |
| frame 常数、450-frame lease | 重新标定或改成 step/time lease |
| Replicant action type 0--8 | 必须替换为目标 native action schema |
| bed、container、双手容量 | 只在目标 benchmark 存在等价语义时适配 |
| room reservation | 无离散探索 scope 时可关闭或替换 |
| 原 CoELA prompt 文本 | 不复制；保留信息槽与候选选择约束 |
| 原导航/抓取 executor | 不移植；使用目标 benchmark 原 executor |
| distance/deadline/quota/navigation heuristic | 不属于 V4 Core；如需加入必须作为可关闭插件消融 |

### C.6 交给另一个 Codex Agent 的推荐任务说明

可以直接使用下面的任务描述：

> 阅读 `doc/REPRODUCTION_V4.md`，然后依次阅读 `tdw-gym/peer_consult.py`、`LLM/LLM.py`、`tdw-gym/lm_agent.py`、`tdw-gym/challenge.py` 和列出的定向测试。先为目标 benchmark 建立 observation、planning boundary、candidate、executor、completion 和 resource ownership 的代码地图，不要修改代码。输出 V4 Core 到目标接口的逐项映射、缺失能力和隐私风险。经审阅后，再实现 episode-isolated Blackboard、agent-specific Dashboard、persistent tasks、atomic claims、factual Validator、one-boundary loop guard、结构化通信和 trace。保留目标 benchmark 原有 planner/skill executor；不要复制 TDW 的距离、bed、container、frame 或 action-type 规则。最后移植 candidate completeness、atomic claim、loop guard、episode isolation 和 executor isolation 测试，并分别报告 Core 与任何 benchmark-specific 插件的消融结果。

在开始编码前，接手者至少应提交以下五份只读分析结果：

1. 目标 benchmark 代码地图；
2. public/private/authoritative 信息分类表；
3. stable task/resource/action identity 方案；
4. proposal 收集与真实副作用边界图；
5. V4 定向测试迁移清单。
