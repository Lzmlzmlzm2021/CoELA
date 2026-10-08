# PeerConsult V4 sharing bed：实现与复现说明

本仓库的协作协议仅支持 `PeerConsultV4`。运行时启用 `--peer_consult`，为两个 `lm_agent` Replicant 提供共享任务板、原子协调和默认开启的床位共享。省略该参数仍运行原始 CoELA 基线入口。ReAct 实验保存在独立 Git 分支 `react-snapshot-2026-10-08`。

## 运行边界

- V4 需要 `--agents lm_agent lm_agent --communication --peer_consult`，协议参数可省略；默认且唯一可选值为 `PeerConsultV4`。
- 每个 agent 仍有自己的图像观察、`AgentMemory`、LLM 高层规划器和原始导航、探索、抓取、装箱、运输执行方法。
- 协调器只维护事实、任务和资源约束。它不计算路线，也不替 LLM 挑选目标、房间或运输时机。
- 床位共享是 TDW-MAT 适配能力：共享板已知道带三维位置的 `type=2` 目标时，默认把它放入两位 agent 各自的决策卡；未观察到床位时不伪造位置。

## 模块与交互

| 模块 | 输入 | 输出及下游 |
| --- | --- | --- |
| `tdw-gym/challenge.py` | CLI、环境观察、评估器进展 | 创建两个 agent 和可选 V4 `TDWPeerConsultCoordinator`；每步将环境状态送入协调器，把协调后的动作送回 TDW；把得分事实反馈给协调器。 |
| `tdw-gym/peer_consult.py:TDWSharedBlackboard` | 两个 agent 的可公开符号事实、动作结果、评估器确认的交付 | 维护目标账本、实体、持有关系、任务生命周期、claim、房间预约、执行证据与协作事件；为每个 agent 编译独立决策卡。 |
| `tdw-gym/peer_consult.py:TDWPeerConsultCoordinator` | 决策卡、两个 agent 的计划与动作提案 | 先把卡注入 LLM 和对话上下文，再调用各 agent；对重复 claim、已经交付、物理持有冲突、房间预约及规划循环做事实校验，返回联合动作；记录 JSONL。 |
| `LLM/LLM.py` | 本地感知与历史、合法候选、V4 决策卡 | 让本地 LLM 选高层计划；活动任务仅影响候选排序，完整合法候选仍保留；读取 `delivery_target` 以开放运输候选；消息只可引用结构化协作事件。 |
| `tdw-gym/lm_agent.py` | LLM 计划、本地记忆、共享床位记录 | 把有效的共享床位适配成原始执行器可读取的 `type=2` 本地记录，执行原版 CoELA 技能，并把动作和规划边界交还协调器。 |
| TDW 环境与 evaluator | 协调后的底层动作 | 返回观察、动作状态及已得分物体；只有 evaluator 确认后，目标账本和 `satisfied` 才记为交付。 |

一次典型循环：

1. 协调器读取新观察和上轮执行证据，更新共享板；每个 agent 仅接收自己的精简卡及公共协作事实，不接收同伴原始图像或私有路径。
2. 各本地 LLM 在本地可执行候选中选计划。共享床位有有效坐标时，持有目标的 agent 即使自己未看见床，也可以选择 transport。
3. `lm_agent` 优先保留本地直接观察到的床；若没有，则校验共享卡的 episode 和有限三维坐标，再把床位放入原始 `object_list[2]` 供运输执行器使用。
4. 协调器比较两个提案，原子处理 claim 和资源冲突；拒绝非法动作时清除该计划，使对应 agent 下次重新规划。重复且无进展的相同计划会触发一次规划边界限制。
5. 环境执行动作。协调器只用实际环境结果与 evaluator 确认交付，释放或更新任务、claim，并将事件写入 `peer_consult.jsonl`。

## 共享床位字段与隔离

`delivery_target` 至少提供 `id`、`name`、`type=2`、`position`，有房间时再提供 `room`；还带 `knowledge_source` 和 `episode_epoch`。坐标来自共享板已观察的实体，并非 LLM 猜测。本地 agent 只接受三维有限数值且属于当前 episode 的记录。直接观察到的床优先；旧 episode 的共享床缓存会清除。卡片的共享坐标是此 TDW-MAT 适配器的明确例外；路径、地图与私有视觉仍由各 agent 保存。

## 复现和检查

从 `tdw_mat` 目录运行：`python -m unittest discover -s tests -q`。完整 TDW 轨迹仍需配置场景、模型服务与模拟器；单元测试验证协议、候选集、床位注入、episode 隔离和原始执行器路径，不等价于基准分数复现。

实验命令应在现有 CoELA 双 agent 启动参数基础上增加 `--communication --peer_consult`。若需要明确写出协议，再追加 `--peer_consult_protocol PeerConsultV4`。基线使用相同命令但省略 `--peer_consult`；模型、场景、预算与随机种子应保持一致。
