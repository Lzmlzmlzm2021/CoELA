# PeerConsult V3.5 最小复现与移植说明

本文面向接手实验的 agent。目标是先在 TDW-MAT 中复现当前实现，再把协调层接到真机或其他 benchmark。代码分支为 `feature/tdw-peer-consult`；所有密钥、服务器地址、模型权重和实验输出都应保存在仓库外。

## 1. 当前实现与结果边界

核心入口和组件：

- `tdw-gym/challenge.py`：评测入口和 coordinator 接线；
- `tdw-gym/peer_consult.py`：Memory Board、Decision Card、任务账本、冲突治理和交付确认；
- `tdw-gym/lm_agent.py`：高层计划与 TDW 技能执行；
- `LLM/LLM.py`：OpenAI-compatible / Hugging Face 模型接口；
- `tests/`：协议、episode 隔离、导航一致性和 Qwen 兼容性回归。

2026-08-25 的 24-episode 汇总如下：

| 组别 | Episode | 成功物体/总物体 | TR |
|---|---|---:|---:|
| V3.5，vLLM 原始 17 条 | 0–5、8–13、16–20 | 114/170 | 0.6706 |
| V3.5，LM Studio Q8_0 新 7 条 | 6、7、14、15、21、22、23 | 43/70 | 0.6143 |
| V3.5 混合后端 24 条 | 0–23 | 157/240 | 0.6542 |
| V2 参考结果 | 0–23 | 153/240 | 0.6375 |

不要把 `+4` 个物体解释成纯算法增益：V2 使用 Qwen3.5-4B，V3.5 使用 Qwen3-VL-4B-Instruct，且 V3.5 的 24 条由 vLLM 与 LM Studio 两种后端合并。它是当前工程结果，不是严格控制变量的消融实验。

## 2. 协作核心与 benchmark 适配边界

移植时必须先划清三层。A 层是方法本体，必须保留其语义和闭环；B 层是 TDW-MAT 适配，换 benchmark 或真机时应重写；C 层只是模型服务与部署设施，可自由替换。

| 层 | 内容 | 移植要求 |
|---|---|---|
| A：PeerConsult 协作核心 | 可验证的共享 Memory Board；每个 agent 的有界 Decision Card；持久任务账本与生命周期；两个 agent 独立 proposal；基于证据的 Peer Review/冲突治理；选择性通信；执行终态与 evaluator-confirmed delivery 回写；episode reset 隔离 | **必须保留。**可以换字段名和实现语言，但不能退化成共享自然语言自述、固定分房脚本或无结果回写的一次性规划 |
| B：benchmark/机器人适配 | TDW RGB-D/segmentation/room 字段解析；Replicant 双手和容器格式；TDW action type 0–6；房间 waypoint 与 `AgentMemory` 导航；床的交付范围；Unity build、端口、3000-frame 终止与结果文件 | **按新环境重写。**这些修改用于让方法在 TDW-MAT 可执行，不是 PeerConsult 的必要假设 |
| C：推理与运行设施 | OpenAI-compatible/HF 客户端、LM Studio/vLLM、Qwen thinking 开关、HTTP 超时重试、Tailscale/Funnel、Windows PowerShell 启动 | **按部署替换。**不得影响 A 层的实验标签、证据来源和决策语义 |

### 2.1 必须保留的协作不变量

1. 两个 agent 保留各自观测、局部记忆、规划器和技能执行器；共享的是经过规范化的符号证据，不是完整私有思维链。
2. Memory Board 只把传感器、物理持有状态、动作终态或 evaluator 结果当作事实；agent 的消息和计划只能是 proposal。
3. 每个决策周期分别生成 agent-specific Decision Card，两个 agent 独立提出动作，再由确定性 reviewer 做最小必要修订。
4. object/room claim 有期限；任务可 pending、in progress、suspended、blocked、carried/in use、completed，不能因临时避让或发消息而丢失。
5. 放下动作成功不等于交付成功。只有 benchmark/机器人提供的交付确认才能完成任务；未确认载荷必须进入有界恢复流程。
6. `reset(episode_id)` 必须清空上一 episode 的实体、床/目标位置、claim、cooldown、消息 cursor 和 pending delivery。
7. 通信由新证据/冲突/恢复需求触发，planner 先决定是否沟通，再生成消息内容；通信不能替代物理状态验证。

### 2.2 TDW-MAT 中哪些代码属于哪一层

| 文件/区域 | 协作核心 A | TDW 适配 B / 运行设施 C |
|---|---|---|
| `tdw-gym/peer_consult.py` | Board、Decision Card、task/claim/review、交付确认与 episode 隔离 | 从 TDW state 字典抽取证据、识别 action type、床/容器的 TDW 规则需要由 adapter 替换 |
| `tdw-gym/challenge.py` | coordinator 的 `observe → propose → review → act → outcome` 调用顺序 | Unity/TDW 创建、端口、episode 文件与 CLI 全部属于 B |
| `tdw-gym/lm_agent.py` | 接收 Decision Card、形成高层 proposal、保留 active task | `goexplore`、`gograb`、`putin`、`goputon` 和 AgentMemory 导航属于 B |
| `tdw-gym/tdw_gym.py` | 提供可验证动作终态/交付证据的接口概念 | Replicant、房间 waypoint、TDW action ticket 和底层环境实现属于 B |
| `LLM/LLM.py` | 将有界 Decision Card 纳入 planner 输入的原则 | OpenAI/HF 客户端、模型参数、thinking 与 HTTP 重试属于 C |

判断标准很简单：如果一段代码回答“双方如何共享证据、分配/恢复任务、审查冲突、确认结果”，它属于 A；如果回答“在这个环境里怎么看见、移动、抓取、放置或计分”，它属于 B；如果回答“请求发到哪个模型服务”，它属于 C。

## 3. 安装

推荐 Python 3.9。仅使用 GT mask 和远程 OpenAI-compatible 推理时，不需要本地 Torch/Transformers：

```powershell
git clone --branch feature/tdw-peer-consult https://github.com/Lzmlzmlzm2021/tdw-mat.git
Set-Location tdw-mat
py -3.9 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
& .\.venv\Scripts\python.exe -m pip install -e .
```

当前验证环境的关键版本为 Python 3.9.23、TDW 1.11.23.5、OpenAI 1.42.0、Gym 0.26.2、NumPy 1.24.3。`requirements-api.txt` 固定了 OpenAI 1.42 与兼容的 `httpx==0.27.2`。

先跑无需启动 Unity 的回归：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前提交前基线为 84/84 通过。

## 4. 配置 OpenAI-compatible 模型服务

LM Studio、vLLM 或其他兼容 `/v1/chat/completions` 的服务均可。不要把真实值写入代码或提交 JSON 配置：

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
$env:OPENAI_API_KEY = "local-placeholder"
$env:TDW_MAT_OPENAI_TIMEOUT = "300"
$env:TDW_MAT_OPENAI_RETRY_MAX_TRIES = "12"
$env:TDW_MAT_OPENAI_RETRY_MAX_TIME = "1800"
$env:TDW_MAT_FIX_LM_SATISFIED = "1"
$env:TDW_MAT_FIX_LM_COMMUNICATION = "1"
$env:TDW_MAT_SAVE_TOPDOWN = "0"
```

远程服务优先使用局域网、VPN/tailnet 或稳定反向代理。公网 Funnel 在长实验中曾出现 TLS EOF；重试只能缓解短暂中断，不能替代稳定链路。Qwen3.5 若因隐藏 thinking 吃完输出预算，可额外设置 `$env:TDW_MAT_DISABLE_MODEL_THINKING = "1"`；当前 Qwen3-VL-4B 实验没有启用该开关。

先用模型服务自身的 `/v1/models` 和一次短 chat completion 验证模型名，再启动 TDW。模型名必须与服务返回值一致，例如 `qwen/qwen3-vl-4b`。

## 5. 单 episode 冒烟测试

每次尝试都使用新的 `run_id`。下例由 TDW Python 包自动启动本机 build，运行 E0：

```powershell
$runId = "v35_e0_$(Get-Date -Format yyyyMMdd_HHmmss)"
& .\.venv\Scripts\python.exe tdw-gym/challenge.py `
  --output_dir results `
  --experiment_name PeerConsultV3.5-Qwen3VL4B `
  --run_id $runId `
  --port 1071 `
  --agents lm_agent lm_agent `
  --embodiments replicant replicant `
  --communication `
  --peer_consult `
  --peer_consult_protocol PeerConsultV3.5 `
  --source openai `
  --lm_id qwen/qwen3-vl-4b `
  --prompt_template_path LLM/prompt_com.csv `
  --max_tokens 256 `
  --cot `
  --data_prefix dataset/dataset_test/ `
  --data_path test_env.json `
  --eval_episodes 0 `
  --max_frames 3000 `
  --screen_size 256 `
  --no_save_img
```

若 TDW build 已由其他进程或机器启动，在命令中增加 `--no_launch_build`，并保证 `--port` 与 build 监听端口一致。并行 shard 必须使用不同端口和不同 `run_id`。

完成标志是：

- `results/<experiment>/<run_id>/<episode>/result_episode.json` 存在且可解析；
- 根目录的 `eval_result.json` 写入汇总；
- `peer_consult.jsonl` 含 proposal、review、outcome 和 Memory Board 轨迹；
- `output.log` 没有 502、TLS/SSL、`APIConnectionError` 或 Unity 断连。

不要依据图片目录、残留 frame 或未完成日志判定 episode 完成，也不要把失败尝试的残轨合入正式结果。

## 6. 完整 24 条与结果标签

把 `--eval_episodes 0` 改为：

```text
--eval_episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

建议每个结果在仓库外另存一份 `run_config.json`，至少记录：代码 commit、协议、模型精确名称、量化、上下文长度、推理后端、base URL 的非敏感主机标签、episode、端口、最大帧、采样参数、重试参数和 run ID。不同模型、后端、代码 fingerprint 或重试策略必须使用不同 tag；合并时只按 episode 选择一个已完成结果。

## 7. 移植到其他 benchmark

不要复制 TDW 的运动实现；完整保留第 2.1 节的协作不变量，在新环境重写 B 层适配器。最小接口如下：

```python
class BenchmarkAdapter:
    def reset(self, episode_id) -> tuple[dict, dict]: ...
    def normalized_agent_state(self, agent_id) -> dict: ...
    def verified_deliveries(self) -> dict[int, str]: ...
    def execute_joint_actions(self, actions: dict[str, dict]) -> dict: ...
    def action_terminal_evidence(self, agent_id) -> dict: ...
```

规范化 state 至少应提供：agent 位姿/区域、可见物体 ID/类型/位置、双手持有物、容器内容、动作状态、当前帧和消息。新 benchmark 的高层动作映射到 `navigate/explore/grasp/put_in_container/deliver/send_message/wait`。只有传感器、控制器终态或 evaluator 能更新 ownership、container contents 和 delivered；LLM 自述不能作为真值。

建议按以下顺序移植：

1. 先让单 agent 的观测与技能适配器通过固定脚本测试；
2. 接入 `TDWSharedBlackboard` 的等价证据层；
3. 为每个 agent 生成有界 Decision Card；
4. 接入两个 proposal 与确定性冲突治理；
5. 最后接 evaluator-confirmed delivery、episode reset 和失败冷却；
6. 用同一 seed/episode、模型、prompt、预算和感知条件对比无协调基线。

更详细的方法边界和字段说明见 `doc/PEERCONSULT_V2_PORTING_GUIDE.md`，V3.3 的持久任务与闭环交付设计见 `doc/PEERCONSULT_V33_DESIGN.md`。

## 8. 真机部署附加要求

真机上必须把安全控制放在 LLM/coordinator 之外：急停、速度/力限制、碰撞检测、可达性检查、禁入区、抓取确认和人工接管都由确定性安全层执行。物体 ID 应由 perception/tracking 提供稳定映射；抓取成功应由夹爪、力觉或视觉复核，交付成功应由目标区域检测或任务系统确认。先在数字孪生或低速隔离场地跑固定动作，再启用 LLM 自主规划。

## 9. 已知问题与排查顺序

- TR 低时先检查房间覆盖、目标首次发现帧和是否存在重复抓取，而不是只看最终分数；
- 容器可能放大装载效率，也可能造成反复抓容器、错误交付和重抓已装载物体；检查 `peer_consult.jsonl` 中 ownership、container contents 与 delivery confirmation；
- frame 连续不变时依次检查 Python、TDW build、Unity log 和模型请求；
- 502/TLS/SSL 问题属于模型链路，CUDA fork/vLLM worker 崩溃属于推理服务；两者不要与 TDW 环境错误混为一谈；
- 重新运行必须换 `run_id`，不能让新结果与旧失败目录互相污染。
