# Common V4：三个 benchmark 的测试交接

以下 COHERENT / PARTNR 命令供合作者在自己的 Linux / Bash 环境执行；路径、Python 环境和模型服务使用合作者自己的配置。TDW 提供原生入口及三并发分片说明。本文命令按当前源码核对；COHERENT / PARTNR 尚未在合作者机器上完成端到端验证。

## 交给合作者 Codex 的任务说明

> 请在本机测试下面指定分支的 Common V4。先检查工作区，保留未提交修改；需要时使用独立 checkout，并接入本机已有数据和环境。复用已能运行原 benchmark 的 Python、模拟器和资源配置。根据下面命令设置本机实际模型服务 ID、API 地址、数据和 tokenizer 路径，密钥使用环境变量且不要打印。先运行契约检查和一个 episode 的烟测，成功后运行固定评测集。只修复环境适配问题；不要为提高成绩修改任务、预算、提示词或协作策略。记录 commit、完整启动参数（隐藏密钥）、模型与数据版本、结果目录、完成数、成功率、异常和跳过的检查。基线与新分支使用相同任务、种子、模型和预算，分别报告各 benchmark 的变化；未完成任务单列，不能以已完成子集成功率代表全量成绩。缺少无法自行获得的数据、凭据或 GPU 时，请准确报告阻塞项。

| Benchmark | 仓库 | 测试分支 | 本次实现 commit |
|---|---|---|---|
| TDW-MAT | https://github.com/Lzmlzmlzm2021/CoELA.git | `codex/v4-llm-context-core` | `7a993619ba2d5ebef1336131826b306242ac378e` |
| COHERENT | https://github.com/wyyjwyyj/ICRA2027-multi-agent.git | `codex/v4-llm-context-core` | `ef86ff26c58c9cd4f1c993c8990d3b7ae4d9ae7a` |
| PARTNR | https://github.com/eraser126/partnr-peerconsult.git | `codex/v4-llm-context-protocol` | `19e0ec7af11dd8bb93d6249ffe1c30878b8d7e7c` |

如需新 checkout，可用 `git clone --branch <上表分支> <上表仓库> <本机新目录>`。已有可用环境则由 Codex 安全切换到目标分支。COHERENT 私有仓库需要已有访问权限。PARTNR 本次变更的 PR 基线是 `codex/v4-qwen3-api`，不是旧 V2 的 `main`。

三个仓库均可在仓库根目录执行公共离线检查：

```bash
python -B peerconsult_core/check_manifest.py
python -B -m unittest discover -s peerconsult_core/tests -v
```

调用模型前，配置 `OPENAI_BASE_URL`（包含 `/v1`）和 `OPENAI_API_KEY`。模型变量必须填写服务端实际 ID；同组新旧实现使用相同模型。以下 `$COHERENT_MODEL`、`$PARTNR_MODEL` 等需要预先配置，不能直接保留为空。

## TDW-MAT

完成原仓库 TDW/Unity/资源安装后，可从仓库的 `tdw_mat/` 目录直接运行单任务烟测。`TDW_MODEL` 为本机服务端模型 ID：

```bash
export TDW_MAT_FIX_LM_COMMUNICATION=1
export TDW_MAT_FIX_LM_SATISFIED=1
python tdw-gym/challenge.py \
  --agents lm_agent lm_agent --source openai --lm_id "$TDW_MODEL" \
  --communication --cot --prompt_template_path LLM/prompt_com.csv \
  --peer_consult --peer_consult_protocol PeerConsultV4 --peer_consult_policy llm \
  --data_prefix dataset/dataset_test/ --eval_episodes 0 \
  --max_frames 3000 --max_tokens 256 --t 0.7 --top_p 1 --n 1 \
  --screen_size 256 --no_save_img --port 1071 \
  --output_dir results --experiment_name common-v4 --run_id "$(date +%Y%m%d_%H%M%S)"
```

全量将 `--eval_episodes 0` 改为 `--eval_episodes -1`。如另行开三并发，分别使用上述三个 episode 列表、不同端口和不同 run_id，避免共享同一模拟器端口或输出目录。保持原评测的观测条件与预算；Unity 启动与模型 API 重试配置沿用本机已验证的环境设置。

## COHERENT

在仓库根目录、Python 3.9+ 环境执行。该入口使用符号环境，不需要 TDW 或 Habitat。

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B verify_upstream.py
mkdir -p log
python main.py --env env0 --task 0 --lm_id "$COHERENT_MODEL" \
  --max_tokens 192 --t 0 --top_p 1 --seed 0 \
  --audit_dir "runs/common-v4-smoke-$(date +%Y%m%d_%H%M%S)/audit"
```

Common V4 默认启用，保留默认协作开关。全量 102 个任务可直接使用下列可移植串行入口，不依赖原作者 `/data/user/...` 的批量脚本。它保存每任务输出和退出码，某个任务失败仍继续收集其他任务：

```bash
RUN_DIR="$PWD/runs/common-v4-full-$(date +%Y%m%d_%H%M%S)"
mkdir -p log "$RUN_DIR"
for env in env0 env1 env2 env3 env4; do
  last=19
  case "$env" in env0|env1) last=20 ;; esac
  for task in $(seq 0 "$last"); do
    out="$RUN_DIR/$env/task_$task"
    mkdir -p "$out/audit"
    rc=0
    python main.py --env "$env" --task "$task" --lm_id "$COHERENT_MODEL" \
      --max_tokens 192 --t 0 --top_p 1 --seed 0 --audit_dir "$out/audit" \
      > "$out/stdout.log" 2> "$out/stderr.log" || rc=$?
    printf '%s\n' "$rc" > "$out/exit_code.txt"
  done
done
python scripts/summarize_results.py "$RUN_DIR" --expected 102
```

查看 `summary.json` 和每任务 audit 的 `final` 事件。汇总脚本中的成功率分母是已完成任务，务必同时核对 `completed_tasks == 102`；启动失败或缺少 final 的任务单独报告。串行调度不改变任务内部默认的 LLM 并行规划配置。

## PARTNR

在仓库根目录执行，复用本机已经能运行 PARTNR 的 Habitat 环境、GPU 渲染、数据与场景资源，安装缺项时遵循仓库 `INSTALLATION.md`。新 checkout 也必须接入原来的 `data/` 资源布局，仅指定 episode JSON 不能替代场景资产。不要直接运行仓库内绑定原作者 Slurm、网络地址和绝对路径的提交脚本。

预先配置：

- `PARTNR_MODEL`：远程服务实际模型 ID。
- `PARTNR_QWEN_TOKENIZER_PATH`：本机与所用 Qwen 模型匹配的 tokenizer 目录，包含相应 chat template；远程推理也需要它计算输入 token 预算。必须覆盖配置中的原作者默认路径。
- `PARTNR_DATASET`：本机 `val_mini.json.gz` 的绝对路径，通常对应 `data/datasets/partnr_episodes/v0_0/val_mini.json.gz`。
- `OPENAI_BASE_URL`、`OPENAI_API_KEY`：本机可访问的服务配置。

先执行本地测试和一个 episode 烟测：

```bash
python -B -m unittest discover -s habitat_llm/tests -p 'test_peer_consult*.py' -v
RUN_DIR="$PWD/outputs/common-v4-smoke-$(date +%Y%m%d_%H%M%S)"
python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v4_zero_shot_react_summary \
  hydra.run.dir="$RUN_DIR" \
  habitat.dataset.data_path="$PARTNR_DATASET" \
  +episode_indices='[0]' num_proc=1 \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
  evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
  evaluation.log_data=True evaluation.log_detailed_traces=True
```

全量 `val_mini`：使用新的 `RUN_DIR` 并删除 `+episode_indices='[0]'`，其余参数保持一致。两个 agent 的 chat 配置和 `constrained_generation=False` 覆盖都需要保留；这是远程 OpenAI 兼容接口的启动方式。

结果位于 `$RUN_DIR/results/`，检查 `episode_result_log.csv`、`run_result_log.csv`、`end_result_log.csv` 及详细轨迹。从本机数据集统计预期 episode 数并核对完成数。保留原配置的动作、时间和输入 token 预算，不通过扩大预算掩盖退化。测试中如出现 grammar 检查跳过，需要在报告中注明；公共/单元测试通过不能替代 Habitat 端到端成功率验证。
