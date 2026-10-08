# TDW-MAT Simple ReAct

Select `--planner react` in the existing `tdw-gym/challenge.py` entry point.
Each robot independently uses its local observation/memory, the last eight
execution-feedback records and eight received dialogue entries. Each planning
boundary makes one model request returning a short `reason` and a native
`action_id`. Message content comes from the same request; sending consumes native
simulation time. No PeerConsult/V4 board, claims or recovery planner is enabled.

The policy reuses `LLM.get_available_plans` without a coordination card, applies
the explicit message cooldown described below, and keeps
original navigation, exploration, grasp, container, delivery and asynchronous
ongoing-action behavior. The existing Box Scout role excludes manipulation.
This is a text policy over the existing visual frontend, not direct image input.

## Run

From `tdw_mat` in PowerShell, using the established TDW runtime and assets:

```powershell
& 'D:/Desktop/AgentRob/tdw_mat_runtime/envs/tdw_mat/python.exe' -B tdw-gym/challenge.py --planner react --agents lm_agent lm_agent --embodiments replicant replicant --communication --react_api_config D:/Desktop/AgentRob/COHERENT/results/zhongke_v4_server/model_api.json --data_prefix dataset/dataset_test/ --data_path test_env.json --eval_episodes 0 --max_frames 3000 --experiment_name simple_react --run_id smoke --no_save_img
```

After validating a physical smoke episode, use `--eval_episodes -1 --run_id full`
for all episodes. Use a fresh output directory for different policies/models:
the legacy result cache does not validate policy identity.
For carrier/scout use `--agents lm_agent scout_agent --embodiments replicant box`;
the established GT-mask frontend is required for the scout.

`--react_api_config` accepts `model`, `base_url`, and `api_key`/`apikey` or
`api_key_file`. Relative key paths resolve against the config directory. The
config model takes precedence over `--lm_id`. Without a config, `--lm_id`,
`OPENAI_BASE_URL` and `OPENAI_API_KEY` are used.

Defaults: `--react_history 8 --react_max_tokens 256 --react_temperature 0
--react_timeout 180 --react_max_retries 2`. Original `--max_tokens` / `--t`
configure CoELA, not ReAct. `--peer_consult`, `--cot` and HF sources are rejected
with ReAct. The old CSV prompt template is not used.

## Semantics and audit

- Invalid JSON, IDs or messages produce a logged one-frame native wait, without
  an extra repair request or random action. All three JSON fields are required;
  reason requires 1-120 printable ASCII characters, message 1-240. Whitespace-only
  values and extra fields are rejected. Physical actions use message "-".
  Exhausted API retries raise an explicit error.
- Feedback status belongs to the last low-level action, not guaranteed high-level
  success. Held and visible objects are observations. `local_executor_records`
  are explicitly legacy bookkeeping, NOT evaluator-confirmed delivery. Official
  transport rate still comes from the unchanged evaluator.
- Current per-room object memory is retained. Teammate messages do not grant
  privileged executor coordinates or create a global task board.
- Reset clears per-episode planner history/metrics, communication cooldown,
  pending execution evidence, and agent memory cursors.
- Per-episode `react_agent_0.jsonl` / `react_agent_1.jsonl` record exact prompts,
  raw responses, plans, settings, policy version, cooldown state before/after
  decisions, token counters and errors. The next planning
  boundary observes the preceding execution; selecting a plan alone does not
  prove it completed.
- `TDW_MAT_FIX_LM_SATISFIED` remains an existing executor option. Keep the same
  value across compared methods. No evaluator or dataset files were changed.

## Validation

```powershell
& 'D:/Desktop/AgentRob/tdw_mat_runtime/envs/tdw_mat/python.exe' -B -m unittest discover -s tests -p test_react.py -v
```

Tests cover role/carrying candidates, single-call communication, invalid output,
API failure, bounded history/reset, native executor dispatch, the ongoing
no-model-call path, cooldown/release, same-frame replanning, invalid waits,
message-only candidate sets and exact ASCII length bounds. Model responses are
mocked; this is not a physical simulator
smoke run or evidence of benchmark success.


## Structured-output repair (2026-09-12)

The initial repair sent a strict JSON schema with candidate-ID enum and required
reason/action_id/message fields. It requested reason at most 180 characters and
message at most 300 through schema length properties; the compute backend did
not reliably enforce those properties. Three consecutive invalid decisions fail
explicitly rather than spending thousands of model calls on one-frame waits.
The initial repaired campaign and its logs are retained separately.

## Message cooldown and bounded ASCII output (2026-09-12)

The current policy version is `simple_react_message_cooldown_ascii_v1`. It is a
new baseline variant and must use a fresh campaign directory; do not combine its
results with the earlier structured or ctx32k policies.

After a valid message decision, the next planning boundaries temporarily omit
`send a message` whenever any native non-message candidate is available. The
model chooses among the unchanged remaining native candidates. Communication is
restored once a selected non-message plan has been attempted: a later planning
boundary must observe an advanced simulation frame and a native non-message
low-level action. Success is not required. An immediately rejected plan followed
by replanning in the same frame does not release the cooldown, and an invalid
decision's fallback wait does not release it. A model-selected native scout
`wait` does count as a non-message action. If communication is the only native
candidate, it remains available so the filter never creates an empty action set.
There is no target selection heuristic, delivery oracle, or new executor action.

The system prompt explains the rule. Each context contains
`communication_cooldown`, and each audit row records that state plus the state
after the decision. `message_cooldown_decisions` counts boundaries where a
message candidate was withheld. Per-episode reset clears all cooldown state.

Both schema and parser now enforce bounded ASCII strings using
`^[ -~]{1,120}$` for reason and `^[ -~]{1,240}$` for message, including physical
actions. Bounded regex patterns avoid relying on the backend's ignored
`minLength`/`maxLength` constraints and leave room for ordinary English replies
under the unchanged default 256-token output limit. This is not a guarantee for
every tokenization; truncation remains an explicit invalid decision. The output
limit remains configurable through `--react_max_tokens`. Simulation frame
budget, environment, dataset and evaluator semantics are unchanged.

## Resumable full-suite supervisor

`tdw_mat_setup/start_react_supervised.ps1` starts the 24-episode campaign
`SimpleReAct-Qwen3VL4B-full24-supervised-20260912`. Its default is one local
Unity environment, rotating through the three logical shards. Inference uses
the separate three-GPU compute service with a 32768-token context.

Every episode attempt has its own directory under `attempts/episode_XX/`.
Initialization with no log progress for 15 minutes, simulation with no frame
progress for 10 minutes, or a two-hour attempt limit triggers an infrastructure
retry; none of these conditions becomes an episode score. After three failed
attempts the supervisor continues other episodes and lists the affected episode
under `needs_attention`. Diagnose the failure before explicitly raising the
launcher's `-MaxAttempts` value. Restarting preserves verified completed work.

`summary.json` records active, pending and completed episodes and infrastructure
failures. Only validated native terminal results are atomically copied to
`s{episode // 8}/{episode}/result_episode.json`; the adjacent `completion.json`
identifies the complete attempt log and validation evidence. The full transport
mean remains null until all 24 episodes are complete. Planner, dataset and
evaluation settings are bound by a semantic contract; infrastructure changes
are recorded separately as runtime settings.
