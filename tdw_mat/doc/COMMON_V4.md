# Common V4: LLM decisions and auditable execution

`PeerConsultV4` now defaults to `--peer_consult_policy llm` in the challenge
runner. Use `--peer_consult_policy legacy` to reproduce the imported historical
V4 baseline. V3 protocols retain their existing behavior.

The repository includes the same dependency-free `peerconsult_core` package as
the PARTNR and COHERENT implementations. No sibling repositories, absolute
developer paths, external services, or editable core install are required.

## Run and compare

Install the existing TDW-MAT dependencies and Unity build following the parent
README. Keep your existing model, episodes, seed, communication, shared-target
setting, frame budget and native executor options identical. Add these arguments
to the existing `tdw-gym/challenge.py` command:

```sh
--peer_consult --peer_consult_protocol PeerConsultV4 --peer_consult_policy llm
```

For the matched historical run replace only `llm` with `legacy`. Use separate
output directories. The common implementation does not change `max_frames` or
add no-progress termination. Shared delivery-target information remains under
the existing `TDW_MAT_V4_SHARED_DELIVERY_TARGET` flag, disabled by default.

From the repository root, the transaction tests need Python and NumPy only:

```sh
python -m unittest discover -s tdw_mat/tests -p test_peer_consult_common.py -v
```

With the TDW/planner dependencies installed, run the complete local suite:

```sh
python -m unittest discover -s tdw_mat/tests -v
```

## Behavior and modules

* `peerconsult_core/` admits one frozen batch, keeps separate attempts and
  per-agent commitments, protects running leases, records attributed evidence,
  and stores explicit cooperation proposals and participant responses.
* `tdw-gym/peer_consult_common.py` adapts local TDW observations, two hands,
  physical object ownership, concurrent bodies and native action IDs. The
  original coordinator supplies environment bookkeeping and trace transport;
  its pre-review task mutations, room exclusion and loop guard are bypassed.
* `LLM/LLM.py` provides repeat exploration, wait, explicit release and structured
  cooperation messages. There is no active-task candidate promotion in common
  mode. Failed attempts are context, and the LLM can retry or change work.
  An invalid or ambiguous model selection is rejected, with the native wait
  cost recorded; the historical fuzzy/random action replacement is disabled.
* `tdw-gym/lm_agent.py` retains the existing navigator and skill executor.
  Evaluator-scored objects remain in scoring memory and can reappear in local
  perception/candidates when physically available. A prior score is not an
  environment illegality.

A rejected or deferred proposal uses TDW's one-frame wait primitive instead of
silently turning the body. The rejected proposal never becomes a running
attempt; the wait cost remains in native frames. High-level attempts bind every
native subaction by `(agent_id, action_id)`. A successful turn or movement is
recorded as subaction evidence and cannot complete an acquisition/transport
attempt. A new actual planner boundary closes the previous attempt as
`completed_unverified`, or as a failure/verified progress when supported by
native/evaluator evidence. Unobserved goal progress remains unknown.

`wait` preserves the agent's commitment; `release current task` withdraws it
after the previous execution has ended. Neither operation drops held objects.
No whole-room resource lock is created. Physical acquisition/manipulation
resources remain exclusive across running attempts.

## Cooperation and information access

After choosing the native message action, the model may name an existing public
event or emit `coordination_intent:` plus a JSON payload. For example:

```json
{"kind":"propose","participants":[0,1],"description":"Meet to arrange the handoff","conditions":["Both participants agree on the meeting point"]}
```

The recipient chooses `accept`, `decline`, `ready` or `cancel` with the displayed
`intent_id`. Each message uses the existing native communication action and its
normal cost; proposals cannot manufacture peer acceptance or physical readiness.
The common `release_work` message withdraws the sender's work after any running
execution ends; it cannot release another agent's resources.
`common_core` is part of the existing Memory Board prompt, so pending proposals,
execution results and peer commitments survive short dialogue windows. The
information change from adding prospective cooperation is an explicit experiment
factor; it should be ablated separately from removing historical guards.

The old V4 documents describe the imported **legacy** baseline. This file governs
the common mode. Unit tests establish protocol behavior only: simulator episodes,
official success/transport metrics, model cost and non-regression across fixed
per-benchmark budgets still need to be measured.

Pull requests and manual workflow runs execute `.github/workflows/peerconsult-contracts.yml`. This CI checks the shared package and offline adapter contracts without a simulator or model service; it does not evaluate benchmark success rates.
