# PeerConsult common protocol 0.1.0

This standard-library-only package is vendored identically into the three
benchmark PRs. It has no simulator, model-client, network or installation
dependency. The environment adapter supplies normalized task/action identities,
native legality, physical holding, resource requests and scoped evidence.

Run the contract suite from the repository root:

    python -m unittest discover -s peerconsult_core/tests -v

The runner constructs CoordinationCore(episode_id, agent_ids), freezes all agent
contexts before collecting proposals, calls review once for the complete batch,
starts only accepted attempts, and associates each native result with that
attempt. accepted means scheduled; deferred carries no execution lease. Running
attempts survive any number of peer planning calls. Request cancellation first;
release occurs only on a confirmed terminal outcome. Subaction evidence uses a
non-attempt scope. Unknown effects use completed_unverified and progress=None.

The default protocol never scores task utility, forces recovery, forbids a retry,
locks an exploration room automatically, or equates historical success with a
permanent action ban. Fixed budget and official evaluation live in the native
runner. Snapshot views separate private evidence, public commitments and
participant-only cooperation proposals; declared readiness is not physical fact.

Coordinate intents are explicitly selected and transported by each adapter.
Calling coordinate itself does not provide a free simulator messaging action;
the adapter documents and accounts for its transport. No peer can accept for
another participant. Task, attempt, physical holding and score credit remain
different concepts. Evidence is deduplicated and bound to its episode and
attempt; prior-episode tickets are invalid even if an episode label is reused.
Event IDs must be unique within an agent and evidence scope in an episode: the
same source event cannot be rebound to a later attempt. Before-dispatch factual
revalidation uses reject_before_start, never a fabricated execution failure.

The release_work control intent releases only the sender's work commitment. If
an execution is active, it records a pending request and waits for that attempt's
terminal acknowledgement; it never silently cancels a skill or drops a lease.
Waiting alone does not request release. A runner may withdraw_deferred when an
agent abstains, without disturbing any accepted execution or work commitment.

The core is a single-writer protocol. Collect model calls concurrently if useful,
but mutate it only at the runner's frozen-snapshot/commit boundaries. Do not call
review concurrently from individual planners. Keep all repository copies byte
identical when updating this package; CORE_MANIFEST.json records source hashes.
