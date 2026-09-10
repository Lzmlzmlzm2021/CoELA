"""Crash-safe evaluation artifacts and strict resume validation.

The published TDW-MAT runner wrote JSON files directly and treated any
existing ``result_episode.json`` as completed work.  An interrupted write or
a result copied from a different episode could therefore be silently reused.
This module keeps the legacy format available for upstream callers while the
Human+Box protocol opts into identity-bound, atomically replaced artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid


LEGACY_PROTOCOL = "legacy"
HUMAN_BOX_PROTOCOL = "human_box_v2"
EPISODE_SCHEMA = "tdw_mat_episode_result.v1"
AGGREGATE_SCHEMA = "tdw_mat_eval_result.v1"
SUPPORTED_PROTOCOLS = {LEGACY_PROTOCOL, HUMAN_BOX_PROTOCOL}


class ArtifactValidationError(RuntimeError):
    """An existing evaluation artifact is unsafe to resume."""


def atomic_write_json(path, payload, *, indent=None):
    """Write JSON beside ``path`` and atomically replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=indent)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def canonical_episode_sha256(episode_spec):
    """Bind a result to the complete canonical dataset episode payload."""
    encoded = json.dumps(
        episode_spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metrics(payload, label):
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    finish = payload.get("finish")
    total = payload.get("total")
    if isinstance(finish, bool) or not isinstance(finish, (int, float)):
        raise ArtifactValidationError(f"{label} finish must be numeric")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        raise ArtifactValidationError(f"{label} total must be numeric")
    if (not math.isfinite(float(finish)) or
            not math.isfinite(float(total)) or
            total <= 0 or finish < 0 or finish > total):
        raise ArtifactValidationError(
            f"{label} has invalid finish/total: {finish}/{total}")
    return finish, total


def build_episode_result(episode_id, episode_spec, max_frames, finish, total,
                         *, protocol=LEGACY_PROTOCOL):
    """Build an episode result in the selected protocol."""
    _validate_metrics({"finish": finish, "total": total}, "episode result")
    if protocol == LEGACY_PROTOCOL:
        return {"finish": finish, "total": total}
    if protocol != HUMAN_BOX_PROTOCOL:
        raise ValueError(f"unsupported result protocol: {protocol!r}")
    return {
        "schema_version": EPISODE_SCHEMA,
        "protocol": HUMAN_BOX_PROTOCOL,
        "episode_id": int(episode_id),
        "episode_sha256": canonical_episode_sha256(episode_spec),
        "max_frames": int(max_frames),
        "finish": finish,
        "total": total,
    }


def validate_episode_result(payload, episode_id, episode_spec, max_frames,
                            *, protocol=LEGACY_PROTOCOL, label="episode result"):
    """Validate metrics and, in strict mode, the complete episode identity."""
    _validate_metrics(payload, label)
    if protocol == LEGACY_PROTOCOL:
        return payload
    if protocol != HUMAN_BOX_PROTOCOL:
        raise ArtifactValidationError(
            f"unsupported result protocol: {protocol!r}")
    expected = {
        "schema_version": EPISODE_SCHEMA,
        "protocol": HUMAN_BOX_PROTOCOL,
        "episode_id": int(episode_id),
        "episode_sha256": canonical_episode_sha256(episode_spec),
        "max_frames": int(max_frames),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactValidationError(
                f"{label} {key} must be {value!r}, got {payload.get(key)!r}")
    return payload


def load_episode_result(path, episode_id, episode_spec, max_frames,
                        *, protocol=LEGACY_PROTOCOL):
    """Load and validate an existing result before allowing it to be skipped."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"invalid existing episode result {path}: {exc}") from exc
    return validate_episode_result(
        payload,
        episode_id,
        episode_spec,
        max_frames,
        protocol=protocol,
        label=str(path),
    )


def build_aggregate_result(results, avg_finish, *, protocol=LEGACY_PROTOCOL):
    payload = {
        "episode_results": results,
        "avg_finish": avg_finish,
    }
    if protocol == HUMAN_BOX_PROTOCOL:
        payload = {
            "schema_version": AGGREGATE_SCHEMA,
            "protocol": HUMAN_BOX_PROTOCOL,
            **payload,
        }
    elif protocol != LEGACY_PROTOCOL:
        raise ValueError(f"unsupported result protocol: {protocol!r}")
    return payload


def validate_aggregate_result(payload, expected_results,
                              *, protocol=LEGACY_PROTOCOL,
                              label="aggregate result"):
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    if protocol == HUMAN_BOX_PROTOCOL:
        if payload.get("schema_version") != AGGREGATE_SCHEMA:
            raise ArtifactValidationError(
                f"{label} schema_version must be {AGGREGATE_SCHEMA!r}")
        if payload.get("protocol") != HUMAN_BOX_PROTOCOL:
            raise ArtifactValidationError(
                f"{label} protocol must be {HUMAN_BOX_PROTOCOL!r}")
    elif protocol != LEGACY_PROTOCOL:
        raise ArtifactValidationError(
            f"unsupported result protocol: {protocol!r}")

    actual_results = payload.get("episode_results")
    expected_by_string = {str(key): value for key, value in expected_results.items()}
    if not isinstance(actual_results, dict) or actual_results != expected_by_string:
        raise ArtifactValidationError(
            f"{label} episode_results do not match validated episode files")
    ratios = []
    for episode_id, result in expected_by_string.items():
        finish, total = _validate_metrics(
            result, f"{label} episode {episode_id}")
        ratios.append(finish / total)
    expected_average = sum(ratios) / len(ratios)
    avg_finish = payload.get("avg_finish")
    if (isinstance(avg_finish, bool) or
            not isinstance(avg_finish, (int, float)) or
            not math.isclose(
                avg_finish, expected_average, rel_tol=1e-9, abs_tol=1e-12)):
        raise ArtifactValidationError(
            f"{label} avg_finish does not match episode results")
    return payload


def validate_resume_directory(run_root, episode_ids, dataset, max_frames,
                              *, protocol):
    """Validate every artifact that a resumed run would otherwise trust."""
    run_root = Path(run_root)
    validated = {}
    for episode_id in episode_ids:
        result_path = run_root / str(episode_id) / "result_episode.json"
        if not result_path.exists():
            continue
        validated[int(episode_id)] = load_episode_result(
            result_path,
            episode_id,
            dataset[episode_id],
            max_frames,
            protocol=protocol,
        )

    aggregate_path = run_root / "eval_result.json"
    if aggregate_path.exists():
        if len(validated) != len(episode_ids):
            raise ArtifactValidationError(
                "aggregate result exists although not all requested episode "
                "results are present")
        try:
            with aggregate_path.open("r", encoding="utf-8") as stream:
                aggregate = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(
                f"invalid existing aggregate result {aggregate_path}: {exc}") from exc
        validate_aggregate_result(
            aggregate,
            validated,
            protocol=protocol,
            label=str(aggregate_path),
        )
    return validated
