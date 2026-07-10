from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from op_bench.task import TaskManifest


REPLAY_SPEC_HASH_KIND = "replay_spec_v1"

# Admission verifies replay behavior, not curation state or reporting metadata.
# Keep this list explicit so adding descriptive fields does not silently change
# the evidence identity.
_REPLAY_FIELDS = (
    "task_id",
    "version",
    "environment_ref",
    "runtime_tier",
    "source_ref",
    "patch_scope",
    "source",
    "environment",
    "evaluation",
    "artifacts",
)


def replay_spec_hash(task: TaskManifest) -> str:
    # Admission receives a registry-resolved TaskManifest, while dataset
    # validation starts from task.json. Hash the checked-in manifest in both
    # cases; resolved asset identity is recorded separately in evidence.
    if task.task_json_path.is_file():
        manifest_data = json.loads(task.task_json_path.read_text(encoding="utf-8"))
    else:
        manifest_data = task.data
    manifest = {
        field: manifest_data[field]
        for field in _REPLAY_FIELDS
        if field in manifest_data
    }
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_config = manifest_data.get("artifacts", {})
    if isinstance(artifact_config, dict):
        for name, value in sorted(artifact_config.items()):
            if not isinstance(value, str):
                continue
            relative_path = Path(value)
            safe_path = not relative_path.is_absolute() and ".." not in relative_path.parts
            path = task.task_dir / relative_path
            artifacts[name] = {
                "path": value,
                "sha256": (
                    f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
                    if safe_path and path.is_file()
                    else None
                ),
            }

    payload = {
        "hash_kind": REPLAY_SPEC_HASH_KIND,
        "manifest": manifest,
        "artifact_contents": artifacts,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
