from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import unittest

from op_bench.factory.admission_rebaseline import (
    build_admission_contract_rebaseline,
)
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "4f5addc"
TASK_RELATIVE = "tasks/pytorch/132616_cuda_mem_get_info"
TASK_PATH = ROOT / TASK_RELATIVE / "task.json"
EVIDENCE_PATH = ROOT / TASK_RELATIVE / "admission/evidence.json"
REBASELINE_PATH = (
    ROOT / "factory/v0.7/p6/cuda_mem_get_info_contract_rebaseline.json"
)


def git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ("git", "show", f"{BASELINE_COMMIT}:{relative_path}"),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def baseline_artifacts() -> dict[str, bytes]:
    return {
        "gold_patch": git_bytes(f"{TASK_RELATIVE}/artifacts/gold.patch"),
        "hidden_test_patch": git_bytes(
            f"{TASK_RELATIVE}/artifacts/hidden_test.patch"
        ),
    }


class V07AdmissionRebaselineTests(unittest.TestCase):
    def test_checked_in_rebaseline_rebuilds_from_exact_historical_bytes(self) -> None:
        payload = build_admission_contract_rebaseline(
            task=TaskManifest.load(TASK_PATH),
            baseline_commit=BASELINE_COMMIT,
            baseline_manifest_bytes=git_bytes(f"{TASK_RELATIVE}/task.json"),
            baseline_admission_bytes=git_bytes(
                f"{TASK_RELATIVE}/admission/evidence.json"
            ),
            baseline_artifact_bytes=baseline_artifacts(),
        )

        self.assertEqual(
            REBASELINE_PATH.read_text(encoding="utf-8"),
            canonical_json(payload) + "\n",
        )
        self.assertEqual(payload["baseline_commit"], BASELINE_COMMIT)
        self.assertEqual(
            payload["old_admission_evidence"]["bytes_sha256"],
            "sha256:"
            "61090ef10cad41edf8448993c3a512eb219d556766ea094074ce2464cf8dc19a",
        )
        self.assertEqual(
            payload["replay_spec_hash"],
            "sha256:"
            "0f6f23a9f6467529c4fe29647ad0ece55617cf988fe166194b45dcfdd5c05ca3",
        )
        self.assertTrue(payload["proofs"]["replay_spec_fields_unchanged"])
        self.assertTrue(payload["proofs"]["private_artifact_bytes_unchanged"])
        self.assertTrue(payload["proofs"]["compatibility_content_unchanged"])
        self.assertTrue(payload["proofs"]["historical_runtime_outcomes_unchanged"])

        historical = json.loads(
            git_bytes(f"{TASK_RELATIVE}/admission/evidence.json")
        )
        migrated = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            {
                key: value
                for key, value in migrated.items()
                if key
                not in {
                    "evidence_id",
                    "task_manifest_hash",
                    "task_manifest_hash_kind",
                }
            },
            {
                key: value
                for key, value in historical.items()
                if key not in {"evidence_id", "task_manifest_hash"}
            },
        )
        self.assertEqual(
            migrated["baseline"],
            historical["baseline"],
        )
        self.assertEqual(migrated["gold"], historical["gold"])
        self.assertEqual(migrated["admission"], historical["admission"])
        self.assertEqual(migrated["source"], historical["source"])
        self.assertEqual(migrated["environment"], historical["environment"])

    def test_rebaseline_rejects_replay_field_private_artifact_and_outcome_drift(
        self,
    ) -> None:
        baseline_manifest = json.loads(
            git_bytes(f"{TASK_RELATIVE}/task.json")
        )
        baseline_admission = json.loads(
            git_bytes(f"{TASK_RELATIVE}/admission/evidence.json")
        )
        mutations = {
            "replay-field": (
                lambda manifest, admission: manifest["evaluation"].__setitem__(
                    "timeout_sec",
                    1,
                )
            ),
            "private-artifact": (
                lambda manifest, admission: manifest["artifacts"].__setitem__(
                    "gold_patch",
                    "artifacts/other.patch",
                )
            ),
            "runtime-outcome": (
                lambda manifest, admission: admission["gold"].__setitem__(
                    "status",
                    "f2p_failed",
                )
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                manifest = copy.deepcopy(baseline_manifest)
                admission = copy.deepcopy(baseline_admission)
                mutate(manifest, admission)
                with self.assertRaises(ContractError):
                    build_admission_contract_rebaseline(
                        task=TaskManifest.load(TASK_PATH),
                        baseline_commit=BASELINE_COMMIT,
                        baseline_manifest_bytes=canonical_json(manifest).encode(
                            "utf-8"
                        ),
                        baseline_admission_bytes=canonical_json(admission).encode(
                            "utf-8"
                        ),
                        baseline_artifact_bytes=baseline_artifacts(),
                    )


if __name__ == "__main__":
    unittest.main()
