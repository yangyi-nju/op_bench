from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from op_bench.factory.artifacts import (
    FactoryArtifactStore,
    load_factory_contract,
)
from op_bench.factory.prompt_quality import (
    PromptQualityEvidence,
    build_prompt_quality_evidence,
    empty_private_index,
)
from op_bench.factory.complexity import ComplexityEvidence, build_complexity_evidence
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError
from tests.test_factory_contracts import candidate


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_factory_artifact.py"


def prompt_quality() -> PromptQualityEvidence:
    return build_prompt_quality_evidence(
        task_id="pytorch__empty_addmv",
        public_task_id="task-v07-empty-addmv",
        rendered_prompt="The public behavior differs for an empty matrix.",
        agent_task_view={"statement_body": "The public behavior differs for an empty matrix."},
        private_index=empty_private_index(),
        scanner_version="prompt-overlap-v1",
        blind_review={
            "decision": "accepted",
            "reviewer": "reviewer-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        semantic_review={
            "decision": "equivalent",
            "reviewer": "curator-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        decision="accepted",
        created_at="2026-07-29T00:00:00Z",
    )


def complexity_evidence() -> ComplexityEvidence:
    return build_complexity_evidence(
        task_id="pytorch__empty_addmv",
        localization=2,
        diagnosis=2,
        repair_regression=1,
        dimension_evidence={
            "localization": "The prompt requires tracing the public operation.",
            "diagnosis": "The failure requires comparing behavior contracts.",
            "repair_regression": "The repair must preserve neighboring behavior.",
        },
        hard_rejections=(),
        risk_signals=(),
        duplicate_fingerprint="sha256:" + "a" * 64,
        duplicate_decision="distinct",
        blind_pilot=None,
        second_review=False,
        reviewer="complexity-reviewer",
        reviewed_at="2026-07-29T00:00:00Z",
    )


class FactoryArtifactStoreTests(unittest.TestCase):
    def test_write_read_and_idempotent_duplicate_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)
            selected = candidate()

            first = store.write_contract("candidates/selected.json", selected)
            second = store.write_contract("candidates/selected.json", selected)

            self.assertEqual(first, second)
            self.assertEqual(store.read_contract(first), selected)
            store.verify_reference(first)
            self.assertEqual(
                (root / first.relative_path).read_bytes(),
                canonical_json(selected.to_dict()).encode("utf-8"),
            )

    def test_relative_normalized_json_paths_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = FactoryArtifactStore(Path(raw) / "factory")
            for path in (
                "/absolute.json",
                "../outside.json",
                "nested/../../outside.json",
                "nested\\candidate.json",
                "nested/./candidate.json",
                "candidate.txt",
            ):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ContractError, "relative"):
                        store.write_contract(path, candidate())

    def test_symlink_target_and_ancestor_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "factory"
            root.mkdir(mode=0o700)
            outside = base / "outside"
            outside.mkdir()
            (root / "linked").symlink_to(outside, target_is_directory=True)
            (root / "target.json").symlink_to(outside / "target.json")
            store = FactoryArtifactStore(root)

            for path in ("linked/candidate.json", "target.json"):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(ContractError, "symlink"):
                        store.write_contract(path, candidate())

    def test_existing_destination_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = FactoryArtifactStore(Path(raw) / "factory")
            store.write_contract("candidates/selected.json", candidate())

            with self.assertRaisesRegex(ContractError, "immutable"):
                store.write_contract(
                    "candidates/selected.json",
                    replace(candidate(), title="Different candidate content"),
                )

    def test_factory_public_artifacts_reject_host_private_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = FactoryArtifactStore(Path(raw) / "factory")
            unsafe = replace(
                candidate(),
                description="Captured from /Users/example/private/source.cpp",
            )

            with self.assertRaisesRegex(ContractError, "sensitive"):
                store.write_contract("candidates/unsafe.json", unsafe)

    def test_reference_hash_and_file_mode_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)
            reference = store.write_contract(
                "candidates/selected.json",
                candidate(),
            )
            path = root / reference.relative_path
            path.chmod(0o644)
            with self.assertRaisesRegex(ContractError, "mode"):
                store.verify_reference(reference)

            path.chmod(0o600)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ContractError, "hash|JSON"):
                store.verify_reference(reference)

    def test_noncanonical_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)
            reference = store.write_contract(
                "candidates/selected.json",
                candidate(),
            )
            path = root / reference.relative_path
            path.write_text(
                json.dumps(candidate().to_dict(), indent=2),
                encoding="utf-8",
            )
            path.chmod(0o600)

            with self.assertRaisesRegex(ContractError, "canonical"):
                store.verify_reference(reference)

    def test_failed_atomic_install_leaves_no_destination_or_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)

            with mock.patch(
                "op_bench.factory.artifacts.os.link",
                side_effect=OSError("injected install failure"),
            ):
                with self.assertRaisesRegex(ContractError, "install"):
                    store.write_contract(
                        "candidates/selected.json",
                        candidate(),
                    )

            self.assertFalse((root / "candidates" / "selected.json").exists())
            self.assertEqual(
                tuple((root / "candidates").glob(".factory-*.tmp")),
                (),
            )

    def test_standalone_loader_rejects_symlink_and_round_trips_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "candidate.json"
            target.write_bytes(
                canonical_json(candidate().to_dict()).encode("utf-8")
            )
            target.chmod(0o600)
            link = root / "linked.json"
            link.symlink_to(target)

            self.assertEqual(load_factory_contract(target), candidate())
            with self.assertRaisesRegex(ContractError, "symlink"):
                load_factory_contract(link)

    def test_prompt_quality_contract_is_registered_and_hash_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)
            selected = prompt_quality()
            reference = store.write_contract("quality/prompt.json", selected)

            self.assertEqual(reference.artifact_type, "prompt_quality")
            self.assertEqual(reference.artifact_id, selected.task_id)
            self.assertEqual(store.read_contract(reference), selected)

            path = root / reference.relative_path
            payload = selected.to_dict()
            payload["prompt_hash"] = "sha256:" + "c" * 64
            path.write_bytes(canonical_json(payload).encode("utf-8"))
            path.chmod(0o600)

            with self.assertRaisesRegex(ContractError, "content_hash"):
                load_factory_contract(path)

    def test_complexity_contract_is_registered_and_hash_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "factory"
            store = FactoryArtifactStore(root)
            selected = complexity_evidence()
            reference = store.write_contract("quality/complexity.json", selected)

            self.assertEqual(reference.artifact_type, "complexity_evidence")
            self.assertEqual(reference.artifact_id, selected.task_id)
            self.assertEqual(store.read_contract(reference), selected)

            path = root / reference.relative_path
            payload = selected.to_dict()
            payload["reviewer"] = "tampered-reviewer"
            path.write_bytes(canonical_json(payload).encode("utf-8"))
            path.chmod(0o600)

            with self.assertRaisesRegex(ContractError, "content_hash"):
                load_factory_contract(path)


class FactoryArtifactValidatorCliTests(unittest.TestCase):
    def _run(self, path: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_validator_accepts_candidate_and_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.json"
            path.write_bytes(
                canonical_json(candidate().to_dict()).encode("utf-8")
            )
            path.chmod(0o600)

            accepted = self._run(path)

            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            summary = json.loads(accepted.stdout)
            self.assertEqual(
                summary,
                {
                    "content_hash": candidate().content_hash,
                    "contract_type": "factory_candidate",
                    "status": "valid",
                },
            )
            self.assertNotIn(str(path), accepted.stdout)

            payload = candidate().to_dict()
            payload["title"] = "tampered"
            path.write_bytes(canonical_json(payload).encode("utf-8"))
            rejected = self._run(path)

            self.assertEqual(rejected.returncode, 1)
            self.assertIn("[contract_invalid]", rejected.stderr)
            self.assertNotIn(str(path), rejected.stderr)

    def test_validator_malformed_invocation_returns_usage_exit(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")

        result = subprocess.run(
            [sys.executable, str(VALIDATOR)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
