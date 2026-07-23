from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError


ATTEMPT_A = "attempt:v1:" + "a" * 64
ATTEMPT_B = "attempt:v1:" + "b" * 64
PROFILE_A = "sha256:" + "c" * 64
PROFILE_B = "sha256:" + "d" * 64


class StepClock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


def open_resources(root: Path):
    module = importlib.import_module("op_bench.runtime.resources")
    ledger = module.AttemptResourceLedger(
        root / "runtime_resources.jsonl",
        attempt_id=ATTEMPT_A,
        retry_index=1,
        runtime_profile_hash=PROFILE_A,
        clock_ms=StepClock(),
    )
    store = module.RuntimeLeaseStore(
        root / "private_runtime_resources.json",
        attempt_id=ATTEMPT_A,
        retry_index=1,
        runtime_profile_hash=PROFILE_A,
    )
    return module, ledger, store


class RuntimeResourceIdentityTests(unittest.TestCase):
    def test_resource_module_is_public(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("op_bench.runtime.resources"))
        runtime_package = importlib.import_module("op_bench.runtime")
        for name in (
            "AttemptResourceLedger",
            "RuntimeCleanupReport",
            "RuntimeLeaseStore",
            "runtime_resource_id",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(runtime_package, name))

    def test_resource_identity_is_deterministic_and_sensitive_to_every_axis(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        baseline = module.runtime_resource_id(
            ATTEMPT_A,
            1,
            PROFILE_A,
            "workspace",
            1,
        )
        variants = (
            module.runtime_resource_id(ATTEMPT_B, 1, PROFILE_A, "workspace", 1),
            module.runtime_resource_id(ATTEMPT_A, 2, PROFILE_A, "workspace", 1),
            module.runtime_resource_id(ATTEMPT_A, 1, PROFILE_B, "workspace", 1),
            module.runtime_resource_id(ATTEMPT_A, 1, PROFILE_A, "process", 1),
            module.runtime_resource_id(ATTEMPT_A, 1, PROFILE_A, "workspace", 2),
        )

        self.assertRegex(baseline, r"^resource:v1:[0-9a-f]{64}$")
        self.assertEqual(
            baseline,
            module.runtime_resource_id(ATTEMPT_A, 1, PROFILE_A, "workspace", 1),
        )
        self.assertEqual(len({baseline, *variants}), 6)
        with self.assertRaisesRegex(ContractError, "resource_type: unsupported value"):
            module.runtime_resource_id(ATTEMPT_A, 1, PROFILE_A, "host_scan", 1)


class AttemptResourceLedgerTests(unittest.TestCase):
    def test_declared_created_released_round_trip_binds_private_handle_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module, ledger, store = open_resources(root)
            declared = ledger.declare("workspace", 1)
            handle = store.put_exact(
                declared.resource_id,
                "workspace",
                1,
                "/private/controller/workspace-a",
            )
            created = ledger.created(declared.resource_id, handle.raw_handle_hash)
            released = ledger.released(declared.resource_id)

            self.assertEqual(
                [record.transition for record in ledger.records],
                ["declared", "created", "released"],
            )
            self.assertIsNone(declared.raw_handle_hash)
            self.assertEqual(created.raw_handle_hash, handle.raw_handle_hash)
            self.assertEqual(released.raw_handle_hash, handle.raw_handle_hash)
            self.assertEqual(store.get_exact(declared.resource_id), handle)
            self.assertEqual(store.active_handles, (handle,))

            public_text = (root / "runtime_resources.jsonl").read_text(encoding="utf-8")
            private_text = (root / "private_runtime_resources.json").read_text(encoding="utf-8")
            self.assertNotIn(handle.raw_handle, public_text)
            self.assertIn(handle.raw_handle, private_text)
            self.assertTrue(public_text.endswith("\n"))
            for line in public_text.splitlines():
                self.assertEqual(line, canonical_json(json.loads(line)))

            reopened = module.AttemptResourceLedger(
                root / "runtime_resources.jsonl",
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
                clock_ms=StepClock(2_000),
            )
            reopened_store = module.RuntimeLeaseStore(
                root / "private_runtime_resources.json",
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            self.assertEqual(reopened.records, ledger.records)
            self.assertEqual(reopened.verify(), ledger.records)
            self.assertEqual(reopened_store.get_exact(declared.resource_id), handle)
            parsed_records = module.parse_runtime_resource_ledger(
                (root / "runtime_resources.jsonl").read_bytes(),
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            parsed_handles = module.parse_runtime_lease_store(
                (root / "private_runtime_resources.json").read_bytes(),
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            report = module.RuntimeCleanupReport(
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
                entries=(
                    module.RuntimeCleanupEntry(
                        resource_id=declared.resource_id,
                        resource_type="workspace",
                        status="released",
                        error_code=None,
                    ),
                ),
                all_released=True,
            )
            module.verify_runtime_resource_evidence(
                parsed_records,
                parsed_handles,
                report,
            )
            self.assertEqual(parsed_records, ledger.records)
            self.assertEqual(parsed_handles, (handle,))

    def test_create_failed_is_terminal_without_a_private_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, ledger, store = open_resources(Path(temporary))
            declared = ledger.declare("container", 1)

            failed = ledger.create_failed(declared.resource_id)

            self.assertEqual(failed.transition, "create_failed")
            self.assertEqual(store.active_handles, ())
            with self.assertRaisesRegex(ContractError, "terminal transition"):
                ledger.created(declared.resource_id, "sha256:" + "e" * 64)

    def test_illegal_duplicate_skipped_and_cross_attempt_transitions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, ledger, store = open_resources(Path(temporary))
            declared = ledger.declare("process", 1)
            with self.assertRaisesRegex(ContractError, "already declared"):
                ledger.declare("process", 1)
            with self.assertRaisesRegex(ContractError, "expected created before released"):
                ledger.released(declared.resource_id)
            with self.assertRaisesRegex(ContractError, "unknown resource_id"):
                ledger.created("resource:v1:" + "f" * 64, "sha256:" + "e" * 64)

            handle = store.put_exact(declared.resource_id, "process", 1, "pid:4242")
            ledger.created(declared.resource_id, handle.raw_handle_hash)
            ledger.cleanup_failed(declared.resource_id)
            with self.assertRaisesRegex(ContractError, "terminal transition"):
                ledger.released(declared.resource_id)

            module = importlib.import_module("op_bench.runtime.resources")
            with self.assertRaisesRegex(ContractError, "attempt_id"):
                module.AttemptResourceLedger(
                    Path(temporary) / "other.jsonl",
                    attempt_id="not-an-attempt",
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )

    def test_private_store_rejects_mismatch_duplicate_handle_and_unknown_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            module, ledger, store = open_resources(Path(temporary))
            first = ledger.declare("workspace", 1)
            second = ledger.declare("remote_workspace", 1)
            store.put_exact(first.resource_id, "workspace", 1, "same-raw-handle")

            with self.assertRaisesRegex(ContractError, "resource_type mismatch"):
                store.put_exact(second.resource_id, "workspace", 1, "other-handle")
            with self.assertRaisesRegex(ContractError, "raw handle is already owned"):
                store.put_exact(second.resource_id, "remote_workspace", 1, "same-raw-handle")
            with self.assertRaisesRegex(ContractError, "unknown resource_id"):
                store.get_exact("resource:v1:" + "9" * 64)

            other = module.RuntimeLeaseStore(
                Path(temporary) / "other-private.json",
                attempt_id=ATTEMPT_B,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            with self.assertRaisesRegex(ContractError, "resource_id does not belong"):
                other.put_exact(first.resource_id, "workspace", 1, "cross-attempt")

    def test_corrupt_tail_hash_and_semantically_illegal_recomputed_chain_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module, ledger, store = open_resources(root)
            declared = ledger.declare("container", 1)
            handle = store.put_exact(declared.resource_id, "container", 1, "container-a")
            ledger.created(declared.resource_id, handle.raw_handle_hash)
            path = root / "runtime_resources.jsonl"
            original_records = ledger.records
            valid = path.read_bytes()

            path.write_bytes(valid.removesuffix(b"\n"))
            with self.assertRaisesRegex(ContractError, "missing final newline"):
                module.AttemptResourceLedger(
                    path,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )

            path.write_bytes(valid)
            rows = [json.loads(line) for line in valid.splitlines()]
            rows[1]["raw_handle_hash"] = "sha256:" + "1" * 64
            path.write_text("\n".join(canonical_json(row) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "record_hash"):
                module.AttemptResourceLedger(
                    path,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )

            illegal = replace(
                original_records[1],
                transition="released",
                record_hash="sha256:" + "0" * 64,
            )
            illegal = replace(
                illegal,
                record_hash=module.runtime_resource_record_hash(illegal),
            )
            path.write_text(
                canonical_json(original_records[0].to_dict())
                + "\n"
                + canonical_json(illegal.to_dict())
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "expected created before released"):
                module.AttemptResourceLedger(
                    path,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )

    def test_symlinked_ledger_or_private_store_is_denied(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real.jsonl"
            real.write_text("", encoding="utf-8")
            linked = root / "linked.jsonl"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ContractError, "symlink"):
                module.AttemptResourceLedger(
                    linked,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )

            private_real = root / "private-real.json"
            private_real.write_text("{}", encoding="utf-8")
            private_link = root / "private-link.json"
            private_link.symlink_to(private_real)
            with self.assertRaisesRegex(ContractError, "symlink"):
                module.RuntimeLeaseStore(
                    private_link,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                )

    def test_resource_files_must_be_regular(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger_directory = root / "ledger-directory"
            ledger_directory.mkdir()
            private_directory = root / "private-directory"
            private_directory.mkdir()

            with self.assertRaisesRegex(ContractError, "regular file|invalid"):
                module.AttemptResourceLedger(
                    ledger_directory,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                    clock_ms=StepClock(),
                )
            with self.assertRaisesRegex(ContractError, "regular file|invalid"):
                module.RuntimeLeaseStore(
                    private_directory,
                    attempt_id=ATTEMPT_A,
                    retry_index=1,
                    runtime_profile_hash=PROFILE_A,
                )

    def test_uncertain_append_reconciles_complete_commit_and_poisons_partial_tail(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            committed = module.AttemptResourceLedger(
                root / "committed.jsonl",
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
                clock_ms=StepClock(),
            )
            with (
                mock.patch(
                    "op_bench.runtime.resources.os.fsync",
                    side_effect=(
                        None,
                        OSError("fixture parent fsync failure"),
                        None,
                        None,
                    ),
                ),
                mock.patch(
                    "op_bench.runtime.resources.os.ftruncate",
                    side_effect=OSError("fixture rollback failure"),
                ),
            ):
                declared = committed.declare("workspace", 1)
            self.assertEqual(committed.records, (declared,))

            poisoned = module.AttemptResourceLedger(
                root / "poisoned.jsonl",
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
                clock_ms=StepClock(),
            )
            real_write = __import__("os").write
            calls = [0]

            def partial_then_fail(descriptor: int, content) -> int:
                calls[0] += 1
                if calls[0] == 1:
                    raw = bytes(content)
                    return real_write(descriptor, raw[: max(1, len(raw) // 2)])
                raise OSError("fixture write failure")

            with (
                mock.patch(
                    "op_bench.runtime.resources.os.write",
                    side_effect=partial_then_fail,
                ),
                mock.patch(
                    "op_bench.runtime.resources.os.ftruncate",
                    side_effect=OSError("fixture rollback failure"),
                ),
            ):
                with self.assertRaisesRegex(ContractError, "uncertain|poison"):
                    poisoned.declare("workspace", 1)
            with self.assertRaisesRegex(ContractError, "poison"):
                poisoned.declare("process", 1)

    def test_concurrent_private_handle_updates_merge_without_loss(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private_runtime_resources.json"
            first = module.RuntimeLeaseStore(
                path,
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            second = module.RuntimeLeaseStore(
                path,
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
            )
            barrier = threading.Barrier(2)
            original_write = module.RuntimeLeaseStore._write_handles

            def synchronized_write(instance, handles):
                barrier.wait(timeout=5)
                return original_write(instance, handles)

            outcomes: list[object] = []

            def put(store, resource_type: str, raw_handle: str) -> None:
                resource_id = module.runtime_resource_id(
                    ATTEMPT_A,
                    1,
                    PROFILE_A,
                    resource_type,
                    1,
                )
                try:
                    outcomes.append(
                        store.put_exact(
                            resource_id,
                            resource_type,
                            1,
                            raw_handle,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - preserve race outcome.
                    outcomes.append(exc)

            with mock.patch.object(
                module.RuntimeLeaseStore,
                "_write_handles",
                synchronized_write,
            ):
                threads = (
                    threading.Thread(
                        target=put,
                        args=(first, "workspace", "/private/workspace"),
                    ),
                    threading.Thread(
                        target=put,
                        args=(second, "process", "pid:4242"),
                    ),
                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(
                all(isinstance(item, module.RuntimeResourceHandle) for item in outcomes),
                outcomes,
            )
            self.assertEqual(len(first.active_handles), 2)

    def test_resource_apis_expose_no_discovery_or_adoption_surface(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        forbidden = ("discover", "scan", "enumerate", "filter", "adopt")
        names = {
            name.lower()
            for cls in (module.AttemptResourceLedger, module.RuntimeLeaseStore)
            for name in dir(cls)
        }

        self.assertFalse(
            {
                name
                for name in names
                if any(fragment in name for fragment in forbidden)
            }
        )

    def test_resource_evidence_rejects_active_missing_extra_and_cleanup_failed_handles(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ledger, store = open_resources(root)
            declared = ledger.declare("workspace", 1)
            handle = store.put_exact(
                declared.resource_id,
                "workspace",
                1,
                "/private/workspace-a",
            )
            ledger.created(declared.resource_id, handle.raw_handle_hash)
            cleanup_failed = module.RuntimeCleanupReport(
                attempt_id=ATTEMPT_A,
                retry_index=1,
                runtime_profile_hash=PROFILE_A,
                entries=(
                    module.RuntimeCleanupEntry(
                        resource_id=declared.resource_id,
                        resource_type="workspace",
                        status="cleanup_failed",
                        error_code="workspace_remove_failed",
                    ),
                ),
                all_released=False,
            )
            with self.assertRaisesRegex(ContractError, "active resource"):
                module.verify_runtime_resource_evidence(
                    ledger.records,
                    store.active_handles,
                    cleanup_failed,
                )

            ledger.cleanup_failed(declared.resource_id)
            with self.assertRaisesRegex(ContractError, "cleanup_failed"):
                module.verify_runtime_resource_evidence(
                    ledger.records,
                    store.active_handles,
                    cleanup_failed,
                )

            released = replace(
                cleanup_failed,
                entries=(
                    replace(
                        cleanup_failed.entries[0],
                        status="released",
                        error_code=None,
                    ),
                ),
                all_released=True,
            )
            with self.assertRaisesRegex(ContractError, "cleanup report differs"):
                module.verify_runtime_resource_evidence(
                    ledger.records,
                    store.active_handles,
                    released,
                )


class RuntimeCleanupReportTests(unittest.TestCase):
    def test_cleanup_report_requires_one_final_status_per_resource(self) -> None:
        module = importlib.import_module("op_bench.runtime.resources")
        released = module.RuntimeCleanupEntry(
            resource_id=module.runtime_resource_id(
                ATTEMPT_A, 1, PROFILE_A, "workspace", 1
            ),
            resource_type="workspace",
            status="released",
            error_code=None,
        )
        report = module.RuntimeCleanupReport(
            attempt_id=ATTEMPT_A,
            retry_index=1,
            runtime_profile_hash=PROFILE_A,
            entries=(released,),
            all_released=True,
        )

        self.assertEqual(module.RuntimeCleanupReport.from_dict(report.to_dict()), report)
        with self.assertRaisesRegex(ContractError, "all_released"):
            replace(report, all_released=False)
        with self.assertRaisesRegex(ContractError, "error_code"):
            replace(
                report,
                entries=(replace(released, status="cleanup_failed"),),
                all_released=False,
            )


if __name__ == "__main__":
    unittest.main()
