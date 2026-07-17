from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.backends import LocalProcessBackend
from op_bench.runtime.validation import ContractError
from tests.test_runtime_backends import LocalBackendFixture


class RuntimeBackendCleanupTests(unittest.TestCase):
    def test_cleanup_failure_is_terminal_public_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)

            with mock.patch(
                "op_bench.runtime.backends.shutil.rmtree",
                side_effect=OSError("private cleanup detail"),
            ):
                result = backend.cleanup(lease)

            self.assertFalse(result.report.all_released)
            self.assertEqual(result.report.entries[-1].status, "cleanup_failed")
            self.assertEqual(
                result.report.entries[-1].error_code,
                "workspace_remove_failed",
            )
            self.assertNotIn("private cleanup detail", repr(result.report.to_dict()))
            with self.assertRaisesRegex(ContractError, "terminal"):
                fixture.ledger.released(lease.handles[0].resource_id)

    def test_cleanup_rejects_handle_not_owned_by_exact_retry_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            private_path = fixture.evidence / "private_runtime_resources.json"
            private_path.unlink()

            with self.assertRaisesRegex(ContractError, "handle|lease"):
                backend.cleanup(lease)


if __name__ == "__main__":
    unittest.main()
