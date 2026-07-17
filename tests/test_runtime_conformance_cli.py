from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.conformance import RuntimeConformanceRunner
from op_bench.runtime.backends import RuntimeBackendUnavailable
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.task_view import assert_public_artifact_safe
from scripts.run_runtime_conformance import main
from tests.runtime_git_fixture import initialize_git_repo


REGISTRY = (
    Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
)


class RuntimeConformanceRunnerTests(unittest.TestCase):
    def test_external_mode_consumes_exact_target_and_caches_backend_reason(self) -> None:
        class UnavailableBackend:
            prepare_calls = 0

            def prepare(self, profile, attempt_context):
                type(self).prepare_calls += 1
                raise RuntimeBackendUnavailable("remote_workspace_create_failed")

            def run(self, lease, command, cwd, timeout_ms):
                raise AssertionError("unavailable backend must not run")

            def collect(self, lease):
                raise AssertionError("unavailable backend must not collect")

            def cleanup(self, lease):
                raise AssertionError("unavailable backend has no lease")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity = root / "identity"
            identity.write_text("fixture", encoding="utf-8")
            target = root / "target.json"
            target.write_text(
                json.dumps(
                    {
                        "backend": "remote_docker",
                        "local_workspace_parent": str(workspaces),
                        "host_alias": "exact.invalid",
                        "remote_user": "runner",
                        "identity_file": str(identity),
                    }
                ),
                encoding="utf-8",
            )
            local = load_runtime_profile_registry(REGISTRY).get(
                "local-cpu-process-v1"
            )
            external = replace(local, backend="remote_docker")
            UnavailableBackend.prepare_calls = 0

            report = RuntimeConformanceRunner(
                fixture_source=fixture,
                runtime_profile=local,
                external_backend_factory=lambda profile, binding: UnavailableBackend(),
            ).run(
                root / "output",
                include_external=True,
                target_config=target,
                external_profile=external,
            )

            entry = report.entries[-1]
            self.assertEqual(report.status, "blocked")
            self.assertEqual(entry.entry_id, "external-exact-target")
            self.assertEqual(entry.status, "blocked")
            self.assertEqual(
                entry.reason_code,
                "remote_workspace_create_failed",
            )
            self.assertEqual(entry.runtime_profile_id, external.profile_id)
            self.assertEqual(UnavailableBackend.prepare_calls, 1)

    def test_deterministic_four_entry_matrix_passes_and_one_difference_is_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            profile = load_runtime_profile_registry(REGISTRY).get(
                "local-cpu-process-v1"
            )
            report = RuntimeConformanceRunner(
                fixture_source=fixture,
                runtime_profile=profile,
            ).run(root / "passed")

            self.assertEqual(report.status, "passed")
            self.assertEqual(len(report.entries), 4)
            self.assertTrue(all(entry.status == "passed" for entry in report.entries))

            changed = RuntimeConformanceRunner(
                fixture_source=fixture,
                runtime_profile=profile,
                semantic_override={"mcp-scripted_remote": "f2p_failed"},
            ).run(root / "failed")
            selected = next(
                entry
                for entry in changed.entries
                if entry.entry_id == "mcp-scripted_remote"
            )
            self.assertEqual(changed.status, "failed")
            self.assertEqual(selected.status, "failed")
            self.assertEqual(selected.differences, ("$.evaluation_outcome",))

    def test_offline_cli_writes_one_canonical_public_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            output = root / "output"

            exit_code = main(
                [
                    "--fixture",
                    str(fixture),
                    "--output-dir",
                    str(output),
                    "--profile-registry",
                    str(REGISTRY),
                ]
            )

            path = output / "runtime_conformance.json"
            raw = path.read_bytes()
            payload = json.loads(raw)
            self.assertEqual(exit_code, 0)
            self.assertEqual(raw, (canonical_json(payload) + "\n").encode("utf-8"))
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(len(payload["entries"]), 4)
            assert_public_artifact_safe(payload)

    def test_external_mode_blocks_exact_missing_target_without_searching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            output = root / "output"

            exit_code = main(
                [
                    "--fixture",
                    str(fixture),
                    "--output-dir",
                    str(output),
                    "--profile-registry",
                    str(REGISTRY),
                    "--include-external",
                ]
            )

            payload = json.loads(
                (output / "runtime_conformance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(payload["entries"][-1]["status"], "blocked")
            self.assertEqual(
                payload["entries"][-1]["reason_code"],
                "target_config_missing",
            )


if __name__ == "__main__":
    unittest.main()
