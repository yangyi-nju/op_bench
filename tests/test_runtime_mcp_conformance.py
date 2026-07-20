from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.conformance import RuntimeConformanceRunner
from op_bench.runtime.profiles import load_runtime_profile_registry
from tests.runtime_git_fixture import initialize_git_repo


REGISTRY = (
    Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
)


class McpStdioConformanceTests(unittest.TestCase):
    def test_real_stdio_matrix_matches_cli_and_reports_transport_counters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            profile = load_runtime_profile_registry(REGISTRY).get(
                "local-cpu-process-v1"
            )
            runner = RuntimeConformanceRunner(
                fixture_source=fixture,
                runtime_profile=profile,
            )

            report = runner.run(root / "output", transport="mcp-stdio")

            self.assertEqual(report.status, "passed")
            self.assertEqual(
                tuple(entry.entry_id for entry in report.entries),
                (
                    "cli-local_process",
                    "cli-scripted_remote",
                    "mcp-stdio-local_process",
                    "mcp-stdio-scripted_remote",
                ),
            )
            self.assertEqual(
                len({entry.normalized_snapshot_hash for entry in report.entries}),
                1,
            )
            stdio_entries = [
                entry for entry in report.entries if entry.transport == "mcp-stdio"
            ]
            for entry in stdio_entries:
                self.assertEqual(
                    entry.transport_counters,
                    {
                        "initialize_count": 1,
                        "tools_list_count": 1,
                        "tools_call_count": 6,
                        "protocol_error_count": 0,
                        "server_terminal_status": "client_closed",
                    },
                )

    def test_stdio_snapshot_covers_events_tree_patch_and_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            initialize_git_repo(fixture)
            profile = load_runtime_profile_registry(REGISTRY).get(
                "local-cpu-process-v1"
            )
            runner = RuntimeConformanceRunner(
                fixture_source=fixture,
                runtime_profile=profile,
            )

            execution = runner._run_entry(
                "mcp-stdio-local_process",
                "mcp-stdio",
                "local_process",
            )
            snapshot = execution.snapshot

            event_types = tuple(item["event_type"] for item in snapshot.event_sequence)
            self.assertEqual(event_types.count("action_requested"), 6)
            self.assertEqual(event_types.count("action_observed"), 6)
            self.assertEqual(snapshot.finish_count, 1)
            self.assertEqual(
                tuple(item["path"] for item in snapshot.workspace_tree),
                ("src/helper.py", "src/operator.py"),
            )
            self.assertIn("patch_bytes_sha256", snapshot.patch_identity)
            self.assertEqual(snapshot.patch_identity["changed_paths"], ["src/operator.py"])


if __name__ == "__main__":
    unittest.main()
