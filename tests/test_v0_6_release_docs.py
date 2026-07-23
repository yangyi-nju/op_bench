from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
README_FILES = (ROOT / "README.md", ROOT / "README.zh-CN.md")
INDEX_FILES = (ROOT / "docs" / "README.md", ROOT / "docs" / "README.zh-CN.md")
GUIDE = ROOT / "docs" / "v0.6" / "developer_guide.md"
MCP_EXPERIMENT = ROOT / "docs" / "v0.6" / "mcp_agent_experiment.md"
MCP_EXPERIMENT_VERIFICATION = (
    ROOT / "docs" / "v0.6" / "mcp_agent_experiment_verification.md"
)
EXPERIMENT_REPORT = ROOT / "docs" / "v0.6" / "experiment_report.md"
MCP_EXPERIMENT_REPORT = (
    ROOT
    / "runs"
    / "v0.6_mcp_full_20260722_event_redaction_r5_report"
    / "experiment_report.md"
)
MCP_EXPERIMENT_INDEX = MCP_EXPERIMENT_REPORT.with_name("experiment_index.json")
MCP_EXPERIMENT_SUMMARY = MCP_EXPERIMENT_REPORT.with_name(
    "experiment_summary.json"
)
M7_VERIFICATION = ROOT / "docs" / "v0.6" / "m7_verification.md"
DEMO_ARTIFACT = (
    ROOT / "configs" / "examples" / "v0.6_scripted_demo_artifact.example.json"
)
RELEASE_NOTES = ROOT / "docs" / "v0.6" / "release_notes.md"
ACCEPTANCE_MATRIX = ROOT / "docs" / "v0.6" / "acceptance_matrix.md"
PROJECT_STATE = ROOT / "docs" / "project_state.md"
CHANGELOG = ROOT / "CHANGELOG.md"
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class V06ReleaseDocumentationTests(unittest.TestCase):
    def test_repository_relative_links_resolve(self) -> None:
        for document in (
            *README_FILES,
            *INDEX_FILES,
            GUIDE,
            RELEASE_NOTES,
            M7_VERIFICATION,
            MCP_EXPERIMENT,
            MCP_EXPERIMENT_VERIFICATION,
            EXPERIMENT_REPORT,
        ):
            text = document.read_text(encoding="utf-8")
            for raw_target in LINK_PATTERN.findall(text):
                target = raw_target.strip().strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target).resolve().exists())

    def test_v0_6_has_a_versioned_human_readable_experiment_report(self) -> None:
        self.assertTrue(EXPERIMENT_REPORT.exists())
        report = EXPERIMENT_REPORT.read_text(encoding="utf-8")
        for fragment in (
            "# OpBench v0.6 实验报告",
            "51/51",
            "35/51",
            "68.6%",
            "15",
            "P2P regression",
            "747",
            "0 次基础设施无效",
            "0 次逻辑重试",
            "四个 Comparability Key",
            "mcp_agent_experiment.md",
            "mcp_agent_experiment_verification.md",
            "experiment_summary.json",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, report)

        for index in (*README_FILES, *INDEX_FILES, RELEASE_NOTES, PROJECT_STATE):
            with self.subTest(index=index.name):
                self.assertIn(
                    "experiment_report.md", index.read_text(encoding="utf-8")
                )

        experiment_index = json.loads(
            MCP_EXPERIMENT_INDEX.read_text(encoding="utf-8")
        )
        for task_id in {row["task_id"] for row in experiment_index["attempts"]}:
            with self.subTest(task_id=task_id):
                self.assertIn(task_id.removeprefix("pytorch__"), report)
        for profile_id in {
            row["runtime_profile_id"] for row in experiment_index["attempts"]
        }:
            with self.subTest(profile_id=profile_id):
                self.assertIn(profile_id, report)

    def test_v0_6_does_not_publish_redundant_run_evidence(self) -> None:
        for relative_path in (
            "v0.6_release3_legacy_replay_exact_complete/replay/replay_manifest.json",
            "v0.6_release_cuda_kernel_canary/replay/replay_manifest.json",
            "v0.6_release_cuda_overlay_canary/run_manifest.json",
            "v0.6_release_remote_cpu_canary/run_manifest.json",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertFalse((ROOT / "runs" / relative_path).exists())

    def test_v0_6_internal_process_drafts_are_not_published(self) -> None:
        internal_drafts = sorted(
            path
            for parent in (
                ROOT / "docs" / "superpowers" / "plans",
                ROOT / "docs" / "superpowers" / "specs",
            )
            for path in parent.glob("*v0.6*")
        )
        self.assertEqual(internal_drafts, [])

    def test_bilingual_quickstarts_cover_the_v1_release_surface(self) -> None:
        required = (
            "scripts/prepare_v0_6_demo.py",
            "scripts/run_experiment.py",
            "--runtime-protocol v1",
            "--runtime-profile local-cpu-process-v1",
            "--agent scripted_canonical",
            "--agent codex_canonical",
            "--enable-external-canary",
            "scripts/verify_runtime_resources.py",
            "scripts/validate_runtime_contract.py",
            "runs/v0.6_m7_scripted_demo",
        )
        for readme in README_FILES:
            text = readme.read_text(encoding="utf-8")
            for fragment in required:
                with self.subTest(readme=readme.name, fragment=fragment):
                    self.assertIn(fragment, text)
            self.assertIn("Completed", text)
            self.assertIn("85/85", text)
            self.assertNotIn("release remains **Blocked**", text)
            self.assertNotIn("发布仍是 **Blocked**", text)
            self.assertIn("synthetic", text.lower())
            self.assertIn("benchmark score", text.lower())

    def test_legacy_and_v1_selection_are_explicit(self) -> None:
        for readme in README_FILES:
            text = readme.read_text(encoding="utf-8")
            self.assertIn("--runtime-protocol v1", text)
            self.assertIn("--runtime-profile", text)
            self.assertIn("Legacy", text)
            self.assertIn("default", text.lower())
            self.assertIn("omit", text.lower())

    def test_developer_guide_covers_support_failures_and_artifacts(self) -> None:
        text = GUIDE.read_text(encoding="utf-8")
        self.assertIn("Status: `opbench-v0.6.0` is Completed", text)
        self.assertIn("85/85", text)
        registry = json.loads(
            (ROOT / "configs" / "runtime_profiles.v1.json").read_text(
                encoding="utf-8"
            )
        )
        for profile in registry["profiles"]:
            self.assertIn(profile["profile_id"], text)
        for fragment in (
            "attempt_validity",
            "agent_terminal",
            "evaluation_outcome",
            "invalid_request",
            "capability_denied",
            "path_denied",
            "selector_denied",
            "budget_exhausted",
            "provider_error",
            "runtime_error",
            "platform_error",
            "invalid_patch",
            "f2p_failed",
            "regression",
            "Comparability Key",
            "platform_version",
            "action_protocol",
            "evaluation_protocol",
            "scoring_protocol",
            "run_manifest.json",
            "attempts.jsonl",
            "agent_task_view.json",
            "events.jsonl",
            "final.patch",
            "session_result.json",
            "public_evaluation.json",
            "private_evaluation.json",
            "runtime_resources.jsonl",
            "runtime_cleanup.json",
            "results.jsonl",
            "summary.json",
            "integrity.json",
            "connection_timeout",
            "no target discovery",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_real_mcp_experiment_guide_freezes_usage_and_safety_boundaries(self) -> None:
        for document in (ROOT / "README.md", GUIDE, MCP_EXPERIMENT):
            text = document.read_text(encoding="utf-8")
            for fragment in (
                "codex_mcp_canonical",
                "--codex-model gpt-5.6-sol",
                "codex-cli 0.145.0-alpha.27",
                "adapter_trace.json",
                "14",
                "Provider",
                "Task network",
                "mcp-stdio",
            ):
                with self.subTest(document=document.name, fragment=fragment):
                    self.assertIn(fragment, text)

        experiment = MCP_EXPERIMENT.read_text(encoding="utf-8")
        for fragment in (
            "v0.6_mcp_full_20260722_event_redaction_r5_remote_cpu",
            "v0.6_mcp_full_20260722_event_redaction_r5_remote_cpu_compile",
            "v0.6_mcp_full_20260722_event_redaction_r5_cuda_overlay",
            "v0.6_mcp_full_20260722_event_redaction_r5_cuda_kernel",
            "scripts/summarize_mcp_experiment.py",
            "--run-root",
            "retry_infrastructure",
            "private_runtime_resources.json",
            "protocol_error_count",
            "no ping",
            "no host/service scan",
            "no process/container enumeration",
            "no target discovery",
            "invocation-local",
            "transport token",
            "read-only cwd",
            "hard byte limits",
            "zero-signal",
            "Dataset digest",
            "17-task partition",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, experiment)

    def test_real_mcp_report_is_complete_redacted_and_identity_bound(self) -> None:
        summary = json.loads(MCP_EXPERIMENT_SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["totals"]["attempts"], 51)
        self.assertEqual(summary["totals"]["cohorts"], 4)
        self.assertEqual(summary["totals"]["trace_complete"], 51)
        self.assertEqual(summary["totals"]["retries"], 0)
        self.assertEqual(
            summary["evaluation_outcomes"],
            {"f2p_failed": 15, "p2p_regression": 1, "resolved": 35},
        )
        self.assertEqual(summary["agent_terminals"], {"finished": 51})
        self.assertEqual(summary["mcp"]["initialize_count"], 51)
        self.assertEqual(summary["mcp"]["tools_list_count"], 51)
        self.assertEqual(summary["mcp"]["tools_call_count"], 747)
        self.assertEqual(summary["mcp"]["protocol_error_count"], 0)
        self.assertEqual(summary["model_id"], "gpt-5.6-sol")
        self.assertEqual(
            summary["codex_cli_version"], "codex-cli 0.145.0-alpha.27"
        )

        index = json.loads(MCP_EXPERIMENT_INDEX.read_text(encoding="utf-8"))
        self.assertEqual(len(index["attempts"]), 51)
        self.assertEqual(len(index["cohorts"]), 4)
        self.assertEqual(len({row["attempt_id"] for row in index["attempts"]}), 51)
        self.assertTrue(all(row["trace_complete"] for row in index["attempts"]))
        self.assertTrue(all(row["retry_index"] == 1 for row in index["attempts"]))

        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                MCP_EXPERIMENT_REPORT,
                MCP_EXPERIMENT_INDEX,
                MCP_EXPERIMENT_SUMMARY,
            )
        )
        for forbidden in (
            "private_evaluation",
            "private_runtime_resources",
            "/Users/",
            "BEGIN PRIVATE KEY",
            "Authorization: Bearer",
            "github_pat_",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, public_text)

    def test_representative_demo_artifact_is_public_and_path_independent(self) -> None:
        artifact = json.loads(DEMO_ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["schema_version"], "opbench.v0.6.demo_artifact.v1"
        )
        self.assertIs(artifact["synthetic_demo"], True)
        self.assertEqual(
            artifact["action_sequence"],
            ["workspace_list", "test_run", "vcs_diff", "session_finish"],
        )
        self.assertEqual(
            artifact["result_axes"],
            {
                "agent_terminal": "finished",
                "attempt_validity": "valid",
                "evaluation_outcome": "no_patch",
            },
        )
        encoded = json.dumps(artifact, sort_keys=True)
        for forbidden in ("/Users/", "C:\\\\", "runs/", "PRIVATE KEY", "sk-"):
            self.assertNotIn(forbidden, encoded)
        for value in artifact["artifact_hashes"].values():
            self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")
        for key in ("comparability_key", "runtime_profile_hash"):
            self.assertRegex(
                artifact["identities"][key], r"^sha256:[0-9a-f]{64}$"
            )
        self.assertIn("not a benchmark score", artifact["claim"])
        release = RELEASE_NOTES.read_text(encoding="utf-8")
        self.assertIn("Decision: **Completed", release)
        self.assertIn("85/85", release)
        self.assertIn("17 baseline + 17 gold + 51 historical", release)
        self.assertIn("not a v0.6 score", release)
        self.assertIn("not a formal Agent", release)
        self.assertIn("does not run the planned feedback-causality", release)
        self.assertNotIn("Decision: **Blocked", release)

    def test_completed_release_status_is_synchronized(self) -> None:
        acceptance = ACCEPTANCE_MATRIX.read_text(encoding="utf-8")
        rows = [
            line
            for line in acceptance.splitlines()
            if re.match(r"^\| [A-Z]-[0-9]{2} \|", line)
        ]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(row=row):
                self.assertTrue(row.endswith("| Passed |"))
        self.assertIn(
            "sha256:193ef08f68f50a50c67f22b41ca2a31043c78d6b2311d23f16c588a86b80daee",
            acceptance,
        )

        project_state = PROJECT_STATE.read_text(encoding="utf-8")
        self.assertIn("| 当前稳定版本 | v0.6 Completed |", project_state)
        self.assertIn("| V06-RELEASE | Passed |", project_state)

        changelog = CHANGELOG.read_text(encoding="utf-8")
        self.assertIn("## v0.6 - Completed", changelog)


if __name__ == "__main__":
    unittest.main()
