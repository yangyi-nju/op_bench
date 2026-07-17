from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
README_FILES = (ROOT / "README.md", ROOT / "README.zh-CN.md")
INDEX_FILES = (ROOT / "docs" / "README.md", ROOT / "docs" / "README.zh-CN.md")
GUIDE = ROOT / "docs" / "v0.6" / "developer_guide.md"
M7_VERIFICATION = ROOT / "docs" / "v0.6" / "m7_verification.md"
DEMO_ARTIFACT = (
    ROOT / "configs" / "examples" / "v0.6_scripted_demo_artifact.example.json"
)
RELEASE_NOTES = ROOT / "docs" / "v0.6" / "release_notes.md"
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class V06ReleaseDocumentationTests(unittest.TestCase):
    def test_repository_relative_links_resolve(self) -> None:
        for document in (
            *README_FILES,
            *INDEX_FILES,
            GUIDE,
            RELEASE_NOTES,
            M7_VERIFICATION,
        ):
            text = document.read_text(encoding="utf-8")
            for raw_target in LINK_PATTERN.findall(text):
                target = raw_target.strip().strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target).resolve().exists())

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
            self.assertIn("Blocked", text)
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
        self.assertIn("Decision: **Blocked", release)
        self.assertIn("not a v0.6 score", release)
        self.assertIn("not a formal Agent", release)
        self.assertIn("does not run the planned feedback-causality", release)
        self.assertNotIn("Decision: **Completed", release)


if __name__ == "__main__":
    unittest.main()
