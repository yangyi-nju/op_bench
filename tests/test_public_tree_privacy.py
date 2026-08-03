from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from op_bench.public_tree_privacy import scan_public_tree


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "check_public_tree_privacy.py"


class PublicTreePrivacyTests(unittest.TestCase):
    def test_repository_tracked_public_tree_is_clean(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        tracked_paths = tuple(
            Path(value)
            for value in result.stdout.decode("utf-8").split("\0")
            if value
        )

        self.assertEqual(
            scan_public_tree(ROOT, tracked_paths=tracked_paths),
            (),
        )

    def test_scanner_reports_codes_without_echoing_private_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runs/release/evidence.json"
            path.parent.mkdir(parents=True)
            private_values = (
                "/Users/private-owner/.ssh/private-key",
                "10.24.7.9",
                "root@private-target",
                "sk-live-private-token-value",
            )
            path.write_text(
                "\n".join(
                    (
                        private_values[0],
                        private_values[1],
                        private_values[2],
                        private_values[3],
                    )
                ),
                encoding="utf-8",
            )

            findings = scan_public_tree(
                root,
                tracked_paths=(Path("runs/release/evidence.json"),),
            )

            self.assertEqual(
                {finding.code for finding in findings},
                {
                    "private.absolute_user_path",
                    "private.credential",
                    "private.ipv4",
                    "private.remote_identity",
                    "private.ssh_identity",
                },
            )
            rendered = repr(findings)
            for value in private_values:
                self.assertNotIn(value, rendered)

    def test_scanner_rejects_connection_fields_but_allows_placeholders(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unsafe = root / "docs/private-config.json"
            unsafe.parent.mkdir(parents=True)
            unsafe.write_text(
                '{"hostname":"private-target","remote_user":"owner",'
                '"identity_file":"/private/key"}',
                encoding="utf-8",
            )
            safe = root / "docs/example-config.json"
            safe.write_text(
                '{"hostname":"worker.example.invalid",'
                '"remote_user":"example-user",'
                '"identity_file":"/tmp/example-key"}',
                encoding="utf-8",
            )

            findings = scan_public_tree(
                root,
                tracked_paths=(
                    Path("docs/private-config.json"),
                    Path("docs/example-config.json"),
                ),
            )

            self.assertEqual(
                {finding.code for finding in findings},
                {
                    "private.hostname_field",
                    "private.identity_file_field",
                    "private.remote_user_field",
                },
            )
            self.assertEqual(
                {finding.path for finding in findings},
                {"docs/private-config.json"},
            )

    def test_scanner_allows_opaque_public_aliases_and_ignores_code_fixtures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = root / "environments/registry.json"
            environment.parent.mkdir(parents=True)
            environment.write_text(
                '{"host":"gpu-a10",'
                '"remote_execution_config_hash":"sha256:' + "a" * 64 + '"}\n'
                'author@example.com\n'
                '/Users/example/project /home/user/project\n'
                '~/.ssh/id_ed25519 ~/.ssh/example-key\n'
                '192.0.2.10 198.51.100.12 203.0.113.7',
                encoding="utf-8",
            )
            fixture = root / "tests/fixture.py"
            fixture.parent.mkdir(parents=True)
            fixture.write_text(
                'PRIVATE_FIXTURE = "/Users/example/.ssh/key"',
                encoding="utf-8",
            )

            findings = scan_public_tree(
                root,
                tracked_paths=(
                    Path("environments/registry.json"),
                    Path("tests/fixture.py"),
                ),
            )

            self.assertEqual(findings, ())

    def test_scanner_fails_closed_for_tracked_symlink(self) -> None:
        if os.name == "nt":
            self.skipTest("symlink behavior differs on Windows")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside.txt"
            target.write_text("safe", encoding="utf-8")
            link = root / "docs/link.txt"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)

            findings = scan_public_tree(
                root,
                tracked_paths=(Path("docs/link.txt"),),
            )

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].code, "tree.symlink")
            self.assertEqual(findings[0].path, "docs/link.txt")


class PublicTreePrivacyCliTests(unittest.TestCase):
    def _run(self, root: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, str(CLI), "--root", str(root)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def _initialize_index(self, root: Path) -> None:
        subprocess.run(
            ["git", "init", "--quiet", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "add", "."],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_cli_returns_one_and_emits_only_safe_finding_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private_value = "sk-private-value-that-must-not-be-echoed"
            artifact = root / "runs/release/result.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(private_value, encoding="utf-8")
            self._initialize_index(root)

            result = self._run(root)

            self.assertEqual(result.returncode, 1, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["finding_count"], 1)
            self.assertEqual(
                payload["findings"],
                [
                    {
                        "code": "private.credential",
                        "line": 1,
                        "path": "runs/release/result.json",
                    }
                ],
            )
            self.assertNotIn(private_value, result.stdout + result.stderr)

    def test_cli_returns_zero_for_clean_public_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "docs/example.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                '{"hostname":"worker.example.invalid"}',
                encoding="utf-8",
            )
            self._initialize_index(root)

            result = self._run(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                {"finding_count": 0, "findings": []},
            )

if __name__ == "__main__":
    unittest.main()
