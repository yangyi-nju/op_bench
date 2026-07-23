from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.actions import (
    CanonicalActionService,
    CommandExecution,
    RegisteredTest,
)
from op_bench.runtime.contracts import ACTION_NAMES, ActionRequest
from op_bench.runtime.workspace import AuthoritativeWorkspace
from tests.runtime_git_fixture import git, initialize_git_repo
from tests.test_runtime_contracts import (
    SHA_A,
    budget_policy,
    capability_policy,
    identity,
)
from tests.test_runtime_workspace import policy as workspace_policy


class FakeCommandBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, int]] = []

    def run(self, command: tuple[str, ...], cwd: str, timeout_ms: int) -> CommandExecution:
        self.calls.append((command, cwd, timeout_ms))
        if command[:2] == ("python", "-m"):
            stdout = "test-ok\n"
        else:
            stdout = "command-ok\n"
        return CommandExecution(
            command=command,
            cwd=cwd,
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_ms=7,
            timed_out=False,
        )


class ExplodingCommandBackend:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, command: tuple[str, ...], cwd: str, timeout_ms: int) -> CommandExecution:
        self.calls += 1
        raise RuntimeError("backend failed in /Users/private/runtime")


class MismatchedCommandBackend:
    def run(self, command: tuple[str, ...], cwd: str, timeout_ms: int) -> CommandExecution:
        return CommandExecution(
            command=("cat", "/etc/passwd"),
            cwd="/Users/private/runtime",
            exit_code=0,
            stdout="unexpected\n",
            stderr="",
            duration_ms=1,
            timed_out=False,
        )


class AdvancingCommandBackend:
    def __init__(self, now: list[int]) -> None:
        self.now = now
        self.timeouts: list[int] = []

    def run(self, command: tuple[str, ...], cwd: str, timeout_ms: int) -> CommandExecution:
        self.timeouts.append(timeout_ms)
        self.now[0] = 11
        return CommandExecution(
            command=command,
            cwd=cwd,
            exit_code=0,
            stdout="late\n",
            stderr="",
            duration_ms=11,
            timed_out=False,
        )


class SensitiveOutputBackend:
    def run(self, command: tuple[str, ...], cwd: str, timeout_ms: int) -> CommandExecution:
        return CommandExecution(
            command=command,
            cwd=cwd,
            exit_code=1,
            stdout="traceback at /Users/private/runtime/task.py\n",
            stderr="Authorization: Bearer private-token-value\n",
            duration_ms=1,
            timed_out=False,
        )


class CanonicalActionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)
        self.backend = FakeCommandBackend()
        self.service = self.make_service()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_service(self, *, capabilities=None, budget=None) -> CanonicalActionService:
        workspace = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@actions", SHA_A),
            policy=workspace_policy(),
        )
        selected_capabilities = capabilities or replace(
            capability_policy(),
            allowed_actions=ACTION_NAMES,
            writable_paths=("src/",),
            allowed_command_prefixes=("python -m unittest", "git diff"),
            registered_tests=("public::smoke",),
            max_read_bytes=1_024,
            max_write_bytes=2_048,
            max_output_bytes=1_024,
        )
        selected_budget = budget or replace(
            budget_policy(),
            wall_clock_ms=100_000,
            max_actions=100,
            max_tests=10,
            max_commands=10,
            max_output_bytes=10_000,
        )
        return CanonicalActionService(
            session_id="session-actions",
            workspace=workspace,
            capability_policy=selected_capabilities,
            budget_policy=selected_budget,
            command_backend=self.backend,
            test_registry={
                "public::smoke": RegisteredTest(
                    selector_id="public::smoke",
                    command=("python", "-m", "unittest", "tests.test_smoke"),
                    cwd=".",
                    timeout_ms=5_000,
                )
            },
            clock_ms=lambda: 1_000,
        )

    def request(
        self,
        action_id: str,
        action_name: str,
        arguments: dict[str, object],
        sequence: int,
        *,
        session_id: str = "session-actions",
        deadline_ms: int = 2_000,
    ) -> ActionRequest:
        return ActionRequest(
            session_id=session_id,
            action_id=action_id,
            action_name=action_name,
            arguments=arguments,
            client_sequence=sequence,
            deadline_ms=deadline_ms,
        )

    def test_all_actions_share_one_service_workspace_and_freeze_result(self) -> None:
        requests = (
            self.request("a01", "workspace_list", {"path": ".", "recursive": False}, 1),
            self.request(
                "a02",
                "workspace_search",
                {"path": "src", "query": "VALUE", "max_matches": 10},
                2,
            ),
            self.request("a03", "workspace_read", {"path": "src/operator.py"}, 3),
            self.request(
                "a04",
                "workspace_write",
                {"path": "src/operator.py", "content": "VALUE = 2\n"},
                4,
            ),
            self.request(
                "a05",
                "workspace_apply_patch",
                {
                    "patch": (
                        "diff --git a/src/helper.py b/src/helper.py\n"
                        "--- a/src/helper.py\n"
                        "+++ b/src/helper.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def helper():\n"
                        "-    return 1\n"
                        "+    return 2\n"
                    )
                },
                5,
            ),
            self.request(
                "a06",
                "command_run",
                {"command": ["git", "diff", "--stat"], "cwd": ".", "timeout_ms": 500},
                6,
            ),
            self.request("a07", "test_run", {"selector_id": "public::smoke"}, 7),
            self.request("a08", "vcs_diff", {}, 8),
            self.request("a09", "session_finish", {}, 9),
        )

        observations = [self.service.execute(request) for request in requests]

        self.assertTrue(all(item.ok for item in observations))
        self.assertEqual(observations[0].data["entries"][0]["path"], "src")
        self.assertEqual(observations[1].data["matches"][0]["path"], "src/operator.py")
        self.assertEqual(
            base64.b64decode(observations[2].data["content_base64"]),
            b"VALUE = 1\n",
        )
        self.assertEqual(observations[3].mutation_state, "mutated")
        self.assertEqual(observations[4].mutation_state, "mutated")
        self.assertEqual(observations[5].data["stdout"], "command-ok\n")
        self.assertEqual(observations[6].data["selector_id"], "public::smoke")
        diff_bytes = base64.b64decode(observations[7].data["patch_base64"])
        self.assertIn(b"VALUE = 2", diff_bytes)
        self.assertEqual(observations[8].mutation_state, "frozen")
        self.assertEqual(observations[8].data["patch"], observations[7].data["patch"])
        self.assertEqual(self.service.workspace_identity, requests and self.service.workspace_identity)
        self.assertEqual(self.service.usage.actions, 9)
        self.assertEqual(self.service.usage.commands, 1)
        self.assertEqual(self.service.usage.tests, 1)
        self.assertEqual(len(self.service.audit_exchanges), 9)

    def test_workspace_search_skips_non_utf8_files(self) -> None:
        root = Path(self.temporary.name) / "binary-search-repo"
        initialize_git_repo(root)
        (root / "tests" / "accuracy.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        git(root, "add", "tests/accuracy.png")
        git(root, "commit", "--quiet", "-m", "add binary fixture")
        workspace = AuthoritativeWorkspace.open(
            root,
            source=identity("source", "fixture@binary-search", SHA_A),
            policy=workspace_policy(),
        )
        service = CanonicalActionService(
            session_id="session-actions",
            workspace=workspace,
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("workspace_search",),
                max_read_bytes=1_024,
                max_output_bytes=1_024,
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=100_000,
                max_actions=10,
                max_output_bytes=10_000,
            ),
            command_backend=self.backend,
            test_registry={},
            clock_ms=lambda: 1_000,
        )

        observation = service.execute(
            self.request(
                "search-with-binary",
                "workspace_search",
                {"path": ".", "query": "VALUE", "max_files": 10},
                1,
            )
        )

        self.assertTrue(observation.ok)
        self.assertEqual(observation.error_code, "ok")
        self.assertEqual(
            [match["path"] for match in observation.data["matches"]],
            ["src/operator.py"],
        )

    def test_idempotency_prevents_repeated_mutation_command_test_and_finish(self) -> None:
        write = self.request(
            "same-write",
            "workspace_write",
            {"path": "src/operator.py", "content": "VALUE = 2\n"},
            1,
        )
        command = self.request(
            "same-command",
            "command_run",
            {"command": ["git", "diff"], "cwd": "."},
            2,
        )
        test = self.request("same-test", "test_run", {"selector_id": "public::smoke"}, 3)
        finish = self.request("same-finish", "session_finish", {}, 4)

        for request in (write, command, test, finish):
            first = self.service.execute(request)
            second = self.service.execute(request)
            self.assertIs(first, second)

        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(len(self.service.audit_exchanges), 4)
        conflict = self.service.execute(
            self.request(
                "same-write",
                "workspace_write",
                {"path": "src/operator.py", "content": "VALUE = 3\n"},
                1,
            )
        )
        self.assertFalse(conflict.ok)
        self.assertEqual(conflict.error_code, "conflict")
        self.assertEqual((self.root / "src" / "operator.py").read_text(), "VALUE = 2\n")

    def test_every_request_revalidates_session_capability_path_selector_and_state(self) -> None:
        cases = (
            (
                self.request("wrong-session", "workspace_read", {"path": "src/operator.py"}, 1, session_id="other"),
                "session_not_running",
            ),
            (
                self.request("path", "workspace_write", {"path": "tests/test_operator.py", "content": "bad\n"}, 2),
                "path_denied",
            ),
            (
                self.request("selector", "test_run", {"selector_id": "hidden::answer"}, 3),
                "selector_denied",
            ),
            (
                self.request("deadline", "workspace_read", {"path": "src/operator.py"}, 4, deadline_ms=999),
                "timeout",
            ),
        )
        for request, error_code in cases:
            with self.subTest(error_code=error_code):
                observation = self.service.execute(request)
                self.assertFalse(observation.ok)
                self.assertEqual(observation.error_code, error_code)

        recovered = self.service.execute(
            self.request(
                "wrong-session",
                "workspace_read",
                {"path": "src/operator.py"},
                4,
            )
        )
        self.assertTrue(recovered.ok)

        self.assertTrue(self.service.execute(self.request("finish", "session_finish", {}, 5)).ok)
        after_finish = self.service.execute(
            self.request("late-write", "workspace_write", {"path": "src/new.py", "content": "x\n"}, 6)
        )
        self.assertEqual(after_finish.error_code, "session_not_running")

        denied_capability = replace(capability_policy(), allowed_actions=("workspace_read",))
        other_root = Path(self.temporary.name) / "capability-repo"
        initialize_git_repo(other_root)
        original_root = self.root
        self.root = other_root
        try:
            denied_service = self.make_service(capabilities=denied_capability)
        finally:
            self.root = original_root
        denied = denied_service.execute(
            self.request("denied", "workspace_write", {"path": "src/operator.py", "content": "x\n"}, 1)
        )
        self.assertEqual(denied.error_code, "capability_denied")

    def test_command_policy_is_argv_prefix_based_and_test_is_registry_bound(self) -> None:
        allowed = self.service.execute(
            self.request(
                "allowed-command",
                "command_run",
                {"command": ["python", "-m", "unittest", "tests.test_one"], "cwd": "src"},
                1,
            )
        )
        denied_requests = (
            self.request("shell", "command_run", {"command": "python -m unittest"}, 2),
            self.request("prefix", "command_run", {"command": ["python-malicious", "-m", "unittest"]}, 3),
            self.request("cwd", "command_run", {"command": ["git", "diff"], "cwd": "../"}, 4),
            self.request("raw-test", "test_run", {"selector_id": "python -c pass"}, 5),
            self.request(
                "outside-argument",
                "command_run",
                {
                    "command": [
                        "git",
                        "diff",
                        "--no-index",
                        "/etc/passwd",
                        "src/operator.py",
                    ]
                },
                6,
            ),
            self.request(
                "unittest-equals-path",
                "command_run",
                {
                    "command": [
                        "python",
                        "-m",
                        "unittest",
                        "discover",
                        "--start-directory=/etc",
                    ]
                },
                7,
            ),
            self.request(
                "unittest-short-path",
                "command_run",
                {
                    "command": [
                        "python",
                        "-m",
                        "unittest",
                        "discover",
                        "-s/etc",
                    ]
                },
                8,
            ),
            self.request(
                "git-orderfile",
                "command_run",
                {"command": ["git", "diff", "-O/etc/passwd"]},
                9,
            ),
        )

        self.assertTrue(allowed.ok)
        self.assertEqual(self.backend.calls[0][1], "src")
        observations = [self.service.execute(item) for item in denied_requests]
        self.assertEqual(
            [item.error_code for item in observations],
            [
                "invalid_request",
                "capability_denied",
                "path_denied",
                "selector_denied",
                "capability_denied",
                "capability_denied",
                "capability_denied",
                "capability_denied",
            ],
        )
        self.assertEqual(len(self.backend.calls), 1)

    def test_longest_command_prefix_wins_independent_of_policy_order(self) -> None:
        root = Path(self.temporary.name) / "overlap-prefix-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        capabilities = replace(
            capability_policy(),
            allowed_actions=("command_run",),
            allowed_command_prefixes=("python", "python -m unittest"),
        )
        try:
            service = self.make_service(capabilities=capabilities)
        finally:
            self.root = original_root

        observation = service.execute(
            self.request(
                "overlap-prefix",
                "command_run",
                {"command": ["python", "-m", "unittest", "tests.test_one"]},
                1,
            )
        )

        self.assertTrue(observation.ok)

    def test_budget_and_output_limits_are_server_enforced(self) -> None:
        limited_budget = replace(
            budget_policy(),
            wall_clock_ms=100_000,
            max_actions=2,
            max_tests=0,
            max_commands=0,
            max_output_bytes=5,
        )
        other_root = Path(self.temporary.name) / "budget-repo"
        initialize_git_repo(other_root)
        original_root = self.root
        self.root = other_root
        try:
            limited = self.make_service(budget=limited_budget)
        finally:
            self.root = original_root

        read = limited.execute(
            self.request("read", "workspace_read", {"path": "src/operator.py"}, 1)
        )
        test = limited.execute(
            self.request("test", "test_run", {"selector_id": "public::smoke"}, 2)
        )
        command = limited.execute(
            self.request("command", "command_run", {"command": ["git", "diff"]}, 3)
        )
        extra = limited.execute(
            self.request("extra", "workspace_list", {"path": "."}, 4)
        )

        self.assertTrue(read.ok)
        self.assertEqual(read.budget_delta.output_bytes, 5)
        self.assertTrue(read.data["truncated"])
        self.assertEqual(test.error_code, "budget_exhausted")
        self.assertEqual(command.error_code, "budget_exhausted")
        self.assertEqual(extra.error_code, "budget_exhausted")

        zero_root = Path(self.temporary.name) / "zero-output-repo"
        initialize_git_repo(zero_root)
        zero_budget = replace(limited_budget, max_actions=1, max_output_bytes=0)
        self.root = zero_root
        try:
            zero_output = self.make_service(budget=zero_budget)
        finally:
            self.root = original_root
        denied_list = zero_output.execute(
            self.request("zero-list", "workspace_list", {"path": "."}, 1)
        )
        self.assertEqual(denied_list.error_code, "budget_exhausted")

    def test_invalid_arguments_and_failed_patch_have_no_partial_mutation(self) -> None:
        unknown = self.service.execute(
            self.request("unknown-arg", "workspace_read", {"path": "src/operator.py", "extra": True}, 1)
        )
        bad_patch = self.service.execute(
            self.request(
                "bad-patch",
                "workspace_apply_patch",
                {
                    "patch": (
                        "diff --git a/src/operator.py b/src/operator.py\n"
                        "--- a/src/operator.py\n"
                        "+++ b/src/operator.py\n"
                        "@@ -1 +1 @@\n"
                        "-NOT CURRENT\n"
                        "+VALUE = 2\n"
                    )
                },
                2,
            )
        )

        self.assertEqual(unknown.error_code, "invalid_request")
        self.assertEqual(bad_patch.error_code, "invalid_request")
        self.assertEqual((self.root / "src" / "operator.py").read_text(), "VALUE = 1\n")

    def test_patch_capability_is_checked_before_workspace_mutation(self) -> None:
        other_root = Path(self.temporary.name) / "narrow-patch-repo"
        initialize_git_repo(other_root)
        narrow_capabilities = replace(
            capability_policy(),
            allowed_actions=("workspace_apply_patch",),
            writable_paths=("src/operator.py",),
        )
        original_root = self.root
        self.root = other_root
        try:
            narrow = self.make_service(capabilities=narrow_capabilities)
        finally:
            self.root = original_root
        denied = narrow.execute(
            self.request(
                "narrow-patch",
                "workspace_apply_patch",
                {
                    "patch": (
                        "diff --git a/src/helper.py b/src/helper.py\n"
                        "--- a/src/helper.py\n"
                        "+++ b/src/helper.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def helper():\n"
                        "-    return 1\n"
                        "+    return 2\n"
                    )
                },
                1,
            )
        )

        self.assertEqual(denied.error_code, "path_denied")
        self.assertEqual(
            (other_root / "src" / "helper.py").read_text(),
            "def helper():\n    return 1\n",
        )

    def test_concurrent_duplicate_action_executes_backend_once(self) -> None:
        request = self.request(
            "concurrent-command",
            "command_run",
            {"command": ["git", "diff"], "cwd": "."},
            1,
        )
        barrier = threading.Barrier(3)
        observations = []

        def call() -> None:
            barrier.wait()
            observations.append(self.service.execute(request))

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(len(observations), 2)
        self.assertIs(observations[0], observations[1])
        self.assertEqual(len(self.backend.calls), 1)

    def test_backend_failure_consumes_command_budget_without_leaking_exception_paths(self) -> None:
        root = Path(self.temporary.name) / "exploding-backend-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        backend = ExplodingCommandBackend()
        self.backend = backend  # type: ignore[assignment]
        try:
            service = self.make_service(
                budget=replace(budget_policy(), max_actions=10, max_commands=1)
            )
        finally:
            self.root = original_root
        first = service.execute(
            self.request("explode-1", "command_run", {"command": ["git", "diff"]}, 1)
        )
        second = service.execute(
            self.request("explode-2", "command_run", {"command": ["git", "diff"]}, 2)
        )

        self.assertEqual(first.error_code, "runtime_error")
        self.assertNotIn("/Users/", first.message)
        self.assertEqual(first.budget_delta.commands, 1)
        self.assertEqual(second.error_code, "budget_exhausted")
        self.assertEqual(backend.calls, 1)

    def test_backend_cannot_replace_authorized_command_or_cwd_in_observation(self) -> None:
        root = Path(self.temporary.name) / "mismatch-backend-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        self.backend = MismatchedCommandBackend()  # type: ignore[assignment]
        try:
            service = self.make_service()
        finally:
            self.root = original_root

        observation = service.execute(
            self.request("mismatch", "command_run", {"command": ["git", "diff"]}, 1)
        )

        self.assertEqual(observation.error_code, "runtime_error")
        self.assertEqual(observation.data, {})
        self.assertNotIn("/Users/", observation.message)
        self.assertNotIn("/etc/passwd", observation.message)

    def test_command_timeout_is_clamped_to_session_wall_budget_and_late_success_fails(self) -> None:
        root = Path(self.temporary.name) / "wall-budget-repo"
        initialize_git_repo(root)
        now = [0]
        backend = AdvancingCommandBackend(now)
        service = CanonicalActionService(
            session_id="session-actions",
            workspace=AuthoritativeWorkspace.open(
                root,
                source=identity("source", "fixture@wall", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("command_run",),
                allowed_command_prefixes=("git diff",),
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=10,
                max_actions=10,
                max_commands=1,
            ),
            command_backend=backend,
            test_registry={},
            clock_ms=lambda: now[0],
        )

        observation = service.execute(
            self.request(
                "wall-command",
                "command_run",
                {"command": ["git", "diff"], "timeout_ms": 4_000},
                1,
                deadline_ms=5_000,
            )
        )

        self.assertEqual(backend.timeouts, [10])
        self.assertEqual(observation.error_code, "budget_exhausted")
        self.assertEqual(observation.budget_delta.commands, 1)

    def test_backend_output_is_scanned_before_agent_observation(self) -> None:
        root = Path(self.temporary.name) / "sensitive-output-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        self.backend = SensitiveOutputBackend()  # type: ignore[assignment]
        try:
            service = self.make_service()
        finally:
            self.root = original_root

        observation = service.execute(
            self.request("sensitive-output", "command_run", {"command": ["git", "diff"]}, 1)
        )

        self.assertTrue(observation.ok)
        self.assertNotIn("/Users/", observation.data["stdout"])
        self.assertNotIn("private-token", observation.data["stderr"])
        self.assertTrue(observation.data["truncated"])

    def test_registered_test_command_metadata_is_not_returned_to_adapter(self) -> None:
        root = Path(self.temporary.name) / "private-test-command-repo"
        initialize_git_repo(root)
        backend = FakeCommandBackend()
        service = CanonicalActionService(
            session_id="session-actions",
            workspace=AuthoritativeWorkspace.open(
                root,
                source=identity("source", "fixture@private-test", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("test_run",),
                registered_tests=("public::private-runtime",),
            ),
            budget_policy=replace(budget_policy(), max_actions=10, max_tests=1),
            command_backend=backend,
            test_registry={
                "public::private-runtime": RegisteredTest(
                    selector_id="public::private-runtime",
                    command=(
                        "/Users/private/venv/bin/python",
                        "-m",
                        "unittest",
                    ),
                    cwd=".",
                    timeout_ms=1_000,
                )
            },
            clock_ms=lambda: 1_000,
        )

        observation = service.execute(
            self.request(
                "private-test",
                "test_run",
                {"selector_id": "public::private-runtime"},
                1,
            )
        )

        self.assertTrue(observation.ok)
        self.assertEqual(observation.data["selector_id"], "public::private-runtime")
        self.assertNotIn("command", observation.data)
        self.assertNotIn("cwd", observation.data)
        self.assertNotIn("/Users/", repr(observation.to_dict()))

    def test_post_execution_deadline_applies_to_read_and_preserves_late_mutation_state(self) -> None:
        root = Path(self.temporary.name) / "post-deadline-repo"
        initialize_git_repo(root)
        now = [0]
        workspace = AuthoritativeWorkspace.open(
            root,
            source=identity("source", "fixture@post-deadline", SHA_A),
            policy=workspace_policy(),
        )
        service = CanonicalActionService(
            session_id="session-actions",
            workspace=workspace,
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("workspace_read", "workspace_write"),
                writable_paths=("src/",),
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=1_000,
                max_actions=10,
            ),
            command_backend=FakeCommandBackend(),
            test_registry={},
            clock_ms=lambda: now[0],
        )
        original_read = workspace.read

        def late_read(*args, **kwargs):
            result = original_read(*args, **kwargs)
            now[0] = 10
            return result

        with mock.patch.object(workspace, "read", side_effect=late_read):
            read = service.execute(
                self.request(
                    "late-read",
                    "workspace_read",
                    {"path": "src/operator.py"},
                    1,
                    deadline_ms=10,
                )
            )
        original_write = workspace.write

        def late_write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            now[0] = 20
            return result

        with mock.patch.object(workspace, "write", side_effect=late_write):
            write = service.execute(
                self.request(
                    "late-write",
                    "workspace_write",
                    {"path": "src/operator.py", "content": "VALUE = 2\n"},
                    2,
                    deadline_ms=20,
                )
            )

        self.assertEqual(read.error_code, "timeout")
        self.assertEqual(read.mutation_state, "none")
        self.assertEqual(write.error_code, "timeout")
        self.assertEqual(write.mutation_state, "mutated")
        self.assertEqual((root / "src" / "operator.py").read_text(), "VALUE = 2\n")

    def test_cwd_validation_deadline_is_reported_as_timeout_before_backend(self) -> None:
        root = Path(self.temporary.name) / "cwd-deadline-repo"
        initialize_git_repo(root)
        now = [0]
        backend = FakeCommandBackend()
        workspace = AuthoritativeWorkspace.open(
            root,
            source=identity("source", "fixture@cwd-deadline", SHA_A),
            policy=workspace_policy(),
        )
        service = CanonicalActionService(
            session_id="session-actions",
            workspace=workspace,
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("command_run",),
                allowed_command_prefixes=("git diff",),
            ),
            budget_policy=replace(budget_policy(), wall_clock_ms=1_000, max_commands=1),
            command_backend=backend,
            test_registry={},
            clock_ms=lambda: now[0],
        )
        original_list = workspace.list_entries

        def late_list(*args, **kwargs):
            result = original_list(*args, **kwargs)
            now[0] = 10
            return result

        with mock.patch.object(workspace, "list_entries", side_effect=late_list):
            observation = service.execute(
                self.request(
                    "cwd-deadline",
                    "command_run",
                    {"command": ["git", "diff"], "cwd": "."},
                    1,
                    deadline_ms=10,
                )
            )

        self.assertEqual(observation.error_code, "timeout")
        self.assertEqual(backend.calls, [])

    def test_first_finish_is_budget_exempt_but_later_finish_is_fully_validated_and_bounded(self) -> None:
        root = Path(self.temporary.name) / "finish-budget-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        try:
            service = self.make_service(
                budget=replace(budget_policy(), max_actions=1)
            )
        finally:
            self.root = original_root
        first_request = self.request("finish-first", "session_finish", {}, 1)
        first = service.execute(first_request)
        duplicate = service.execute(first_request)
        later = service.execute(
            self.request("finish-later", "session_finish", {}, 2)
        )

        self.assertTrue(first.ok)
        self.assertIs(first, duplicate)
        self.assertEqual(later.error_code, "budget_exhausted")
        self.assertEqual(service.usage.actions, 1)
        self.assertEqual(len(service.audit_exchanges), 2)

    def test_finish_exemption_is_one_shot_and_not_granted_to_invalid_or_failed_retries(self) -> None:
        root = Path(self.temporary.name) / "finish-exemption-repo"
        initialize_git_repo(root)
        original_root = self.root
        self.root = root
        try:
            service = self.make_service(
                budget=replace(budget_policy(), max_actions=0)
            )
        finally:
            self.root = original_root
        invalid = [
            service.execute(
                self.request(
                    f"invalid-finish-{index}",
                    "session_finish",
                    {"unexpected": index},
                    1,
                )
            )
            for index in range(4)
        ]
        valid = service.execute(
            self.request("valid-finish", "session_finish", {}, 1)
        )

        self.assertTrue(all(item.error_code == "invalid_request" for item in invalid))
        self.assertTrue(valid.ok)
        self.assertEqual(service.usage.actions, 1)
        self.assertEqual(len(service.audit_exchanges), 1)

        failed_root = Path(self.temporary.name) / "failed-finish-repo"
        initialize_git_repo(failed_root)
        failed_workspace = AuthoritativeWorkspace.open(
            failed_root,
            source=identity("source", "fixture@failed-finish", SHA_A),
            policy=workspace_policy(max_patch_bytes=32),
        )
        failed_workspace.write(
            "src/operator.py",
            "VALUE = 'large enough to exceed the tiny patch budget'\n",
        )
        failed_service = CanonicalActionService(
            session_id="session-actions",
            workspace=failed_workspace,
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("session_finish",),
            ),
            budget_policy=replace(budget_policy(), max_actions=0),
            command_backend=FakeCommandBackend(),
            test_registry={},
            clock_ms=lambda: 1_000,
        )
        failed = failed_service.execute(
            self.request("failed-finish", "session_finish", {}, 1)
        )
        retry = failed_service.execute(
            self.request("failed-finish-retry", "session_finish", {}, 2)
        )

        self.assertEqual(failed.error_code, "workspace_error")
        self.assertEqual(retry.error_code, "budget_exhausted")
        self.assertEqual(failed_service.usage.actions, 1)


if __name__ == "__main__":
    unittest.main()
