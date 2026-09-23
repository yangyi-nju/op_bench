"""Exercise the fixed MCP tools through actual temporary execution sessions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from op_bench.runtime.execution import create_session, materialize_source
from op_bench.runtime.submission import freeze_patch, _git, initialize_workspace
from op_bench.runner.mcp import MCPServer, MCPToolClient
from op_bench.runner.loop import run_harness
from op_bench.data.task import TaskSpec
from op_bench.runner.tools import ToolExecutor
from _support import FakeModel, make_task


@unittest.skipUnless(shutil.which("git"), "Git is required for controlled workspaces")
class ControlledToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opbench-controlled-tools-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.task = TaskSpec.load(make_task(self.root / "task"))
        self.workspace = self.root / "workspace"
        materialize_source(self.task, self.workspace)
        self.database = self.root / "capture.git"
        initialize_workspace(self.workspace, self.database, ["operator_impl.py"])
        self.session = self.enterContext(create_session(self.task, self.workspace))
        self.tools = ToolExecutor(self.session, self.root / "tool-logs",
                                  reserved=(".opbench", "custom/home"))

    def call(self, name, arguments=None, *, tools=None):
        return (tools or self.tools).call(name, arguments or {}, 15)

    def test_read_search_and_file_lifecycle_are_visible_in_frozen_patch(self):
        source = "def identity(value):\n    return value\n"
        name = "operators/new [case].py"
        self.assertFalse(self.call("write_file", {"path": name, "content": source}).get("is_error"))
        read = self.call("read_file", {"path": name, "start_line": 2, "max_lines": 1})
        self.assertIn("2:     return value", read["text"])
        found = self.call("search", {"query": "def identity"})
        self.assertTrue(any(row["path"] == name and row["line"] == 1 for row in found["matches"]))
        self.call("replace_text", {"path": name, "old_text": "return value", "new_text": "return value + 1"})
        self.assertEqual(source.replace("return value", "return value + 1"), (self.workspace / name).read_text())
        self.call("delete_file", {"path": "operator_impl.py"})
        difference = self.call("diff")
        self.assertEqual(0, difference["exit_code"])
        self.assertIn("return value + 1", difference["diff"])
        self.assertIn("operator_impl.py", difference["diff"])
        frozen = freeze_patch(self.workspace, self.database)
        replay = self.root / "replay"
        materialize_source(self.task, replay)
        patch_path = self.root / "patch.diff"
        patch_path.write_text(frozen)
        _git(replay, "apply", str(patch_path))
        self.assertEqual((self.workspace / name).read_text(), (replay / name).read_text())
        self.assertFalse((replay / "operator_impl.py").exists())

    def test_file_contents_preserve_literal_task_placeholder_text(self):
        original = "PATHS = ('{workspace}', '{python}', '{grader}')\n"
        result = self.call("write_file", {"path": "literal_template.py", "content": original})
        self.assertFalse(result.get("is_error"))
        self.assertEqual(original, (self.workspace / "literal_template.py").read_text())
        replacement = "('{grader}', '{workspace}', '{python}')"
        changed = self.call("replace_text", {"path": "literal_template.py",
            "old_text": "('{workspace}', '{python}', '{grader}')", "new_text": replacement})
        self.assertFalse(changed.get("is_error"))
        self.assertEqual("PATHS = " + replacement + "\n", (self.workspace / "literal_template.py").read_text())

    def test_hidden_source_directories_are_listed_and_searchable(self):
        created = {
            ".github/workflows/build.yml": "name: operator-build\n",
            ".config/operator.py": "LABEL = 'operator-build'\n",
        }
        for name, content in created.items():
            self.assertFalse(self.call("write_file", {"path": name, "content": content}).get("is_error"))
        listed = self.call("list_files")["files"]
        self.assertTrue(set(created).issubset(listed))
        matched = {row["path"] for row in self.call("search", {"query": "operator-build"})["matches"]}
        self.assertEqual(set(created), matched)
        self.assertFalse(any(name.startswith(".git/") for name in listed))

    def test_git_symlink_aliases_cannot_expose_or_modify_git_internals(self):
        config = self.workspace / ".git/config"
        original = config.read_text()
        (self.workspace / "metadata-alias").symlink_to(self.workspace / ".git", target_is_directory=True)
        (self.workspace / "config-alias").symlink_to(config)
        for name in ("metadata-alias/config", "config-alias"):
            with self.subTest(path=name):
                self.assertTrue(self.call("read_file", {"path": name})["is_error"])
                self.assertTrue(self.call("write_file", {"path": name, "content": "replacement"})["is_error"])
                self.assertEqual(original, config.read_text())
        self.assertTrue(self.call("list_files", {"path": "metadata-alias"})["is_error"])
        self.assertTrue(self.call("search", {"path": "metadata-alias", "query": "core"})["is_error"])

    def test_long_public_feedback_is_truncated_without_killing_the_workspace(self):
        command = ("{python}", "-c", "print('PUBLIC START'); print('x' * 80000); print('PUBLIC END')")
        tools = ToolExecutor(self.session, self.root / "long-output-logs", public_commands=(command,))
        result = self.call("run_public_tests", tools=tools)
        output = result["results"][0]
        self.assertEqual(0, output["exit_code"])
        self.assertFalse(output["timed_out"])
        self.assertFalse(output["output_limited"])
        self.assertTrue(output["feedback_truncated"])
        self.assertIn("PUBLIC START", output["output"])
        self.assertIn("PUBLIC END", output["output"])
        self.assertLess(len(output["output"]), 80000)
        self.assertFalse(tools.finished)
        self.assertFalse(self.session.closed)
        self.assertIn("def row_sum", self.call("read_file", {"path": "operator_impl.py"}, tools=tools)["text"])
        logs = list((self.root / "long-output-logs").glob("*.log"))
        self.assertTrue(any("x" * 80000 in path.read_text() for path in logs))

    def test_pagination_and_replacement_ambiguity_preserve_source(self):
        self.call("write_file", {"path": "a.py", "content": "same\nsame\n"})
        self.call("write_file", {"path": "b.py", "content": "other\n"})
        seen, offset = [], 0
        while True:
            page = self.call("list_files", {"offset": offset, "limit": 1})
            seen.extend(page["files"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        self.assertEqual({"operator_impl.py", "a.py", "b.py"}, set(seen))
        self.assertEqual(len(seen), len(set(seen)))
        for old in ("same", "not present"):
            with self.subTest(old=old):
                rejected = self.call("replace_text", {"path": "a.py", "old_text": old, "new_text": "wrong"})
                self.assertTrue(rejected["is_error"])
                self.assertEqual("same\nsame\n", (self.workspace / "a.py").read_text())

    def test_escaping_and_external_symlink_paths_cannot_read_or_modify_host_file(self):
        outside = self.root / "outside.txt"
        outside.write_text("private fixture asset\n")
        (self.workspace / "linked.txt").symlink_to(outside)
        (self.workspace / "external").symlink_to(self.root, target_is_directory=True)
        paths = ("../outside.txt", str(outside), "linked.txt", "external/outside.txt")
        for name in paths:
            for operation, extra in (("read_file", {}), ("write_file", {"content": "changed"}),
                                     ("delete_file", {})):
                with self.subTest(path=name, operation=operation):
                    result = self.call(operation, {"path": name, **extra})
                    self.assertTrue(result["is_error"])
                    self.assertNotIn("private fixture asset", json.dumps(result))
                    self.assertEqual("private fixture asset\n", outside.read_text())
        listed = self.call("list_files")["files"]
        self.assertNotIn("linked.txt", listed)
        self.assertNotIn("external/outside.txt", listed)
        self.assertEqual([], self.call("search", {"query": "private fixture asset"})["matches"])

    def test_reserved_paths_and_aliases_reject_writes_without_blocking_sibling_source(self):
        public = self.workspace / ".opbench/task.md"
        public.parent.mkdir()
        public.write_text("public instructions\n")
        runtime = self.workspace / "custom/home"
        runtime.mkdir(parents=True)
        (runtime / "cache.txt").write_text("runtime state\n")
        (self.workspace / "runtime-alias").symlink_to(runtime, target_is_directory=True)
        self.assertIn("public instructions", self.call("read_file", {"path": ".opbench/task.md"})["text"])
        for name in (".opbench/task.md", ".git/config", "custom/home/cache.txt", "runtime-alias/cache.txt"):
            with self.subTest(path=name):
                result = self.call("write_file", {"path": name, "content": "changed"})
                self.assertTrue(result["is_error"])
        self.assertEqual("public instructions\n", public.read_text())
        self.assertEqual("runtime state\n", (runtime / "cache.txt").read_text())
        for name in ("custom/home/cache.txt", "runtime-alias/cache.txt", ".git/config"):
            self.assertTrue(self.call("read_file", {"path": name})["is_error"])
        allowed = self.call("write_file", {"path": "custom/helper.py", "content": "VALUE = 1\n"})
        self.assertFalse(allowed.get("is_error"))
        self.assertEqual("VALUE = 1\n", (self.workspace / "custom/helper.py").read_text())

    def test_bad_tool_calls_are_rejected_without_mutation(self):
        original = (self.workspace / "operator_impl.py").read_text()
        bad_calls = [
            ("shell", {"command": "touch escaped"}),
            ("write_file", {"path": "operator_impl.py"}),
            ("write_file", {"path": "operator_impl.py", "content": "changed", "command": "extra"}),
            ("write_file", {"path": "operator_impl.py", "content": None}),
            ("write_file", {"path": "operator_impl.py", "content": "bad\0source"}),
            ("write_file", {"path": "operator_impl.py", "content": "x" * 1000001}),
            ("read_file", {"path": "operator_impl.py", "start_line": True}),
            ("list_files", {"offset": -1}),
            ("search", {"query": ""}),
            ("build", {"argv": ["arbitrary"]}),
        ]
        for name, arguments in bad_calls:
            with self.subTest(name=name, arguments=arguments):
                self.assertTrue(self.call(name, arguments)["is_error"])
        for timeout in (0, -1, float("nan"), float("inf")):
            self.assertTrue(self.tools.call("delete_file", {"path": "operator_impl.py"}, timeout)["is_error"])
        self.assertEqual(original, (self.workspace / "operator_impl.py").read_text())
        self.assertFalse((self.workspace / "escaped").exists())
        self.assertFalse(self.tools.finished)

    def test_invalid_later_call_rejects_entire_model_response_before_writing(self):
        source = self.workspace / "operator_impl.py"
        original = source.read_text()
        for index, invalid_content in enumerate((None, "bad\0source", "x" * 1000001)):
            with self.subTest(case=index):
                response = {"content": "", "tool_calls": [
                    {"id": "valid", "name": "write_file", "arguments": {
                        "path": "operator_impl.py", "content": "unexpected mutation\n"}},
                    {"id": "invalid", "name": "write_file", "arguments": {
                        "path": "new.py", "content": invalid_content}},
                ]}
                result = run_harness(FakeModel(response), {}, MCPToolClient(self.tools),
                                     self.root / f"invalid-batch-{index}", 5)
                self.assertEqual(result["state"], "protocol_error")
                self.assertEqual(result["tool_calls"], 0)
                self.assertEqual(source.read_text(), original)
                self.assertFalse((self.workspace / "new.py").exists())

    def test_explicit_ignored_new_source_survives_capture_but_generated_and_runtime_do_not(self):
        workspace, database = self.root / "ignored-workspace", self.root / "ignored-capture.git"
        workspace.mkdir()
        (workspace / ".gitignore").write_text("ignored/\n")
        (workspace / "initial.py").write_text("VALUE = 0\n")
        (workspace / "build").mkdir()
        generated = "build/cached.py"
        (workspace / generated).write_text("GENERATED = 1\n")
        reserved = (".opbench", "custom/home")
        initialize_workspace(workspace, database, [".gitignore", "initial.py"], reserved=reserved)
        with create_session(self.task, workspace) as session:
            tools = ToolExecutor(session, self.root / "ignored-tool-logs", reserved=reserved)
            created = self.call("write_file", {"path": "ignored/new_source.py", "content": "VALUE = 2\n"}, tools=tools)
            self.assertFalse(created.get("is_error"))
            changed = self.call("replace_text", {"path": "ignored/new_source.py", "old_text": "2", "new_text": "3"}, tools=tools)
            self.assertFalse(changed.get("is_error"))
            self.call("write_file", {"path": generated, "content": "GENERATED = 999\n"}, tools=tools)
            for name in (".opbench/internal.txt", "custom/home/log.txt"):
                self.assertTrue(self.call("write_file", {"path": name, "content": "excluded"}, tools=tools)["is_error"])
                # Simulate build-created runtime artifacts, outside explicit tool edits.
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("runtime artifact\n")
        frozen = freeze_patch(workspace, database, reserved=reserved, generated=(generated,))
        replay = self.root / "ignored-replay"
        replay.mkdir()
        (replay / ".gitignore").write_text("ignored/\n")
        (replay / "initial.py").write_text("VALUE = 0\n")
        patch_path = self.root / "ignored.patch"
        patch_path.write_text(frozen)
        _git(replay, "apply", str(patch_path))
        self.assertEqual("VALUE = 3\n", (replay / "ignored/new_source.py").read_text())
        for name in (generated, ".opbench/internal.txt", "custom/home/log.txt"):
            self.assertFalse((replay / name).exists(), name)

    def test_declared_public_checks_are_feedback_and_only_finish_ends_solving(self):
        command = ("{python}", "-c", "from operator_impl import row_sum; assert row_sum([[1],[2]]) == [1,2]")
        build = ("{python}", "-c", "from pathlib import Path; Path('built.marker').write_text('built')")
        tools = ToolExecutor(self.session, self.root / "checks-logs", build_commands=(build,), public_commands=(command,))
        failure = self.call("run_public_tests", tools=tools)
        self.assertNotEqual(0, failure["results"][0]["exit_code"])
        self.assertIn("AssertionError", failure["results"][0]["output"])
        self.assertFalse(tools.finished)
        self.assertNotIn("resolved", failure)
        self.call("replace_text", {"path": "operator_impl.py", "old_text": "sum(rows[0])", "new_text": "sum(row)"}, tools=tools)
        built = self.call("build", tools=tools)
        self.assertEqual(0, built["results"][0]["exit_code"])
        self.assertEqual("built", (self.workspace / "built.marker").read_text())
        passed = self.call("run_public_tests", tools=tools)
        self.assertEqual(0, passed["results"][0]["exit_code"])
        self.assertFalse(tools.finished)
        self.assertNotIn("resolved", passed)
        final = self.call("finish", {"summary": "Ready for independent grading"}, tools=tools)
        self.assertTrue(final["submitted"])
        self.assertTrue(tools.finished)
        self.assertNotIn("resolved", final)
        self.assertTrue(self.call("delete_file", {"path": "operator_impl.py"}, tools=tools)["is_error"])
        self.assertTrue((self.workspace / "operator_impl.py").exists())

    def test_tools_cannot_be_constructed_with_private_grading_access(self):
        with create_session(self.task, self.workspace, self.task.grader_dir) as session:
            with self.assertRaisesRegex(ValueError, "grading session"):
                ToolExecutor(session, self.root / "invalid-tools")
        with self.assertRaisesRegex(ValueError, "private graders"):
            ToolExecutor(self.session, self.root / "invalid-public", public_commands=(("{python}", "{grader}/check.py"),))

    def test_local_commands_and_observations_do_not_inherit_host_credentials(self):
        key = "OPBENCH_TEST_HOST_SECRET"
        declared = "OPBENCH_TEST_PUBLIC_CONFIGURATION"
        task = replace(self.task, environment=replace(self.task.environment, environment={declared: "visible"}))
        program = (
            "import json,os; print(json.dumps({'secret_present': "
            f"{key!r} in os.environ, 'declared': os.getenv({declared!r})" + "}))"
        )
        with patch.dict(os.environ, {key: "fixture-only-do-not-inherit"}), create_session(task, self.workspace) as session:
            log = self.root / "clean-environment.log"
            outcome = session.run(("{python}", "-c", program), 10, log)
            self.assertEqual(0, outcome.exit_code)
            self.assertEqual({"secret_present": False, "declared": "visible"}, json.loads(log.read_text()))
            observation = self.root / "clean-observation.json"
            outcome = session.observe(("{python}", "-c", program), 10, b"{}", observation, self.root / "observation.log")
            self.assertEqual(0, outcome.exit_code)
            self.assertEqual({"secret_present": False, "declared": "visible"}, json.loads(observation.read_text()))

    def test_mcp_initialization_listing_calls_and_errors_use_same_workspace(self):
        server = MCPServer(self.tools)
        def request(identifier, method, params=None):
            return json.loads(server.exchange(json.dumps({"jsonrpc": "2.0", "id": identifier,
                                                         "method": method, "params": params or {}})))
        self.assertIn("error", request(1, "tools/list"))
        initialized = request(2, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
        self.assertIn("tools", initialized["result"]["capabilities"])
        self.assertIsNone(server.exchange(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})))
        definitions = request(3, "tools/list")["result"]["tools"]
        write = next(tool for tool in definitions if tool["name"] == "write_file")
        self.assertIn("content", write["inputSchema"]["required"])
        response = request(4, "tools/call", {"name": "write_file", "arguments": {"path": "from_mcp.py", "content": "VALUE = 7\n"}})
        self.assertEqual(4, response["id"])
        self.assertFalse(response["result"]["isError"])
        self.assertEqual("VALUE = 7\n", (self.workspace / "from_mcp.py").read_text())
        self.assertEqual(response["result"]["structuredContent"], json.loads(response["result"]["content"][0]["text"]))
        forbidden = request(5, "tools/call", {"name": "read_file", "arguments": {"path": "../private"}})
        self.assertTrue(forbidden["result"]["isError"])
        self.assertIn("error", request(6, "resources/read", {"uri": "file:///private"}))
        client = MCPToolClient(self.tools)
        self.assertIn("VALUE = 7", client.call("read_file", {"path": "from_mcp.py"}, 15)["text"])

if __name__ == "__main__":
    unittest.main()
