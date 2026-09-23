from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from op_bench.runner.loop import run_harness
from op_bench.runner.model_client import ModelError


DEFINITIONS = [{"type": "function", "function": {"name": name, "description": name,
    "parameters": {"type": "object", "properties": {field: {"type": "string"}},
                   "required": [field], "additionalProperties": False}}}
    for name, field in [("read_file", "path"), ("finish", "summary")]]


def decision(name="finish", arguments=None, usage=None):
    return {"content": "", "tool_calls": [{"id": "call-1", "name": name,
            "arguments": arguments or {"summary": "done"}}], "usage": usage}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
    def complete(self, messages, tools, timeout_sec):
        self.messages.append(json.loads(json.dumps(messages)))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class Tools:
    definitions = DEFINITIONS
    def __init__(self):
        self.finished = False
        self.calls = []
    def call(self, name, arguments, timeout_sec):
        self.calls.append((name, arguments, timeout_sec))
        if name == "finish":
            self.finished = True
        return {"ok": True, "content": "public source"}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run"
        self.tools = Tools()

    def run_loop(self, responses, **kwargs):
        self.client = Client(responses)
        return run_harness(self.client, {"description": "repair"}, self.tools, self.output, kwargs.pop("budget_sec", 5), **kwargs)

    def test_fixed_loop_passes_tool_observation_and_records_trace(self):
        result = self.run_loop([decision("read_file", {"path": "source.py"}), decision()])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(self.client.messages[1][-1]["role"], "tool")
        self.assertEqual(self.client.messages[1][-1]["tool_call_id"], "call-1")
        self.assertIn("public source", self.client.messages[1][-1]["content"])
        rows = [json.loads(line) for line in Path(result["trace_path"]).read_text().splitlines()]
        self.assertEqual(rows[0]["event"], "started")
        self.assertEqual(rows[-1]["event"], "finished")
        self.assertEqual(json.loads((self.output / "harness-result.json").read_text())["state"], "completed")

    def test_illegal_tool_never_executes(self):
        result = self.run_loop([decision("shell", {"command": "cat private"})])
        self.assertEqual(result["state"], "protocol_error")
        self.assertEqual(result["error"], "unknown_tool")
        self.assertEqual(self.tools.calls, [])

    def test_invalid_arguments_never_execute(self):
        for index, value in enumerate((3, "file\0name", "x" * 1000001)):
            with self.subTest(case=index):
                self.output = self.output.parent / f"invalid-{index}"
                result = self.run_loop([decision("read_file", {"path": value})])
                self.assertEqual(result["state"], "protocol_error")
                self.assertEqual(self.tools.calls, [])

    def test_finish_mixed_with_edit_rejected_as_whole_message(self):
        response = decision()
        response["tool_calls"].append({"id": "second", "name": "read_file", "arguments": {"path": "x"}})
        result = self.run_loop([response])
        self.assertEqual(result["error"], "finish_must_be_only_call")
        self.assertEqual(self.tools.calls, [])

    def test_missing_usage_stays_unknown_instead_of_zero(self):
        result = self.run_loop([decision("read_file", {"path": "x"}, {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}), decision()])
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(result["usage"]["known_totals"]["total_tokens"], 7)
        self.assertEqual(result["usage"]["known_calls"]["total_tokens"], 1)
        self.assertIsNone(result["usage"]["cost_usd"])

    def test_budget_counts_model_time_and_does_not_execute_late_decision(self):
        class Slow:
            def complete(self, *args):
                time.sleep(0.03)
                return decision()
        result = run_harness(Slow(), {}, self.tools, self.output, 0.01)
        self.assertEqual(result["state"], "timeout")
        self.assertEqual(self.tools.calls, [])

    def test_transport_failure_and_timeout_are_distinct(self):
        result = self.run_loop([ModelError("bad_provider")])
        self.assertEqual((result["state"], result["error"]), ("model_error", "bad_provider"))
        self.output = self.output.parent / "timeout"
        result = self.run_loop([ModelError("model_deadline", timeout=True)])
        self.assertEqual(result["state"], "timeout")

    def test_invalid_model_output_is_protocol_terminal_and_links_raw_evidence(self):
        evidence = self.output.parent / "public-response.json"
        evidence.write_text('{"message":"invalid decision"}')
        result = self.run_loop([ModelError("model_invalid_tool_arguments", invalid_output=True,
                                         evidence_path=str(evidence))])
        self.assertEqual((result["state"], result["error"]), ("protocol_error", "model_invalid_tool_arguments"))
        self.assertEqual(self.tools.calls, [])
        rows = [json.loads(line) for line in Path(result["trace_path"]).read_text().splitlines()]
        error = next(row for row in rows if row["event"] == "model_error")
        self.assertTrue(error["invalid_output"])
        self.assertEqual(error["evidence_path"], str(evidence))

    def test_context_and_turn_limits_have_explicit_terminal_states(self):
        result = self.run_loop([], max_context_chars=1)
        self.assertEqual(result["state"], "context_limit")
        self.assertEqual(result["model_calls"], 0)
        self.output = self.output.parent / "turns"
        result = self.run_loop([decision("read_file", {"path": "x"})], max_turns=1)
        self.assertEqual(result["state"], "turn_limit")

    def test_plain_text_does_not_imply_success_or_finish(self):
        result = self.run_loop([{"content": "fixed", "tool_calls": []}])
        self.assertEqual(result["error"], "finish_required")

    def test_public_check_error_is_observation_not_harness_failure(self):
        original = self.tools.call
        def invoke(name, arguments, timeout_sec):
            result = original(name, arguments, timeout_sec)
            return {"ok": False, "error": "public_test_failed"} if name != "finish" else result
        self.tools.call = invoke
        result = self.run_loop([decision("read_file", {"path": "x"}), decision()])
        self.assertEqual(result["state"], "completed")
        self.assertIn("public_test_failed", self.client.messages[1][-1]["content"])

    def test_closed_tool_session_stops_without_another_model_request(self):
        for timed_out, expected in [(True, "timeout"), (False, "tool_error")]:
            with self.subTest(timed_out=timed_out):
                self.output = self.output.parent / ("closed-timeout" if timed_out else "closed-error")
                self.tools = Tools()
                self.tools.call = lambda *args: {"session_closed": True, "timed_out": timed_out, "is_error": True}
                result = self.run_loop([decision("read_file", {"path": "x"})])
                self.assertEqual(result["state"], expected)
                self.assertEqual(result["model_calls"], 1)
                self.assertEqual(result["tool_calls"], 1)
                rows = [json.loads(line) for line in Path(result["trace_path"]).read_text().splitlines()]
                self.assertEqual(len([row for row in rows if row["event"] == "tool_result"]), 1)

    def test_existing_trajectory_cannot_be_overwritten(self):
        self.run_loop([decision()])
        with self.assertRaises(FileExistsError):
            self.run_loop([decision()])


if __name__ == "__main__":
    unittest.main()
