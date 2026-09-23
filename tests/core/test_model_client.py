from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from op_bench.runner.model_client import CodexCLIClient, ChatCompletionsClient, ModelError, ModelSpec, _decision_schema


TOOLS = [{"type": "function", "function": {"name": "finish", "description": "finish",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "optional": {"type": "integer"}},
                   "required": ["summary"], "additionalProperties": False}}}]


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def server(self, *, status=200, response=None, delay=0, content_type="application/json"):
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                observed.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                 "authorization": self.headers.get("Authorization")})
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                if delay:
                    time.sleep(delay)
                try:
                    self.wfile.write(json.dumps(response).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join()
        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}/v1", observed

    def test_spec_is_explicit_and_roundtrips_without_credentials(self):
        spec = ModelSpec("small", "codex_cli", "gpt-5.5", reasoning_effort="high")
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)
        for value in ({"model_id": "a", "backend": "native", "model": "x"},
                      {"model_id": "../escape", "backend": "codex_cli", "model": "x"},
                      {"model_id": "a", "backend": "codex_cli", "model": "x", "api_key": "secret"}):
            with self.assertRaises(ValueError):
                ModelSpec.from_dict(value)

    def test_chat_response_normalizes_decision_and_usage(self):
        response = {"id": "response-1", "model": "model-revision", "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "f", "type": "function", "function": {
            "name": "finish", "arguments": '{"summary":"done"}'}}]}}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
        url, observed = self.server(response=response)
        client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "selected", base_url=url), self.root)
        result = client.complete([{"role": "user", "content": "task"}], TOOLS, 5)
        self.assertEqual(result["tool_calls"][0]["arguments"], {"summary": "done"})
        self.assertEqual(result["usage"]["total_tokens"], 6)
        self.assertEqual(observed[0]["body"]["model"], "selected")
        self.assertFalse(observed[0]["body"]["parallel_tool_calls"])
        self.assertEqual(observed[0]["path"], "/v1/chat/completions")
        self.assertIsNone(observed[0]["authorization"])

    def test_provider_error_does_not_copy_body_or_retry(self):
        url, observed = self.server(status=429, response={"error": "private-provider-data"})
        client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url), self.root)
        with self.assertRaisesRegex(ModelError, "^model_http_status_429$"):
            client.complete([], TOOLS, 1)
        self.assertEqual(len(observed), 1)

    def test_aborted_http_calls_release_the_response(self):
        getresponse = http.client.HTTPConnection.getresponse
        responses = []
        def capture(connection):
            response = getresponse(connection)
            responses.append(response)
            return response
        cases = [
            ({"status": 429}, {}, 2, "model_http_status_429"),
            ({"content_type": "text/plain"}, {}, 2, "model_response_type"),
            ({}, {"max_response_bytes": 1}, 2, "model_response_size_limit"),
            ({"delay": 0.2}, {}, 0.04, "model_deadline"),
        ]
        for index, (server_options, model_options, timeout, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                url, _ = self.server(response={"unused": "response body"}, **server_options)
                client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url,
                                                       **model_options), self.root / str(index))
                with patch.object(http.client.HTTPConnection, "getresponse", capture):
                    with self.assertRaisesRegex(ModelError, "^" + reason + "$"):
                        client.complete([], TOOLS, timeout)
                self.assertTrue(responses[-1].isclosed())

    def test_deadline_includes_slow_http_response_body(self):
        url, _ = self.server(response={}, delay=0.2)
        client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url), self.root)
        with self.assertRaises(ModelError) as caught:
            client.complete([], TOOLS, 0.04)
        self.assertTrue(caught.exception.timeout)

    def test_truncated_completion_cannot_become_finish(self):
        url, _ = self.server(response={"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "done"}}]})
        client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url), self.root)
        with self.assertRaisesRegex(ModelError, "model_completion_length") as caught:
            client.complete([], TOOLS, 2)
        self.assertTrue(caught.exception.invalid_output)
        self.assertEqual(json.loads(Path(caught.exception.evidence_path).read_text())["choices"][0]["finish_reason"], "length")

    def test_generated_invalid_tool_arguments_are_scored_protocol_output_with_evidence(self):
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "f", "type": "function", "function": {"name": "finish", "arguments": '{"summary":'}}]}}]}
        url, observed = self.server(response=response)
        with patch.dict(os.environ, {"OPBENCH_TEST_MODEL_KEY": "DO_NOT_SAVE_CREDENTIAL"}):
            client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url,
                                           api_key_env="OPBENCH_TEST_MODEL_KEY"), self.root)
        with self.assertRaisesRegex(ModelError, "model_invalid_tool_arguments") as caught:
            client.complete([], TOOLS, 2)
        self.assertTrue(caught.exception.invalid_output)
        evidence = json.loads(Path(caught.exception.evidence_path).read_text())
        self.assertEqual(evidence["choices"], response["choices"])
        self.assertEqual(observed[0]["authorization"], "Bearer DO_NOT_SAVE_CREDENTIAL")
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn("DO_NOT_SAVE_CREDENTIAL", path.read_text())

    def test_damaged_response_envelope_is_service_error_not_generated_failure(self):
        url, _ = self.server(response={"choices": []})
        client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url), self.root)
        with self.assertRaisesRegex(ModelError, "model_invalid_response_envelope") as caught:
            client.complete([], TOOLS, 2)
        self.assertFalse(caught.exception.invalid_output)

    def test_refusal_and_content_filter_are_model_outputs(self):
        for index, (reason, message) in enumerate([
                ("stop", {"role": "assistant", "content": None, "refusal": "I cannot help"}),
                ("content_filter", {"role": "assistant", "content": ""})]):
            with self.subTest(reason=reason):
                url, _ = self.server(response={"choices": [{"finish_reason": reason, "message": message}]})
                client = ChatCompletionsClient(ModelSpec("test", "chat_completions", "x", base_url=url), self.root / str(index))
                with self.assertRaises(ModelError) as caught:
                    client.complete([], TOOLS, 2)
                self.assertTrue(caught.exception.invalid_output)
                self.assertEqual(json.loads(Path(caught.exception.evidence_path).read_text())["choices"][0]["message"], message)

    def codex_client(self, body):
        home = self.root / "auth-home"
        home.mkdir()
        catalog = {"models": [{"slug": "gpt-5.5", "shell_type": "unified_exec", "apply_patch_tool_type": "freeform"}]}
        (home / "models_cache.json").write_text(json.dumps(catalog))
        (home / "auth.json").write_text('{"secret":"DO_NOT_COPY"}')
        script = self.root / "fake_cli.py"
        script.write_text("import json,sys,time,pathlib\n" + body)
        with patch.dict(os.environ, {"CODEX_HOME": str(home)}):
            client = CodexCLIClient(ModelSpec("test", "codex_cli", "gpt-5.5"), self.root / "requests")
        original = client._command
        def command(scratch):
            client.observed_command = original(scratch)
            return [sys.executable, str(script), str(scratch)]
        client._command = command
        return client, home, catalog

    def test_codex_only_decision_uses_scratch_and_does_not_copy_auth(self):
        client, home, original = self.codex_client('''scratch=pathlib.Path(sys.argv[1])
request=json.load(sys.stdin)
assert request["conversation"] == [{"role":"user","content":"public task"}]
assert request["available_opbench_tools"][0]["function"]["name"] == "finish"
json.dump({"tool_calls":[{"id":"f","name":"finish","arguments":{"summary":"done","optional":None}}],"content":""},open(scratch/"decision.json","w"))
print(json.dumps({"type":"thread.started","thread_id":"x"}))
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"decision"}}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}))
''')
        result = client.complete([{"role": "user", "content": "public task"}], TOOLS, 2)
        self.assertEqual(result["tool_calls"][0]["arguments"], {"summary": "done"})
        self.assertEqual(result["usage"]["total_tokens"], 14)
        self.assertEqual(result["backend_metadata"]["mode"], "inference_smoke")
        self.assertEqual(json.loads((home / "models_cache.json").read_text()), original)
        self.assertEqual(client.model_catalog["shell_type"], "disabled")
        self.assertIsNone(client.model_catalog["apply_patch_tool_type"])
        command = client.observed_command
        self.assertIn("read-only", command)
        self.assertIn("features.shell_tool=false", command)
        self.assertIn("features.hooks=false", command)
        self.assertIn('model_provider="opbench_openai_https"', command)
        self.assertIn("model_providers.opbench_openai_https.supports_websockets=false", command)
        self.assertIn("model_providers.opbench_openai_https.requires_openai_auth=true", command)
        self.assertIn("model_providers.opbench_openai_https.stream_max_retries=0", command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
        self.assertEqual(result["backend_metadata"]["transport"], "https")
        self.assertNotIn(str(home / "auth.json"), " ".join(command))
        for path in (self.root / "requests").rglob("*"):
            if path.is_file():
                self.assertNotIn("DO_NOT_COPY", path.read_text())

    def test_codex_native_tool_event_kills_and_rejects_immediately(self):
        marker = self.root / "should-not-exist"
        client, _, _ = self.codex_client('''print(json.dumps({"type":"item.started","item":{"type":"command_execution","command":"bad"}}),flush=True)
time.sleep(0.5)
pathlib.Path(''' + repr(str(marker)) + ''').write_text("executed")
''')
        with self.assertRaisesRegex(ModelError, "codex_native_tool_attempt") as caught:
            client.complete([], TOOLS, 3)
        self.assertFalse(caught.exception.invalid_output)
        self.assertFalse(marker.exists())
        log = client.output_dir / "request-0001/events.jsonl"
        self.assertIn("command_execution", log.read_text())

    def test_codex_invalid_generated_decision_is_retained_and_marked_protocol_output(self):
        client, _, _ = self.codex_client('''scratch=pathlib.Path(sys.argv[1])
(scratch/"decision.json").write_text("not valid decision JSON")
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}))
''')
        with self.assertRaisesRegex(ModelError, "codex_invalid_decision") as caught:
            client.complete([], TOOLS, 2)
        self.assertTrue(caught.exception.invalid_output)
        self.assertEqual(Path(caught.exception.evidence_path).read_text(), "not valid decision JSON")

    def test_codex_decision_without_completed_turn_is_rejected(self):
        client, _, _ = self.codex_client('''scratch=pathlib.Path(sys.argv[1])
(scratch/"decision.json").write_text('{"tool_calls":[],"content":"done"}')
print(json.dumps({"type":"thread.started","thread_id":"x"}))
''')
        with self.assertRaisesRegex(ModelError, "codex_missing_completed_turn"):
            client.complete([], TOOLS, 2)

    def test_codex_wrong_model_does_not_fallback(self):
        home = self.root / "home"
        home.mkdir()
        (home / "models_cache.json").write_text('{"models":[{"slug":"other"}]}')
        with patch.dict(os.environ, {"CODEX_HOME": str(home)}):
            with self.assertRaisesRegex(ModelError, "selected_model_missing"):
                CodexCLIClient(ModelSpec("test", "codex_cli", "gpt-5.5"), self.root / "out")

    def test_schema_keeps_tool_argument_names_and_models_optional_as_nullable(self):
        schema = _decision_schema(TOOLS)
        call = schema["properties"]["tool_calls"]["items"]["anyOf"][0]
        self.assertEqual(call["properties"]["name"]["enum"], ["finish"])
        arguments = call["properties"]["arguments"]
        self.assertEqual(arguments["required"], ["summary", "optional"])
        self.assertEqual(arguments["properties"]["optional"]["anyOf"][1], {"type": "null"})


if __name__ == "__main__":
    unittest.main()
