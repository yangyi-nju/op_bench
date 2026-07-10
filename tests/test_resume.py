from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.resume import (
    BASELINE_CACHE_DIRNAME,
    BaselineCache,
    ResultsStore,
    RunState,
    RESULTS_FILE,
)


class RunStateTests(unittest.TestCase):
    def test_signature_stable_with_task_order(self) -> None:
        s1 = RunState.build(["a", "b", "c"], ["codex"], 3)
        s2 = RunState.build(["c", "a", "b"], ["codex"], 3)
        self.assertEqual(s1.dataset_signature, s2.dataset_signature)

    def test_signature_changes_when_tasks_change(self) -> None:
        s1 = RunState.build(["a", "b"], ["codex"], 3)
        s2 = RunState.build(["a", "b", "c"], ["codex"], 3)
        self.assertNotEqual(s1.dataset_signature, s2.dataset_signature)

    def test_signature_changes_when_task_content_changes(self) -> None:
        s1 = RunState.build(["a"], ["codex"], 3, task_signatures=["sha256:one"])
        s2 = RunState.build(["a"], ["codex"], 3, task_signatures=["sha256:two"])
        self.assertNotEqual(s1.dataset_signature, s2.dataset_signature)

    def test_save_and_load_roundtrip(self) -> None:
        state = RunState.build(["a", "b"], ["codex"], 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            state.save(path)
            loaded = RunState.load(path)
            self.assertEqual(loaded, state)

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(RunState.load(Path(tmp) / "missing.json"))

    def test_load_corrupt_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            path.write_text("not-json{")
            self.assertIsNone(RunState.load(path))

    def test_compatible_when_matching(self) -> None:
        s1 = RunState.build(["a"], ["codex"], 3)
        s2 = RunState.build(["a"], ["codex"], 3)
        ok, _ = s1.is_compatible(s2)
        self.assertTrue(ok)

    def test_incompatible_agent_repeat(self) -> None:
        s1 = RunState.build(["a"], ["codex"], 5)
        s2 = RunState.build(["a"], ["codex"], 3)
        ok, reason = s1.is_compatible(s2)
        self.assertFalse(ok)
        self.assertIn("agent_repeat", reason)

    def test_incompatible_agents(self) -> None:
        s1 = RunState.build(["a"], ["codex", "claude"], 3)
        s2 = RunState.build(["a"], ["codex"], 3)
        ok, reason = s1.is_compatible(s2)
        self.assertFalse(ok)
        self.assertIn("agent list", reason)

    def test_incompatible_dataset(self) -> None:
        s1 = RunState.build(["a", "b"], ["codex"], 3)
        s2 = RunState.build(["c", "d"], ["codex"], 3)
        ok, reason = s1.is_compatible(s2)
        self.assertFalse(ok)
        self.assertIn("dataset", reason)


class ResultsStoreTests(unittest.TestCase):
    def test_empty_store_reports_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            self.assertEqual(store.completed_agent_keys(), set())
            self.assertEqual(store.completed_baseline_task_ids(), {})

    def test_append_and_readback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_result({"task_id": "t1", "agent": "codex", "attempt": 1, "status": "resolved"})
            store.append_result({"task_id": "t1", "agent": "codex", "attempt": 2, "status": "resolved"})
            store.append_baseline({"task_id": "t1", "agent": "baseline", "status": "baseline_reproduced"})
            keys = store.completed_agent_keys()
            self.assertEqual(keys, {("t1", "codex", 1), ("t1", "codex", 2)})
            baselines = store.completed_baseline_task_ids()
            self.assertEqual(list(baselines.keys()), ["t1"])
            self.assertEqual(baselines["t1"]["status"], "baseline_reproduced")

    def test_tolerates_malformed_trailing_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_result({"task_id": "t1", "agent": "codex", "attempt": 1, "status": "resolved"})
            # simulate a crash mid-write: append a partial line
            with (Path(tmp) / RESULTS_FILE).open("a", encoding="utf-8") as h:
                h.write('{"task_id": "t2", "agent": "codex", "attempt": 2, "sta')
            keys = store.completed_agent_keys()
            self.assertEqual(keys, {("t1", "codex", 1)})

    def test_records_without_attempt_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            # No 'attempt' field — shouldn't be counted as a resumable agent record.
            store.append_result({"task_id": "t1", "agent": "codex", "status": "resolved"})
            self.assertEqual(store.completed_agent_keys(), set())

    def test_transient_attempt_is_retried_and_latest_result_is_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_result({
                "task_id": "t1", "agent": "codex", "attempt": 1,
                "status": "environment_unavailable",
            })
            self.assertEqual(store.completed_agent_keys(), set())

            store.append_result({
                "task_id": "t1", "agent": "codex", "attempt": 1,
                "status": "resolved",
            })
            self.assertEqual(store.completed_agent_keys(), {("t1", "codex", 1)})
            results = store.load_all_results()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "resolved")

    def test_timeout_and_runner_error_are_terminal_attempt_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_result({
                "task_id": "t1", "agent": "codex", "attempt": 1,
                "status": "timeout",
            })
            store.append_result({
                "task_id": "t1", "agent": "codex", "attempt": 2,
                "status": "runner_error",
            })

            self.assertEqual(
                store.completed_agent_keys(),
                {("t1", "codex", 1), ("t1", "codex", 2)},
            )

    def test_transient_baseline_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_baseline({
                "task_id": "t1", "agent": "baseline",
                "status": "environment_unavailable",
            })
            self.assertEqual(store.completed_baseline_task_ids(), {})

            store.append_baseline({
                "task_id": "t1", "agent": "baseline",
                "status": "baseline_reproduced",
            })
            baselines = store.completed_baseline_task_ids()
            self.assertEqual(baselines["t1"]["status"], "baseline_reproduced")
            self.assertEqual(len(store.load_all_baselines()), 1)

    def test_load_all_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultsStore(Path(tmp))
            store.append_result({"task_id": "t1", "agent": "codex", "attempt": 1, "status": "resolved"})
            store.append_result({"task_id": "t2", "agent": "codex", "attempt": 1, "status": "fail_to_pass_failed"})
            records = store.load_all_results()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["task_id"], "t1")
            self.assertEqual(records[1]["task_id"], "t2")


class BaselineCacheTests(unittest.TestCase):
    def test_cache_disabled_when_dir_is_none(self) -> None:
        cache = BaselineCache(None)
        # key_for returns None
        with tempfile.TemporaryDirectory() as tmp:
            hidden = Path(tmp) / "hidden.patch"
            hidden.write_text("diff --git a/x b/x\n")
            key = cache.key_for("t1", "hash1", hidden)
            self.assertIsNone(key)
            # get / put no-ops
            self.assertIsNone(cache.get(None))
            cache.put(None, {"x": 1})  # doesn't raise

    def test_roundtrip_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / BASELINE_CACHE_DIRNAME
            hidden = root / "hidden.patch"
            hidden.write_text("diff --git a/x b/x\nline\n")
            cache = BaselineCache(cache_dir)
            key = cache.key_for("t1", "snap-abc", hidden)
            self.assertIsNotNone(key)
            cache.put(key, {"status": "baseline_reproduced", "agent": "baseline"})
            hit = cache.get(key)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["status"], "baseline_reproduced")

    def test_key_changes_when_patch_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / BASELINE_CACHE_DIRNAME
            hidden = root / "hidden.patch"
            cache = BaselineCache(cache_dir)
            hidden.write_text("v1\n")
            k1 = cache.key_for("t1", "snap", hidden)
            hidden.write_text("v2\n")
            k2 = cache.key_for("t1", "snap", hidden)
            self.assertIsNotNone(k1)
            self.assertIsNotNone(k2)
            self.assertNotEqual(k1, k2)

    def test_key_changes_when_snapshot_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden.patch"
            hidden.write_text("v1\n")
            cache = BaselineCache(root / BASELINE_CACHE_DIRNAME)
            k1 = cache.key_for("t1", "snap-a", hidden)
            k2 = cache.key_for("t1", "snap-b", hidden)
            self.assertNotEqual(k1, k2)

    def test_key_stable_when_only_task_id_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden.patch"
            hidden.write_text("same content\n")
            cache = BaselineCache(root / BASELINE_CACHE_DIRNAME)
            k1 = cache.key_for("t1", "snap", hidden)
            k2 = cache.key_for("t1", "snap", hidden)
            self.assertEqual(k1, k2)

    def test_missing_patch_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = BaselineCache(root / BASELINE_CACHE_DIRNAME)
            # nonexistent path — should still produce a stable key, not raise
            k = cache.key_for("t1", "snap", root / "does_not_exist.patch")
            self.assertIsNotNone(k)

    def test_get_returns_none_on_corrupt_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / BASELINE_CACHE_DIRNAME
            cache = BaselineCache(cache_dir)
            hidden = Path(tmp) / "h.patch"
            hidden.write_text("x")
            key = cache.key_for("t1", "s", hidden)
            (cache_dir / f"{key}.json").write_text("not-json{")
            self.assertIsNone(cache.get(key))


if __name__ == "__main__":
    unittest.main()
