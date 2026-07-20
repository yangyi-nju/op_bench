from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.integrity import load_run_manifest_artifact
from op_bench.runtime.legacy import agent_spec_for_v1_adapter
from op_bench.runtime.orchestrator import V06RunRequest
from tests.runtime_orchestrator_fixture import (
    StepClock,
    build_orchestrator_fixture,
    orchestrator_for,
)
from tests.test_runtime_contracts import identity
from tests.test_runtime_manifest import manifest
from tests.test_runtime_orchestrator import McpScenarioAdapter

try:
    from op_bench.runtime.experiment_report import (
        FORMAL_MCP_EXPERIMENT_CONTRACT,
        McpExperimentCohortContract,
        McpExperimentContract,
        build_mcp_experiment_report,
    )
    from scripts.summarize_mcp_experiment import main as summarize_main
except ImportError:
    build_mcp_experiment_report = None
    McpExperimentCohortContract = None
    McpExperimentContract = None
    FORMAL_MCP_EXPERIMENT_CONTRACT = None
    summarize_main = None


MODEL_ID = "gpt-5.6-sol"
CLI_VERSION = "codex-cli 0.145.0-alpha.18"


class CyclingMcpAdapter:
    def __init__(self) -> None:
        self.run_count = 0

    def run(self, context):
        scenarios = ("resolved", "f2p_failed", "timeout", "no_patch")
        scenario = scenarios[self.run_count % len(scenarios)]
        self.run_count += 1
        return McpScenarioAdapter(scenario).run(context)


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class McpExperimentReportTests(unittest.TestCase):
    def test_formal_contract_freezes_exact_17_task_51_attempt_partition(self) -> None:
        contract = FORMAL_MCP_EXPERIMENT_CONTRACT
        self.assertIsNotNone(contract)
        self.assertEqual(contract.dataset_identifier, "pytorch_v0.5")
        self.assertEqual(contract.platform_version, "opbench-v0.6.0")
        self.assertEqual(contract.expected_attempt_count, 51)
        self.assertEqual(
            sorted(len(item.expected_attempts) for item in contract.cohorts),
            [3, 6, 6, 36],
        )
        self.assertEqual(
            len({task_id for item in contract.cohorts for task_id in item.task_ids}),
            17,
        )

    @classmethod
    def setUpClass(cls) -> None:
        if build_mcp_experiment_report is None:
            cls.temporary = None
            cls.roots = ()
            return
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        mcp_agent = agent_spec_for_v1_adapter(
            "codex_mcp_canonical",
            model_id=MODEL_ID,
            codex_cli_version=CLI_VERSION,
        )
        base = build_orchestrator_fixture(root / "base", selected_agent=mcp_agent)
        roots = []
        for cohort_index, repeat_count in enumerate((36, 3, 6, 6), start=1):
            task = replace(
                base.manifest.tasks[0],
                task=identity(
                    "task",
                    f"experiment-report-task-{cohort_index}",
                    "sha256:" + str(cohort_index) * 64,
                ),
                statement_body=f"Synthetic report cohort {cohort_index}",
            )
            frozen_manifest = manifest(
                tasks=(task,),
                agents=(mcp_agent,),
                capability=base.manifest.capability_policy,
                budget=base.manifest.budget_policy,
                evaluation=base.manifest.evaluation,
                scoring=base.manifest.scoring,
                repeat_count=repeat_count,
            )
            output = root / f"cohort-{cohort_index}"
            fixture = replace(
                base,
                manifest=frozen_manifest,
                expected=frozen_manifest.expected_attempts[0],
                output_root=output,
            )
            adapter = CyclingMcpAdapter()
            request = V06RunRequest(
                manifest=frozen_manifest,
                selected_attempt_ids=tuple(
                    item.attempt_id for item in frozen_manifest.expected_attempts
                ),
                runtime_profile_registry=fixture.registry,
                runtime_profile_id=fixture.profile.profile_id,
                target_binding=fixture.target_binding,
                output_root=fixture.output_root,
                resume_policy="retry_infrastructure",
                adapter_id="codex_mcp_canonical",
                enable_external_canary=True,
                clock_ms=StepClock(),
            )
            result = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: __import__(
                    "op_bench.runtime.backends",
                    fromlist=["LocalProcessBackend"],
                ).LocalProcessBackend(),
                adapter=adapter,
            ).run(request)
            if result.integrity.status != "passed":
                raise AssertionError(result.integrity)
            roots.append(output)
        cls.roots = tuple(roots)
        manifests = tuple(load_run_manifest_artifact(item) for item in cls.roots)
        cls.contract = McpExperimentContract(
            dataset_identifier=manifests[0].dataset.identifier,
            dataset_digest=manifests[0].dataset.digest,
            platform_version=manifests[0].platform_version,
            cohorts=tuple(
                McpExperimentCohortContract(
                    profile_id=manifest.runtime_profiles[0].profile_id,
                    task_repeats=tuple(
                        (
                            task.task.identifier,
                            tuple(
                                item.repeat
                                for item in manifest.expected_attempts
                                if item.task.identifier == task.task.identifier
                            ),
                        )
                        for task in manifest.tasks
                    ),
                )
                for manifest in manifests
            ),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def builder(self, roots=None):
        self.assertIsNotNone(build_mcp_experiment_report)
        return build_mcp_experiment_report(
            self.roots if roots is None else roots,
            expected_adapter_id="codex_mcp_canonical",
            expected_model_id=MODEL_ID,
            expected_codex_cli_version=CLI_VERSION,
            experiment_contract=self.contract,
        )

    def test_builder_rejects_dataset_platform_profile_task_and_repeat_drift(self) -> None:
        mutations = (
            replace(self.contract, dataset_digest="sha256:" + "0" * 64),
            replace(self.contract, platform_version="opbench-v9.9.9"),
            replace(
                self.contract,
                cohorts=(
                    replace(self.contract.cohorts[0], profile_id="wrong-profile-v1"),
                    *self.contract.cohorts[1:],
                ),
            ),
            replace(
                self.contract,
                cohorts=(
                    replace(
                        self.contract.cohorts[0],
                        task_repeats=(("wrong-task", (1,)),),
                    ),
                    *self.contract.cohorts[1:],
                ),
            ),
            replace(
                self.contract,
                cohorts=(
                    replace(
                        self.contract.cohorts[0],
                        task_repeats=(
                            (
                                self.contract.cohorts[0].task_repeats[0][0],
                                (1,),
                            ),
                        ),
                    ),
                    *self.contract.cohorts[1:],
                ),
            ),
        )
        for contract in mutations:
            with self.subTest(contract=contract), self.assertRaises(Exception):
                build_mcp_experiment_report(
                    self.roots,
                    expected_adapter_id="codex_mcp_canonical",
                    expected_model_id=MODEL_ID,
                    expected_codex_cli_version=CLI_VERSION,
                    experiment_contract=contract,
                )

    def test_complete_four_cohort_report_has_all_51_attempts_and_metrics(self) -> None:
        index, summary = self.builder()

        self.assertEqual(index["report_type"], "mcp_experiment_index")
        self.assertEqual(len(index["attempts"]), 51)
        self.assertEqual(summary["totals"]["attempts"], 51)
        self.assertEqual(summary["totals"]["trace_complete"], 51)
        self.assertEqual(
            sorted(item["selected_attempts"] for item in index["cohorts"]),
            [3, 6, 6, 36],
        )
        self.assertEqual(
            set(summary["evaluation_outcomes"]),
            {"f2p_failed", "no_patch", "resolved"},
        )
        self.assertEqual(
            set(summary["agent_terminals"]),
            {"finished", "timeout"},
        )
        self.assertEqual(summary["action_coverage"]["finish"]["denominator"], 51)
        self.assertEqual(summary["attribution"]["agent"], 51)

    def test_builder_and_cli_are_byte_deterministic(self) -> None:
        first = self.builder()
        second = self.builder()
        self.assertEqual(canonical_json(list(first)), canonical_json(list(second)))
        self.assertIsNotNone(summarize_main)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = (root / "first", root / "second")
            for output in outputs:
                arguments = []
                for run_root in self.roots:
                    arguments.extend(("--run-root", str(run_root)))
                arguments.extend(
                    (
                        "--output-dir",
                        str(output),
                        "--expected-model",
                        MODEL_ID,
                        "--expected-cli-version",
                        CLI_VERSION,
                    )
                )
                self.assertEqual(
                    summarize_main(arguments, experiment_contract=self.contract),
                    0,
                )
            self.assertEqual(file_hashes(outputs[0]), file_hashes(outputs[1]))
            self.assertEqual(
                summarize_main(arguments, experiment_contract=self.contract),
                0,
            )
            summary_path = outputs[1] / "experiment_summary.json"
            summary_path.write_bytes(summary_path.read_bytes() + b"\n")
            self.assertEqual(
                summarize_main(arguments, experiment_contract=self.contract),
                2,
            )

    def test_rejects_duplicate_cohort_identity_and_wrong_exact_identity(self) -> None:
        with self.assertRaisesRegex(Exception, "duplicate|four|count"):
            self.builder((self.roots[0], self.roots[1], self.roots[2], self.roots[2]))
        with self.assertRaisesRegex(Exception, "model"):
            build_mcp_experiment_report(
                self.roots,
                expected_adapter_id="codex_mcp_canonical",
                expected_model_id="gpt-5.6-terra",
                expected_codex_cli_version=CLI_VERSION,
                experiment_contract=self.contract,
            )
        with self.assertRaisesRegex(Exception, "CLI"):
            build_mcp_experiment_report(
                self.roots,
                expected_adapter_id="codex_mcp_canonical",
                expected_model_id=MODEL_ID,
                expected_codex_cli_version="codex-cli 0.146.0",
                experiment_contract=self.contract,
            )
        with self.assertRaisesRegex(Exception, "adapter"):
            build_mcp_experiment_report(
                self.roots,
                expected_adapter_id="codex_canonical",
                expected_model_id=MODEL_ID,
                expected_codex_cli_version=CLI_VERSION,
                experiment_contract=self.contract,
            )

    def test_rejects_noncanonical_public_json_and_missing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "cohort"
            shutil.copytree(self.roots[0], mutated)
            results = mutated / "results.jsonl"
            results.write_bytes(results.read_bytes().replace(b"\n", b" \n", 1))
            roots = (mutated, *self.roots[1:])
            with self.assertRaisesRegex(Exception, "Integrity|canonical|failed"):
                self.builder(roots)

        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "cohort"
            shutil.copytree(self.roots[0], mutated)
            row = json.loads((mutated / "results.jsonl").read_text().splitlines()[0])
            trace = (
                mutated
                / "attempts"
                / row["attempt_id"]
                / "retries"
                / f"retry-{row['retry_index']:04d}"
                / "adapter_trace.json"
            )
            trace.unlink()
            roots = (mutated, *self.roots[1:])
            with self.assertRaisesRegex(Exception, "Integrity|trace|failed"):
                self.builder(roots)

    def test_rejects_failed_runtime_resource_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "cohort"
            shutil.copytree(self.roots[1], mutated)
            row = json.loads((mutated / "results.jsonl").read_text().splitlines()[0])
            cleanup = (
                mutated
                / "attempts"
                / row["attempt_id"]
                / "retries"
                / f"retry-{row['retry_index']:04d}"
                / "runtime_cleanup.json"
            )
            payload = json.loads(cleanup.read_text(encoding="utf-8"))
            payload["entries"][0]["status"] = "cleanup_failed"
            payload["entries"][0]["error_code"] = "fixture_cleanup_failed"
            payload["all_released"] = False
            cleanup.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            roots = (self.roots[0], mutated, self.roots[2], self.roots[3])
            with self.assertRaisesRegex(Exception, "Integrity|resource|failed"):
                self.builder(roots)


if __name__ == "__main__":
    unittest.main()
