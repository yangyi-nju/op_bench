from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from op_bench.matched_runtime.contracts import (
    BuildIdentity,
    CompatibilityCheck,
    CompatibilityEvidence,
    CompatibilityFailure,
    RuntimeIdentity,
    SourceIdentity,
    compatibility_content_hash,
)
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
BASE_COMMIT = "1" * 40


def passed_checks() -> tuple[CompatibilityCheck, ...]:
    return tuple(
        CompatibilityCheck(
            name=name,
            command_digest=SHA_C,
            exit_code=0,
            status="passed",
            summary=f"{name} passed",
        )
        for name in (
            "source_identity",
            "runtime_identity",
            "target_module_provenance",
            "target_import",
            "selector_collection",
            "minimal_operation",
        )
    )


def source_identity() -> SourceIdentity:
    return SourceIdentity(
        source_id="pytorch-ff89ebc-cuda-overlay",
        commit=BASE_COMMIT,
        snapshot_digest=SHA_A,
        snapshot_digest_kind="git_archive_sha256",
        target_module_path="torch/_refs/__init__.py",
        target_module_sha256=SHA_B,
        runtime_path_suffix="torch/_refs/__init__.py",
    )


def runtime_identity(*, target_hash: str = SHA_B) -> RuntimeIdentity:
    return RuntimeIdentity(
        environment_id="pytorch-matched-ff89ebc-torch2.4.0-py311-cu124",
        artifact_kind="official_wheel",
        artifact_id="torch-2.4.0+cu124-cp311-linux_x86_64",
        artifact_digest=SHA_A,
        artifact_digest_kind="wheel_sha256",
        torch_version="2.4.0+cu124",
        python_implementation="CPython",
        python_abi="cpython-311-x86_64-linux-gnu",
        platform="linux/amd64",
        cuda_build="12.4",
        cuda_runtime="12.4",
        device_name="Tesla V100-SXM2-32GB",
        compute_capability="7.0",
        source_loading_mode="python_overlay",
        target_module_path_suffix="torch/_refs/__init__.py",
        target_module_sha256=target_hash,
    )


def unavailable_runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        environment_id="pytorch-matched-ff89ebc-torch2.4.0-py311-cu124",
        artifact_kind="official_wheel",
        artifact_id="torch-2.4.0+cu124-cp311-linux_x86_64",
        artifact_digest=None,
        artifact_digest_kind=None,
        torch_version=None,
        python_implementation=None,
        python_abi=None,
        platform=None,
        cuda_build=None,
        cuda_runtime=None,
        device_name=None,
        compute_capability=None,
        source_loading_mode="python_overlay",
        target_module_path_suffix=None,
        target_module_sha256=None,
    )


def wheel_build_identity() -> BuildIdentity:
    return BuildIdentity(
        flags=(),
        gpu_arches=(),
        ccache_key=None,
        artifact_digest=None,
        toolchain=(),
    )


def compatible_evidence() -> CompatibilityEvidence:
    return CompatibilityEvidence(
        task_id="pytorch__129154__exp_decomp_numerics",
        strategy="matched_wheel",
        status="compatible",
        source=source_identity(),
        runtime=runtime_identity(),
        build=wheel_build_identity(),
        checks=passed_checks(),
        failure=None,
        created_at="2026-07-26T12:00:00Z",
    )


def incompatible_evidence() -> CompatibilityEvidence:
    checks = list(passed_checks())
    checks[3] = replace(
        checks[3],
        exit_code=1,
        status="failed",
        summary="target import failed",
    )
    return CompatibilityEvidence(
        task_id="pytorch__129154__exp_decomp_numerics",
        strategy="matched_wheel",
        status="incompatible",
        source=source_identity(),
        runtime=runtime_identity(),
        build=wheel_build_identity(),
        checks=tuple(checks),
        failure=CompatibilityFailure(
            code="target_import_failed",
            check="target_import",
            summary="Target import raised an API compatibility exception.",
        ),
        created_at="2026-07-26T12:00:00Z",
    )


def source_build_evidence() -> CompatibilityEvidence:
    return CompatibilityEvidence(
        task_id="pytorch__129154__exp_decomp_numerics",
        strategy="source_built_wheel",
        status="compatible",
        source=source_identity(),
        runtime=replace(
            runtime_identity(),
            artifact_kind="source_built_wheel",
            artifact_id="pytorch-ff89ebc-py311-cu124-sm70-v1",
            artifact_digest_kind="build_artifact_sha256",
        ),
        build=BuildIdentity(
            flags=(
                "BUILD_TEST=0",
                "TORCH_CUDA_ARCH_LIST=7.0",
                "USE_CUDA=1",
            ),
            gpu_arches=("7.0",),
            ccache_key="pytorch-ff89ebc-py311-cu124-sm70-v1",
            artifact_digest=SHA_A,
            toolchain=("gcc=11.4.0", "nvcc=12.4"),
        ),
        checks=passed_checks(),
        failure=None,
        created_at="2026-07-26T12:00:00Z",
    )


class CompatibilityContractTests(unittest.TestCase):
    def test_compatible_evidence_round_trips_exactly(self) -> None:
        selected = compatible_evidence()

        encoded = selected.to_dict()

        self.assertEqual(CompatibilityEvidence.from_dict(encoded), selected)
        self.assertEqual(CompatibilityEvidence.from_dict(encoded).to_dict(), encoded)

    def test_content_hash_detects_mutation(self) -> None:
        payload = compatible_evidence().to_dict()
        modified = replace(
            compatible_evidence(),
            runtime=replace(runtime_identity(), torch_version="9.9.9"),
        )
        payload["runtime"] = modified.runtime.to_dict()
        payload["evidence_id"] = modified.evidence_id

        with self.assertRaisesRegex(ContractError, "content_hash"):
            CompatibilityEvidence.from_dict(payload)

    def test_evidence_id_is_derived_from_unhashed_payload(self) -> None:
        payload = compatible_evidence().to_dict()
        payload["evidence_id"] = "compatibility:v1:" + "f" * 64
        payload["content_hash"] = compatibility_content_hash(payload)

        with self.assertRaisesRegex(ContractError, "evidence_id"):
            CompatibilityEvidence.from_dict(payload)

    def test_compatible_requires_snapshot_target_module_identity(self) -> None:
        with self.assertRaisesRegex(ContractError, "target module"):
            replace(
                compatible_evidence(),
                runtime=runtime_identity(target_hash=SHA_C),
            )

    def test_incompatible_requires_failure_matching_a_failed_check(self) -> None:
        with self.assertRaisesRegex(ContractError, "failure"):
            replace(incompatible_evidence(), failure=None)

        with self.assertRaisesRegex(ContractError, "failed check"):
            replace(
                incompatible_evidence(),
                failure=CompatibilityFailure(
                    code="target_import_failed",
                    check="minimal_operation",
                    summary="Wrong check reference.",
                ),
            )

    def test_source_build_requires_content_addressed_build_details(self) -> None:
        for field, value in (
            ("ccache_key", None),
            ("artifact_digest", None),
            ("flags", ()),
            ("toolchain", ()),
        ):
            with self.subTest(field=field):
                selected = source_build_evidence()
                with self.assertRaisesRegex(ContractError, "build"):
                    replace(
                        selected,
                        build=replace(selected.build, **{field: value}),
                    )

    def test_matched_wheel_rejects_source_build_only_fields(self) -> None:
        with self.assertRaisesRegex(ContractError, "matched_wheel"):
            replace(
                compatible_evidence(),
                build=BuildIdentity(
                    flags=("USE_CUDA=1",),
                    gpu_arches=("7.0",),
                    ccache_key="unexpected",
                    artifact_digest=SHA_A,
                    toolchain=("gcc=11",),
                ),
            )

    def test_cuda_runtime_requires_complete_device_identity(self) -> None:
        for field in ("cuda_build", "cuda_runtime", "device_name", "compute_capability"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ContractError, "CUDA"):
                    replace(
                        compatible_evidence(),
                        runtime=replace(runtime_identity(), **{field: None}),
                    )

    def test_checks_are_complete_unique_ordered_and_coherent(self) -> None:
        selected = compatible_evidence()
        cases = (
            selected.checks[:-1],
            selected.checks + (selected.checks[-1],),
            tuple(reversed(selected.checks)),
        )
        for checks in cases:
            with self.subTest(checks=checks):
                with self.assertRaises(ContractError):
                    replace(selected, checks=checks)

        with self.assertRaisesRegex(ContractError, "exit_code"):
            replace(
                selected.checks[0],
                exit_code=1,
                status="passed",
            )

    def test_unavailable_requires_unavailable_check_and_failure(self) -> None:
        selected = compatible_evidence()
        checks = list(selected.checks)
        checks[1] = replace(
            checks[1],
            exit_code=None,
            status="unavailable",
            summary="runtime could not be reached",
        )
        unavailable = replace(
            selected,
            status="unavailable",
            checks=tuple(checks),
            failure=CompatibilityFailure(
                code="runtime_unavailable",
                check="runtime_identity",
                summary="The declared runtime could not be reached.",
            ),
        )

        self.assertEqual(
            CompatibilityEvidence.from_dict(unavailable.to_dict()),
            unavailable,
        )

    def test_unavailable_can_record_an_artifact_that_was_not_observed(self) -> None:
        selected = compatible_evidence()
        checks = list(selected.checks)
        checks[1] = replace(
            checks[1],
            exit_code=None,
            status="unavailable",
            summary="runtime artifact was unavailable",
        )
        unavailable = replace(
            selected,
            status="unavailable",
            runtime=unavailable_runtime_identity(),
            checks=tuple(checks),
            failure=CompatibilityFailure(
                code="artifact_not_found",
                check="runtime_identity",
                summary="The declared wheel or image was not available.",
            ),
        )

        self.assertEqual(
            CompatibilityEvidence.from_dict(unavailable.to_dict()),
            unavailable,
        )

    def test_paths_timestamps_hashes_and_summaries_are_canonical(self) -> None:
        mutations = (
            ("source", {"target_module_path": "../torch/_refs/__init__.py"}),
            ("source", {"runtime_path_suffix": "/torch/_refs/__init__.py"}),
            ("created_at", "2026-07-26T12:00:00+00:00"),
            ("check_summary", "x" * 501),
            ("artifact_digest", "sha256:short"),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                selected = compatible_evidence()
                with self.assertRaises(ContractError):
                    if name == "source":
                        replace(
                            selected,
                            source=replace(selected.source, **mutation),
                        )
                    elif name == "created_at":
                        replace(selected, created_at=mutation)
                    elif name == "check_summary":
                        replace(
                            selected,
                            checks=(
                                replace(selected.checks[0], summary=mutation),
                                *selected.checks[1:],
                            ),
                        )
                    else:
                        replace(
                            selected,
                            runtime=replace(
                                selected.runtime,
                                artifact_digest=mutation,
                            ),
                        )

    def test_contract_rejects_unknown_nested_fields(self) -> None:
        payload = compatible_evidence().to_dict()
        payload["runtime"]["host"] = "private.example"
        payload["content_hash"] = compatibility_content_hash(payload)

        with self.assertRaisesRegex(ContractError, "unknown fields"):
            CompatibilityEvidence.from_dict(payload)

    def test_json_schema_tracks_contract_enums(self) -> None:
        schema = json.loads(
            (
                ROOT / "schemas" / "matched_runtime_compatibility.schema.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            schema["properties"]["strategy"]["enum"],
            ["matched_wheel", "source_built_wheel", "full_source_build"],
        )
        self.assertEqual(
            schema["properties"]["status"]["enum"],
            ["compatible", "incompatible", "unavailable"],
        )
        self.assertEqual(
            schema["$defs"]["check"]["properties"]["status"]["enum"],
            ["passed", "failed", "unavailable"],
        )

    def test_schema_required_fields_match_wire_contract(self) -> None:
        schema = json.loads(
            (
                ROOT / "schemas" / "matched_runtime_compatibility.schema.json"
            ).read_text(encoding="utf-8")
        )

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            set(CompatibilityEvidence.wire_fields()),
        )
        self.assertEqual(
            set(schema["properties"]),
            set(CompatibilityEvidence.wire_fields()),
        )


if __name__ == "__main__":
    unittest.main()
