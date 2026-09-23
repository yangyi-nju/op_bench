"""Check recorded execution evidence and score arithmetic, not byte identity."""
from __future__ import annotations

import math
from pathlib import Path

from op_bench.evaluation.judgment import RESULT_STATUSES, UNKNOWN_STATUSES, unittest_witness_error, variant_status
from op_bench.io import read_json
from op_bench.evaluation import numeric


GROUPS = {"fail_to_pass", "pass_to_pass"}


def evaluation_outcome_issue(evaluation: dict, identity: dict | None) -> str | None:
    """Explain why verified evidence cannot yield a known logical task outcome.

    Evidence can be valid while a required environment failed or was never
    started. Reporting consumes this rule without implementing grading again.
    """
    if evaluation["status"] in UNKNOWN_STATUSES:
        return evaluation["status"]
    if identity is not None and evaluation.get("task_identity") != identity:
        return "evaluation_identity_mismatch"
    if not identity or identity.get("status") != "declared":
        return None
    expected = {variant["variant_id"] for variant in identity["required_variants"]}
    variants = evaluation.get("variants")
    if variants is None and expected == {"default"}:
        return None
    if not isinstance(variants, dict) or variants.keys() != expected:
        return "required_variant_evidence_missing"
    for result in variants.values():
        if (not isinstance(result, dict) or not isinstance(result.get("status"), str)
                or result["status"] not in RESULT_STATUSES):
            return "required_variant_evidence_invalid"
        if type(result.get("resolved")) is not bool or result["resolved"] != (result["status"] == "resolved"):
            return "required_variant_evidence_invalid"
        if result.get("task_id", identity["task_id"]) != identity["task_id"]:
            return "required_variant_identity_mismatch"
    if any(result["status"] in UNKNOWN_STATUSES for result in variants.values()):
        return "required_variant_evaluation_unavailable"
    if evaluation["status"] != variant_status(variants, list(expected)):
        return "required_variant_outcome_mismatch"
    return None


def _variant_evidence(root: Path, data: dict) -> list[str]:
    findings = []
    required, variants = data.get("required_variants"), data.get("variants")
    if (not isinstance(required, list) or not required or
            any(not isinstance(name, str) or not name for name in required) or len(set(required)) != len(required)
            or not isinstance(variants, dict) or not set(variants).issubset(required)):
        return ["invalid required variant declaration or references"]
    # Version 1 stored full inline results and a separate artifact map. Project
    # only its references here; independently saved children remain authoritative.
    if data.get("schema_version") == 1:
        artifacts = data.get("variant_artifacts")
        if not isinstance(artifacts, dict) or not set(artifacts).issubset(required):
            return ["invalid legacy variant artifact map"]
        variants = {name: {"evaluation_path": f"{artifacts[name]}/result.json"
                           if isinstance(artifacts.get(name), str) else None,
                           "status": child.get("status"), "resolved": child.get("resolved")}
                    if isinstance(child, dict) else child for name, child in variants.items()}
    identity = data.get("task_identity")
    if isinstance(identity, dict) and identity.get("status") == "declared":
        declared = identity.get("required_variants")
        if (not isinstance(declared, list) or any(not isinstance(item, dict) for item in declared)
                or [item.get("variant_id") for item in declared] != required):
            findings.append("required variants disagree with the logical task identity")
    try:
        submitted_patch = (root / "patch.diff").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"cannot read submitted patch: {exc}"]
    visited = set()
    for name, child in variants.items():
        try:
            if not isinstance(child, dict):
                raise ValueError("variant reference must be an object")
            if (not isinstance(child.get("status"), str) or child["status"] not in RESULT_STATUSES
                    or type(child.get("resolved")) is not bool
                    or child["resolved"] != (child["status"] == "resolved")):
                raise ValueError("invalid variant outcome summary")
            relative = child.get("evaluation_path")
            path = (root / relative).resolve() if isinstance(relative, str) else None
            directory = path.parent if path is not None else None
            if (directory is None or path.name != "result.json" or Path(relative).is_absolute() or directory == root
                    or not directory.is_relative_to(root) or directory in visited):
                raise ValueError("variant result must be result.json in a distinct child directory")
            visited.add(directory)
            checked = verify_evaluation(directory, expected_task_id=data["task_id"])
            if not checked["valid"]:
                findings.extend(f"{name}: {finding}" for finding in checked["findings"])
                continue
            saved = read_json(directory / "result.json")
            if "required_variants" in saved:
                raise ValueError("a variant must be a single-environment evaluation")
            if any(child.get(key) != saved.get(key) for key in ("status", "resolved")):
                findings.append(f"{name}: variant summary disagrees with saved outcome")
            if isinstance(identity, dict) and identity.get("status") == "declared":
                declaration = next((item for item in identity.get("required_variants", [])
                                    if isinstance(item, dict) and item.get("variant_id") == name), None)
                if declaration is not None:
                    environment = declaration.get("environment")
                    expected_identity = {**identity, "solver_environment": environment,
                                         "required_variants": [{"variant_id": "default", "environment": environment}]}
                    if saved.get("task_identity") != expected_identity:
                        findings.append(f"{name}: variant task identity differs from its declared grading environment or revision")
            if (directory / "patch.diff").read_text(encoding="utf-8") != submitted_patch:
                findings.append(f"{name}: variant did not use the logical task's frozen patch")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            findings.append(f"{name}: invalid variant evidence: {exc}")
    try:
        if data["status"] != variant_status(variants, required):
            findings.append("logical task status disagrees with required variant outcomes")
    except (KeyError, TypeError):
        findings.append("invalid required variant outcomes")
    return findings


def _nonnegative(value: object) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _artifact(root: Path, relative, limit: int) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("artifact is missing, outside the result directory, or exceeds its limit")
    return path


def _numeric_evidence(root: Path, data: dict, case: dict, index: int, findings: list) -> bool:
    try:
        oracle_path = _artifact(root, case.get("oracle_log"), numeric.MAX_DOCUMENT_BYTES)
        observation_path = _artifact(root, case.get("observation_log"), numeric.MAX_DOCUMENT_BYTES)
        oracle = numeric.validate_oracle(numeric.read_document(oracle_path))
        observation = observation_path.read_bytes()
        decision = numeric.process_judgment(oracle, observation, case)
        if case.get("numeric") != decision or case.get("reason") != decision["reason"]:
            findings.append(f"case {index}: numeric judgment disagrees with saved observation/oracle and process outcome")
        session = data.get("sessions", {}).get(case.get("session_role"))
        if not isinstance(session, dict) or session.get("grader_access") is not False or session.get("execution_stopped") is not True:
            findings.append(f"case {index}: numeric execution lacks stopped session without grader access")
        if session is not None and session.get("backend") == "docker" and session.get("observed_network_mode") != "none":
            findings.append(f"case {index}: numeric Docker session did not observe network none")
        return decision["passed"]
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        findings.append(f"case {index}: invalid numeric evidence: {exc}")
        return False


def verify_evaluation(path: str | Path, *, expected_task_id: str | None = None) -> dict:
    """Check saved evidence, optionally binding it to an Attempt's task.

    Numeric decisions are recomputed from saved observations and oracles;
    candidate code is never rerun and artifact contents are not authenticated.
    """
    root = Path(path).resolve()
    findings = []

    def outcome() -> dict:
        return {"valid": not findings, "findings": findings,
                "scope": "Artifact presence, recorded process outcomes, numeric observation comparison and score arithmetic; no candidate code is imported. Rerun evaluate to replay the patch; this check does not authenticate artifacts."}

    try:
        data = read_json(root / "result.json")
    except (OSError, ValueError) as exc:
        findings.append(f"cannot read result.json: {exc}")
        return outcome()
    if not isinstance(data, dict):
        findings.append("result must be an object")
        return outcome()
    variant_result = "variants" in data or "required_variants" in data
    if type(data.get("schema_version")) is not int or data["schema_version"] not in ({1, 2} if variant_result else {1}):
        findings.append("unsupported result schema_version")
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        findings.append("result task_id must be a nonempty string")
    if expected_task_id is not None and task_id != expected_task_id:
        findings.append("evaluation task_id does not match the Attempt")
    status = data.get("status")
    if not isinstance(status, str) or status not in RESULT_STATUSES:
        findings.append("unknown result status")
    if type(data.get("resolved")) is not bool or data["resolved"] != (status == "resolved"):
        findings.append("resolved flag and status disagree")
    if not _nonnegative(data.get("duration_sec")):
        findings.append("result duration_sec must be finite and nonnegative")
    if variant_result:
        findings.extend(_variant_evidence(root, data))
        return outcome()
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        findings.append("result has no required cases")
        return outcome()
    if any(not isinstance(case, dict) for case in cases):
        findings.append("test cases must be objects")
        return outcome()
    identifiers = set()
    for index, case in enumerate(cases):
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            findings.append(f"case {index}: id must be a nonempty string")
        elif identifier in identifiers:
            findings.append(f"duplicate test case ID: {identifier}")
        else:
            identifiers.add(identifier)
        group, case_status = case.get("group"), case.get("status")
        if not isinstance(group, str) or group not in GROUPS:
            findings.append(f"case {index}: invalid test group")
        if case_status not in ("passed", "failed", "not_run", "error"):
            findings.append(f"case {index}: invalid test status")
        kind, expected_tests = case.get("kind"), case.get("expected_tests")
        if kind not in ("command", "unittest", "numeric"):
            findings.append(f"case {index}: kind must be command, unittest or numeric")
        if kind == "numeric" and expected_tests != 1:
            findings.append(f"case {index}: numeric expected_tests must equal one protocol case")
        if type(expected_tests) is not int or expected_tests < 1:
            findings.append(f"case {index}: expected_tests must be a positive integer")
        if type(case.get("timed_out")) is not bool or type(case.get("output_limited")) is not bool:
            findings.append(f"case {index}: process limit flags must be booleans")
        if not _nonnegative(case.get("duration_sec")):
            findings.append(f"case {index}: duration_sec must be finite and nonnegative")
        if case_status == "not_run":
            if case.get("exit_code") is not None or case.get("timed_out") or case.get("output_limited"):
                findings.append(f"case {index}: not_run case has a recorded process outcome")
            if case.get("witness") is not None or case.get("witness_log") is not None:
                findings.append(f"case {index}: not_run case has unittest execution evidence")
        elif case_status == "error":
            if type(case.get("exit_code")) is not int:
                findings.append(f"case {index}: evidence collection error requires the recorded process exit_code")
            supported_error = (kind == "unittest" and case.get("reason") == "unittest_witness_collection_error") or (
                kind == "numeric" and case.get("reason") in {"numeric_execution_boundary_unavailable", "numeric_observation_collection_error"})
            if status != "evaluation_error" or not supported_error:
                findings.append(f"case {index}: unsupported or incorrectly classified execution evidence error")
            error = case.get("error")
            if not isinstance(error, dict) or not isinstance(error.get("message"), str) or not error["message"]:
                findings.append(f"case {index}: missing evidence collection error detail")
            if case.get("witness") is not None or case.get("witness_log") is not None:
                findings.append(f"case {index}: failed witness collection contains a witness")
        elif case_status in ("passed", "failed"):
            if type(case.get("exit_code")) is not int:
                findings.append(f"case {index}: executed case requires an integer exit_code")
            successful = case.get("exit_code") == 0 and not case.get("timed_out") and not case.get("output_limited")
            if kind == "unittest":
                witness = case.get("witness")
                if witness is not None and not isinstance(witness, dict):
                    findings.append(f"case {index}: unittest witness must be an object or null")
                witness_error = unittest_witness_error(witness, expected_tests)
                successful = successful and witness_error is None
                if witness_error is not None and case.get("reason") != witness_error:
                    findings.append(f"case {index}: unittest failure reason disagrees with execution evidence")
                witness_log = case.get("witness_log")
                if witness is None:
                    if witness_log is not None:
                        findings.append(f"case {index}: witness_log exists without inline witness")
                else:
                    try:
                        candidate = (root / witness_log).resolve() if isinstance(witness_log, str) and witness_log else None
                        valid_witness_log = candidate is not None and not Path(witness_log).is_absolute() and candidate.is_relative_to(root) and candidate.is_file() and candidate.stat().st_size <= 65536
                    except (OSError, ValueError):
                        valid_witness_log = False
                    if not valid_witness_log:
                        findings.append(f"case {index}: missing or invalid unittest witness_log")
                    else:
                        try:
                            recorded = read_json(candidate)
                            if recorded != witness or unittest_witness_error(recorded, expected_tests) != witness_error:
                                findings.append(f"case {index}: unittest witness_log disagrees with inline witness")
                        except (OSError, ValueError) as exc:
                            findings.append(f"case {index}: cannot read unittest witness_log: {exc}")
            elif kind == "numeric":
                successful = _numeric_evidence(root, data, case, index, findings)
                if case.get("witness") is not None or case.get("witness_log") is not None:
                    findings.append(f"case {index}: numeric case has undeclared unittest witness")
            elif case.get("witness") is not None or case.get("witness_log") is not None:
                findings.append(f"case {index}: command case has undeclared unittest witness")
            if (case_status == "passed") != successful:
                findings.append(f"case {index}: test status disagrees with process or scoring outcome")
        log = case.get("log")
        if case_status in ("passed", "failed", "error") and (not isinstance(log, str) or not log):
            findings.append(f"missing case log: {identifier}")
        elif log is not None:
            if not isinstance(log, str) or not log:
                findings.append(f"invalid case log: {identifier}")
            else:
                try:
                    candidate = (root / log).resolve()
                    valid_log = not Path(log).is_absolute() and candidate.is_relative_to(root) and candidate.is_file()
                except (OSError, ValueError):
                    valid_log = False
                if not valid_log:
                    findings.append(f"missing or invalid case log: {identifier}")

    if not any(case.get("group") == "fail_to_pass" for case in cases):
        findings.append("result has no fail_to_pass cases")
    all_pass = all(case.get("status") == "passed" for case in cases)
    if status == "resolved" and not all_pass:
        findings.append("resolved result contains missing or failed required tests")
    if status == "test_failed" and not any(case.get("status") == "failed" for case in cases):
        findings.append("test_failed result contains no failed case")
    if status in ("invalid_patch", "build_failed") and any(case.get("status") != "not_run" for case in cases):
        findings.append("pre-test failure contains executed test cases")
    groups = data.get("groups")
    if not isinstance(groups, dict):
        findings.append("result groups must be an object")
    else:
        for group in sorted(GROUPS):
            expected = {"passed": sum(c.get("status") == "passed" for c in cases if c.get("group") == group),
                        "total": sum(c.get("group") == group for c in cases)}
            counts = groups.get(group)
            if not isinstance(counts, dict) or any(type(counts.get(key)) is not int for key in expected) or counts != expected:
                findings.append(f"incorrect {group} counts")
    if not (root / "patch.diff").is_file():
        findings.append("missing submitted patch")
    return outcome()
