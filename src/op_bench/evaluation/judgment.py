"""Pure grading rules shared by execution and saved-evidence verification."""
from __future__ import annotations


SCORABLE_STATUSES = {"resolved", "test_failed", "invalid_patch", "build_failed"}
UNKNOWN_STATUSES = {"environment_error", "evaluation_error"}
RESULT_STATUSES = SCORABLE_STATUSES | UNKNOWN_STATUSES


def unittest_witness_error(witness: dict | None, expected_tests: int) -> str | None:
    if not isinstance(witness, dict):
        return "missing_unittest_execution_witness"
    counts = ("runner_invocations", "tests_run", "failures", "errors", "skipped",
              "expected_failures", "unexpected_successes")
    if type(witness.get("schema_version")) is not int or witness.get("schema_version") != 1 or any(
        type(witness.get(key)) is not int or witness[key] < 0 for key in counts
    ):
        return "invalid_unittest_execution_witness"
    if witness["runner_invocations"] < 1:
        return "unittest_runner_was_not_executed"
    if witness["tests_run"] != expected_tests:
        return "unexpected_unittest_test_count"
    if any(witness[key] for key in counts[2:]):
        return "unittest_case_failed_or_was_not_completed"
    return None


def variant_status(results: dict, required: list[str]) -> str:
    if set(results) != set(required):
        return "evaluation_error"
    statuses = {result["status"] for result in results.values()}
    if "environment_error" in statuses:
        return "environment_error"
    if "evaluation_error" in statuses:
        return "evaluation_error"
    if statuses == {"resolved"}:
        return "resolved"
    if statuses == {"invalid_patch"}:
        return "invalid_patch"
    # Pre-test failures remain separate in each variant; the logical task is
    # unsuccessful. Variants do not create new attempts or independent tasks.
    if "build_failed" in statuses or "invalid_patch" in statuses:
        return "build_failed"
    return "test_failed"
