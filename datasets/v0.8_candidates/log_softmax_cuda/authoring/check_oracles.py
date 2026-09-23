"""Small CPU-only authoring checks; these are not CUDA candidate controls."""
import ast
import copy
import json
import math
from pathlib import Path
import struct

from generate_oracles import reference, operation
from op_bench.benchmark import TaskSpec
from op_bench.benchmark.numeric import judge, load_oracle


ROOT = Path(__file__).resolve().parent.parent


def scalar_reference(item):
    values = item["values"]
    if item["dtype"] == "float32":
        values = [struct.unpack("f", struct.pack("f", value))[0] for value in values]
    rows, width = item["shape"]
    indices = ([row * width + col for col in range(width)] for row in range(rows)) if item["dim"] % 2 else (
        [row * width + col for row in range(rows)] for col in range(width))
    observed = [0.0] * len(values)
    for group in indices:
        selected = [values[index] for index in group]
        anchor = max(selected)
        exps = [math.exp(value - anchor) for value in selected]
        denominator = math.fsum(exps)
        probabilities = [value / denominator for value in exps]
        function = item["operation"]
        if function == "log_softmax":
            result = [value - anchor - math.log(denominator) for value in selected]
        elif function == "softmax":
            result = probabilities
        else:
            gradient = [item["gradient"][index] for index in group]
            total = math.fsum(gradient) if function == "grad_log_softmax" else math.fsum(
                value * probability for value, probability in zip(gradient, probabilities))
            result = [value - probability * total for value, probability in zip(gradient, probabilities)] if function == "grad_log_softmax" else [
                probability * (value - total) for value, probability in zip(gradient, probabilities)]
        for index, value in zip(group, result):
            observed[index] = value
    return observed


def main():
    task = TaskSpec.load(ROOT / "task.json")
    checks = []
    for case in task.tests:
        oracle = load_oracle(task.grader_dir, case.oracle)
        independent = [value for item in oracle["input"]["operations"] for value in scalar_reference(item)]
        decision = judge(oracle, json.dumps({"shape": [len(independent)], "values": independent}).encode())
        assert decision["passed"], (case.id, decision)
        checks.append({"case_id": case.id, "stdlib_binary64_formula_matches_decimal_reference": True,
                       "compared_values": len(independent), "max_absolute_error_decimal": decision["max_absolute_error_decimal"]})
        if case.group == "fail_to_pass":
            normalized_wrong = []
            for item in oracle["input"]["operations"]:
                if item["operation"] == "log_softmax":
                    normalized_wrong.extend([-math.log(item["shape"][item["dim"] % 2])] * len(item["values"]))
                else:
                    normalized_wrong.extend(scalar_reference(item))
            wrong = judge(oracle, json.dumps({"shape": [len(normalized_wrong)], "values": normalized_wrong}).encode())
            assert not wrong["passed"] and wrong["mismatched_values"] > 0
            checks[-1]["uniform_log_probabilities_rejected_in_controller_check"] = wrong["mismatched_values"]
    for function in ("grad_log_softmax", "grad_softmax"):
        item = operation("small_derivative_check", 2, 3, function=function)
        analytic = reference(item)
        errors = []
        for index in range(len(item["values"])):
            objectives = []
            step = 1e-5
            for direction in (-1, 1):
                shifted = copy.deepcopy(item)
                shifted["operation"] = function.removeprefix("grad_")
                shifted["values"][index] += direction * step
                objectives.append(math.fsum(value * weight for value, weight in zip(reference(shifted), item["gradient"])))
            numerical = (objectives[1] - objectives[0]) / (2 * step)
            errors.append(abs(numerical - analytic[index]))
        assert max(errors) < 1e-8
        checks.append({"operation": function, "small_shape_gradient_finite_difference_check": True,
                       "max_error": max(errors), "step": step})
    for path in (ROOT / "public").glob("*.py"):
        ast.parse(path.read_text(), feature_version=(3, 10))
    result = {"date": "2026-09-09", "status": "authoring_checks_passed",
              "scope": "Task schema, bounded oracle parsing, independent scalar formula checks, small derivative finite differences, uniform-output rejection by controller comparison and Python 3.10 syntax. No candidate CUDA execution or full framework build occurred in this check.",
              "runtime_validation": "not_executed", "formal_admission": "not_established", "checks": checks}
    (ROOT / "validation/oracle_review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
