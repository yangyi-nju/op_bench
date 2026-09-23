"""Maintainer-only scalar Decimal reference; never deployed to candidate code."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent.parent


def operation(name, rows, width, *, dtype="float64", function="log_softmax", dim=1,
              layout="contiguous", extreme=False, offset=0):
    values = [((((column * 37 + row * 13) % 113) - 56) / 8) + offset
              for row in range(rows) for column in range(width)]
    if extreme:
        values = [(600.0 if index % 31 == 0 else -600.0) + value / 16
                  for index, value in enumerate(values)]
    item = {"name": name, "operation": function, "shape": [rows, width], "values": values,
            "dtype": dtype, "dim": dim, "layout": layout}
    if function.startswith("grad_"):
        item["gradient"] = [(((index * 17) % 29) - 14) / 16 for index in range(rows * width)]
    return item


def reference(item):
    """Analytic output/first derivative, using no framework implementation."""
    def number(value):
        if item["dtype"] == "float32":
            value = struct.unpack("f", struct.pack("f", value))[0]
        return Decimal.from_float(float(value))
    rows, width = item["shape"]
    dimension = item["dim"] % 2
    values = [number(value) for value in item["values"]]
    gradient = [number(value) for value in item.get("gradient", [])]
    result = [None] * len(values)
    groups = ([row * width + column for column in range(width)] for row in range(rows)) if dimension == 1 else (
        [row * width + column for row in range(rows)] for column in range(width))
    with localcontext() as context:
        context.prec = 90
        for indices in groups:
            selected = [values[index] for index in indices]
            anchor = max(selected)
            exponentials = [(value - anchor).exp() for value in selected]
            total = sum(exponentials, Decimal(0))
            probabilities = [value / total for value in exponentials]
            if item["operation"] == "log_softmax":
                outputs = [value - anchor - total.ln() for value in selected]
            elif item["operation"] == "softmax":
                outputs = probabilities
            elif item["operation"] == "grad_log_softmax":
                weights = [gradient[index] for index in indices]
                aggregate = sum(weights, Decimal(0))
                outputs = [weight - probability * aggregate for weight, probability in zip(weights, probabilities)]
            elif item["operation"] == "grad_softmax":
                weights = [gradient[index] for index in indices]
                aggregate = sum((weight * probability for weight, probability in zip(weights, probabilities)), Decimal(0))
                outputs = [probability * (weight - aggregate) for weight, probability in zip(weights, probabilities)]
            else:
                raise ValueError("Unsupported reference operation")
            for index, output in zip(indices, outputs):
                result[index] = float(output)
    return result


def batches():
    return {
        "double_tail_values": {
            "group": "fail_to_pass", "atol": 1e-10, "rtol": 1e-11,
            "operations": [
                operation("log_double_513", 5, 513),
                operation("log_double_517", 5, 517),
                operation("log_double_515_strided", 5, 515, layout="noncontiguous"),
                operation("softmax_double_513", 5, 513, function="softmax"),
                operation("log_double_513_positive_shift", 3, 513, offset=600),
            ]},
        "float32_forward_regressions": {
            "group": "pass_to_pass", "atol": 2e-5, "rtol": 2e-6,
            "operations": [
                operation("log_float_511", 3, 511, dtype="float32"),
                operation("log_float_512", 3, 512, dtype="float32"),
                operation("log_float_513", 3, 513, dtype="float32"),
                operation("log_float_517_strided", 3, 517, dtype="float32", layout="noncontiguous"),
                operation("softmax_float_517", 3, 517, dtype="float32", function="softmax"),
                operation("log_float_axis0", 5, 17, dtype="float32", dim=-2),
                operation("log_float_extreme", 3, 512, dtype="float32", extreme=True),
            ]},
        "double_neighbor_and_gradient_regressions": {
            "group": "pass_to_pass", "atol": 1e-10, "rtol": 1e-11,
            "operations": [
                operation("log_double_511", 3, 511),
                operation("log_double_512", 3, 512),
                operation("log_double_514_aligned", 3, 514),
                operation("log_double_axis0", 5, 17, dim=0, layout="noncontiguous"),
                operation("log_double_extreme", 3, 512, extreme=True),
                operation("grad_log_double_512", 3, 512, function="grad_log_softmax"),
                operation("grad_softmax_double_512", 3, 512, function="grad_softmax"),
            ]},
    }


def main():
    (ROOT / "grader").mkdir(exist_ok=True)
    manifest = []
    for name, batch in batches().items():
        expected, segments = [], []
        for item in batch["operations"]:
            values = reference(item)
            segments.append({"name": item["name"], "start": len(expected), "count": len(values),
                             "shape": item["shape"], "operation": item["operation"], "dtype": item["dtype"]})
            expected.extend(values)
        oracle = {"schema_version": 1, "input": {"operations": batch["operations"]},
                  "expected": {"shape": [len(expected)], "values": expected},
                  "atol": batch["atol"], "rtol": batch["rtol"]}
        (ROOT / "grader" / (name + ".json")).write_text(json.dumps(oracle, allow_nan=False, separators=(",", ":")) + "\n")
        manifest.append({"case_id": name, "provisional_group": batch["group"], "segments": segments,
                         "atol": batch["atol"], "rtol": batch["rtol"]})
    (ROOT / "grader/segments.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
