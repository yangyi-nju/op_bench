"""Local entry-point control only: no Torch build, CUDA device or admission claim.

Run from the repository root with PYTHONPATH=src. The bridge-only patch computes
the public formula without changing Torch; the frozen driver must not execute it.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from op_bench.benchmark.numeric import judge, load_oracle
from op_bench.benchmark.spec import TaskSpec


ROOT = Path(__file__).resolve().parent.parent


def main():
    task = TaskSpec.load(ROOT / "task.json")
    with tempfile.TemporaryDirectory(prefix="opbench-fixed-driver-control-") as directory:
        workspace = Path(directory)
        # Exercise only the existing public asset installation command. This is
        # deliberately not a framework preparation or build execution.
        install = [token.replace("{python}", sys.executable) for token in task.environment.prepare[1]]
        subprocess.run(install, cwd=workspace, check=True, capture_output=True, timeout=10)
        subprocess.run(["git", "apply", str(ROOT / "controls/bridge_only.patch")],
                       cwd=workspace, check=True, capture_output=True, timeout=10)
        comparisons = []
        for case in task.tests:
            oracle = load_oracle(task.grader_dir, case.oracle)
            request = json.dumps(oracle["input"]).encode()
            bridge = subprocess.run([sys.executable, "-I", str(workspace / "opbench_public/numeric_worker.py")],
                                    input=request, cwd=workspace, capture_output=True, timeout=20)
            bridge_result = judge(oracle, bridge.stdout)
            assert bridge.returncode == 0 and bridge_result["passed"], "Negative bridge control must reproduce the public math"
            fixed = [token.replace("{python}", sys.executable).replace("{workspace}", str(workspace))
                     for token in case.argv]
            actual = subprocess.run(fixed, input=request, cwd=workspace, capture_output=True, timeout=20)
            # This workspace has no built framework. Replacing its self-test
            # copy with a formula cannot satisfy the fixed driver's API entry.
            assert actual.returncode != 0 and not actual.stdout.strip(), "Frozen driver unexpectedly used the patched self-test bridge"
            comparisons.append({"case": case.id, "bridge_formula_values": bridge_result["compared_values"],
                                "fixed_driver_exit_code_without_framework": actual.returncode,
                                "fixed_driver_produced_observation": False})
    print(json.dumps({"entrypoint_controls": comparisons,
        "scope": "Local bridge dispatch and missing-build prerequisite only; no CUDA behavior or admission established"}, indent=2))


if __name__ == "__main__":
    main()
