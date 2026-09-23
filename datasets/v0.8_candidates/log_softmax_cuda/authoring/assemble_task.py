"""Assemble the portable task's explicitly public preparation commands."""
import json
from pathlib import Path

from generate_oracles import batches


ROOT = Path(__file__).resolve().parent.parent
BASE = "2409b49a33c0ef594d89f9f477d56abad47e65bf"


def main():
    recipe = json.loads((ROOT / "environment/build_recipe.json").read_text())
    files = {name: (ROOT / "public" / name).read_text() for name in ("numeric_worker.py", "reproduce.py")}
    fixed_driver = files["numeric_worker.py"].replace(
        "workspace = Path(__file__).resolve().parent.parent", "workspace = Path(sys.argv[1]).resolve()")
    install_public = (
        "from pathlib import Path\n"
        "root=Path('opbench_public'); root.mkdir(exist_ok=True)\n"
        f"files={files!r}\n"
        "for name,content in files.items(): (root/name).write_text(content,encoding='utf-8')\n"
    )
    prerequisites = (
        "import importlib.util,pathlib,shutil,subprocess\n"
        "missing=[name for name in ('git','cmake','ninja','gcc','g++','nvcc') if shutil.which(name) is None]\n"
        "missing += [name for name in ('numpy','yaml','astunparse','setuptools','sympy','filelock','typing_extensions','jinja2','fsspec') if importlib.util.find_spec(name) is None]\n"
        "assert not missing, 'Missing offline full-source build prerequisites: '+str(missing)\n"
        "assert pathlib.Path('setup.py').is_file() and pathlib.Path('aten/src/ATen/native/cuda/SoftMax.cu').is_file(), 'Full upstream source checkout is required'\n"
        "subprocess.run(['nvcc','--version'],check=True)\n"
    )
    task = {
        "schema_version": 2, "task_id": "opbench-cuda-logsoftmax-tail-001", "task_revision": "2",
        "scoring_revision": "2", "scope": "operator", "defect_group": "pytorch:cuda-softmax-aligned-prefix-tail-remainder",
        "statement": (
            "CUDA torch.nn.functional.log_softmax returns incorrect values for some float64 row reductions with widths just above 512, including a (5,513) tensor. Repair the complete provided framework source so each output along the selected dimension equals x_i - log(sum_j(exp(x_j))) to the declared numerical tolerance, using a stable equivalent formula when needed. Merely making exponentiated outputs sum to one is insufficient: unequal inputs must retain their distinct probabilities. Preserve ordinary CUDA softmax values, nearby aligned widths, float32 behavior and the first-order input gradients of softmax/log_softmax. "
            "This candidate exercises finite two-dimensional float32/float64 tensors with 1 to 5 rows, widths from 1 to 1024 and input values in [-608,608], including noncontiguous tensors, either axis and equivalent negative dimensions. Outputs retain input shape, dtype and CUDA device. For float64 output and input gradients the elementwise bound is abs(actual-reference) <= 1e-10 + 1e-11*abs(reference); for float32 it is 2e-5 + 2e-6*abs(reference). Gradient tests use finite upstream gradients bounded by 1. Device/shape/dtype diagnostics are execution checks in the public harness; final numeric comparisons occur independently. Nonfinite inputs, empty dimensions, half/bfloat16, higher-order gradients, graph compilation, distributed execution and performance are outside this candidate's acceptance scope. "
            "Run python3 setup.py build to compile the whole configured CPU/CUDA framework, then python3 -I opbench_public/reproduce.py for a public numerical reproducer. The preparation recipe supplies the public numeric worker and reproducer as readable, editable self-test copies. Final scoring runs a fixed, published copy of the numeric worker from the frozen task command with python3 -I -c, replacing only its main workspace locator with Path(sys.argv[1]).resolve(). It imports and calls the source-built torch APIs; modifying opbench_public/numeric_worker.py alone cannot change the scoring entry point or satisfy this API repair. The fixed driver contains no oracle, expected values or pass/fail judgment and receives only the current input batch. Changes may affect any source, build or public helper file consistent with the API and execution contract. No specific kernel implementation or patch path is mandated. "
            "The environment provides the CUDA toolkit and Python build dependencies. All preparation/build outputs must stay in the workspace; outside installations do not carry into grading. Scoring loads the package and native extension produced under build/lib.* and runs on the selected CUDA compute-capability 7.0 device. Each numeric case receives only its input batch in a fresh grader-free execution session, returns flattened values, and is compared with a private independent reference after execution stops. This is a migration candidate pending real source-build controls and admission, not a formally admitted benchmark task."
        ),
        "source": {"path": f"../../../.op_bench_cache/sources/pytorch/pytorch/{BASE}/source",
                   "repo_url": "https://github.com/pytorch/pytorch.git", "revision": BASE},
        "environment": {
            "backend": "docker", "image": recipe["dependency_image"], "python": "python3",
            "environment_id": "cuda-cu124-sm70-python310-source", "revision": "1",
            "prepare": [["{python}", "-I", "-c", prerequisites], ["{python}", "-I", "-c", install_public]],
            "build": [["{python}", "setup.py", "build"]],
            "prepare_timeout_sec": 120, "build_timeout_sec": recipe["build_timeout_sec"],
            "cpus": 12, "memory": "96g", "pids_limit": 4096, "gpus": "device=2",
            "environment": recipe["environment"],
        },
        "grader_dir": "grader",
        "public_commands": [["{python}", "setup.py", "build"],
                            ["{python}", "-I", "{workspace}/opbench_public/reproduce.py"]],
        "tests": [{"id": name, "group": batch["group"], "kind": "numeric", "oracle": name + ".json",
                   "argv": ["{python}", "-I", "-c", fixed_driver, "{workspace}"], "timeout_sec": 120}
                  for name, batch in batches().items()],
        "metadata": {
            "formal_benchmark_member": False, "status": "candidate", "admission_status": "not_admitted",
            "validation_records": "validation/",
            "origin": "upstream_issue_migration", "upstream_pr": "https://github.com/pytorch/pytorch/pull/144009",
            "upstream_issue": "https://github.com/pytorch/pytorch/issues/143644",
            "migration_review": "migration_review.md", "environment_recipe": "environment/build_recipe.json",
            "oracle_method": "Independent scalar Python Decimal arithmetic at 90 decimal digits, stable log-sum-exp and analytical first derivatives, then binary64 rounding. No candidate or framework imports generate the private expected outputs.",
            "source_loading": "complete_source_build_no_overlay",
            "gpu_allocation": "device=2 is a deployment selection for the currently scheduled V100 host, not a portable GPU identity or exclusive allocation. Rebind explicitly and record actual device before execution.",
            "trust_limit": "The independent controller verifies output numbers. Editing the workspace self-test worker does not change the fixed scoring driver. The driver still imports candidate torch in the same process; deliberate interception of imports or runtime metadata is not ruled out. Device/dtype diagnostics and mapped libraries do not prove CUDA computation attribution or exclude CPU fallback; high-risk alternatives need source/runtime review.",
            "scoring_driver": "fixed_public_worker_inline_from_task_with_explicit_workspace_argument",
        },
    }
    (ROOT / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
