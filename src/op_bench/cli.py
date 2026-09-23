"""The public OpBench CLI. Historical release scripts are not its runtime."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from op_bench.evaluation.controls import check_task
from op_bench.data.dataset import Dataset
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runtime.execution import materialize_source
from op_bench.runner.experiment import run_experiment
from op_bench.io import write_json
from op_bench.evaluation.replay import replay_attempt
from op_bench.runtime.provisioning import SourceProvisionError, provision_source
from op_bench.provenance import SOFTWARE_VERSION
from op_bench.results.report import summarize_run
from op_bench.data.task import TaskSpec
from op_bench.evaluation.verification import verify_evaluation


def _emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opbench", description="Independent evaluation of operator repairs")
    parser.add_argument("--version", action="version", version=f"OpBench {SOFTWARE_VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect core capabilities without requiring Agent CLIs")
    datasets = commands.add_parser("datasets", help="Read a portable task bundle")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("--dataset", required=True, type=Path)
    prepare = commands.add_parser("prepare", help="Export a clean task source and public input")
    prepare.add_argument("--task", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--fetch", action="store_true",
                         help="Prepare a missing source from repo_url and revision before exporting")
    prepare.add_argument("--fetch-timeout", type=_positive_integer, default=900,
                         help="Total source preparation time limit in seconds (default: 900)")
    baseline_build = commands.add_parser("build-baseline", help="Build a trusted unpatched workspace for explicit reuse")
    baseline_build.add_argument("--task", required=True, type=Path)
    baseline_build.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate", help="Score patches independently of an Agent")
    evaluate.add_argument("--task", required=True, type=Path)
    patch = evaluate.add_mutually_exclusive_group(required=True)
    patch.add_argument("--patch", type=Path)
    patch.add_argument("--baseline", action="store_true")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--baseline-artifact", type=Path,
                          help="Explicit trusted baseline build; candidate builds still run")
    run = commands.add_parser("run", help="Run models with the fixed OpBench tools and harness")
    run.add_argument("--experiment", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", help="Recompute an experiment report from results")
    report.add_argument("--run", required=True, type=Path)
    verify = commands.add_parser("verify", help="Check evidence structure and score arithmetic")
    verify.add_argument("--evaluation", required=True, type=Path)
    replay = commands.add_parser("replay", help="Reevaluate a frozen patch without rerunning its Agent or replacing history")
    replay.add_argument("--attempt", required=True, type=Path)
    replay.add_argument("--task", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--reason", required=True)
    replay.add_argument("--baseline-artifact", type=Path,
                        help="Explicit trusted baseline build; defaults to fresh evaluation and always rebuilds the frozen patch")
    controls = commands.add_parser("check-task", help="Run baseline/reference/alternative/mutation controls")
    controls.add_argument("--task", required=True, type=Path)
    controls.add_argument("--reference", required=True, type=Path)
    controls.add_argument("--alternative", action="append", default=[], type=Path)
    controls.add_argument("--mutation", action="append", default=[], type=Path)
    controls.add_argument("--output", required=True, type=Path)
    cache = controls.add_mutually_exclusive_group()
    cache.add_argument("--reuse-baseline", action="store_true",
                       help="Build one unpatched baseline per environment, then independently rebuild each control")
    cache.add_argument("--baseline-artifact", type=Path,
                       help="Reuse an existing trusted baseline while freezing a fresh control matrix")
    return parser


def _doctor() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "git": shutil.which("git"), "docker_cli": shutil.which("docker"),
              "docker_server": None, "agent_cli_required": False}
    if result["docker_cli"]:
        try:
            probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                   capture_output=True, text=True, timeout=10)
            if probe.returncode == 0:
                result["docker_server"] = probe.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    result["core_ready"] = sys.version_info >= (3, 12) and bool(result["git"])
    result["container_ready"] = bool(result["docker_server"])
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = _doctor()
            _emit(result)
            return 0 if result["core_ready"] else 2
        if args.command == "datasets":
            dataset = Dataset.load(args.dataset)
            _emit({"dataset_id": dataset.dataset_id, "version": dataset.version,
                   "tasks": len(dataset.tasks), "valid_structure": True,
                   "unprepared_task_ids": [task.task_id for task in dataset.tasks if not task.source.path.is_dir()],
                   "semantic_admission": "not_established_by_schema_validation"})
        elif args.command == "prepare":
            task = TaskSpec.load(args.task)
            args.output = task.validate_output_path(args.output)
            args.output.mkdir(parents=True, exist_ok=False)
            if args.fetch:
                try:
                    preparation = provision_source(task, timeout_sec=args.fetch_timeout)
                except SourceProvisionError as exc:
                    write_json(args.output / "preparation.json", exc.result or {"error": str(exc)})
                    raise
                write_json(args.output / "preparation.json", preparation)
            source = materialize_source(task, args.output / "workspace")
            write_json(args.output / "source.json", source)
            write_json(args.output / "task_input.json", task.visible_dict())
            _emit({"workspace": str((args.output / "workspace").resolve()), "source": source})
        elif args.command == "evaluate":
            task = TaskSpec.load(args.task)
            patch = args.patch.read_text(encoding="utf-8") if args.patch else ""
            result = PatchEvaluator().evaluate(task, patch, args.output, baseline_artifact=args.baseline_artifact)
            _emit(result)
            return 0 if result["status"] in {"resolved", "test_failed", "invalid_patch", "build_failed"} else 2
        elif args.command == "build-baseline":
            from op_bench.runtime.baseline import build_baseline
            result = build_baseline(TaskSpec.load(args.task), args.output)
            _emit(result)
            return 0 if result["status"] == "ready" else 2
        elif args.command == "run":
            result = run_experiment(args.experiment, args.output, resume=args.resume)
            _emit(result)
            return 0 if result["complete"] else 2
        elif args.command == "report":
            result = summarize_run(args.run)
            write_json(args.run / "report.json", result)
            _emit(result)
        elif args.command == "verify":
            result = verify_evaluation(args.evaluation)
            _emit(result)
            return 0 if result["valid"] else 2
        elif args.command == "replay":
            result = replay_attempt(args.attempt, args.task, args.output, reason=args.reason,
                                    baseline_artifact=args.baseline_artifact)
            _emit(result)
            return 0 if result["evaluation"]["status"] in {"resolved", "test_failed", "invalid_patch", "build_failed"} else 2
        elif args.command == "check-task":
            result = check_task(TaskSpec.load(args.task), args.reference.read_text(encoding="utf-8"), args.output,
                                alternatives=[p.read_text(encoding="utf-8") for p in args.alternative],
                                mutations=[p.read_text(encoding="utf-8") for p in args.mutation],
                                reuse_baseline=args.reuse_baseline, baseline_artifact=args.baseline_artifact)
            _emit(result)
            return 0 if result["execution_checks_passed"] else 2
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"opbench: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("opbench: interrupted; completed and in-progress attempt records were retained", file=sys.stderr)
        return 130
