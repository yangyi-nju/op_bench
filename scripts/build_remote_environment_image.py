#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.remote import RemoteDockerExecutor, load_hosts_config


_IMAGE = re.compile(r"^op-bench/[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")


def environment_directory(value: str) -> Path:
    path = Path(value).resolve()
    environments_root = (ROOT / "environments").resolve()
    try:
        path.relative_to(environments_root)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("build context must be under environments/") from exc
    if not path.is_dir() or not (path / "Dockerfile").is_file():
        raise argparse.ArgumentTypeError("build context must contain Dockerfile")
    return path


def image_name(value: str) -> str:
    if not _IMAGE.fullmatch(value):
        raise argparse.ArgumentTypeError("image must be a tagged op-bench image name")
    return value


def sanitized_build_tail(stdout: str, stderr: str, remote_workspace: str) -> str:
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    tail = "\n".join(lines[-40:])
    return tail.replace(remote_workspace, "<remote-context>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one repository-declared environment on a configured remote Docker host "
            "without emitting remote connection details."
        )
    )
    parser.add_argument("--context", required=True, type=environment_directory)
    parser.add_argument("--image", required=True, type=image_name)
    parser.add_argument("--host", default="gpu-a10")
    parser.add_argument(
        "--hosts-config",
        default=str(ROOT / "configs/remote_hosts.json"),
    )
    parser.add_argument("--output")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 60 <= args.timeout_sec <= 7200:
        raise SystemExit("--timeout-sec must be between 60 and 7200")

    hosts = load_hosts_config(args.hosts_config)
    host = hosts.get(args.host)
    if host is None:
        raise SystemExit("requested remote host is not configured")

    build_id = uuid.uuid4().hex[:12]
    executor = RemoteDockerExecutor(
        host=host,
        image=args.image,
        container_name=f"op-bench-env-build-{build_id}",
    )
    sync_result = executor.sync_to_remote(args.context, timeout_sec=600)
    if sync_result.exit_code != 0:
        raise SystemExit(f"remote environment context sync failed (exit={sync_result.exit_code})")

    try:
        build_result = executor._ssh(
            [
                "docker",
                "build",
                "--pull=false",
                "--label",
                "op-bench.managed=true",
                "--tag",
                args.image,
                "--file",
                f"{executor.remote_workspace}/Dockerfile",
                executor.remote_workspace,
            ],
            timeout_sec=args.timeout_sec,
        )
        if build_result.exit_code != 0:
            detail = sanitized_build_tail(
                build_result.stdout,
                build_result.stderr,
                executor.remote_workspace,
            )
            message = f"remote environment build failed (exit={build_result.exit_code})"
            if detail:
                message += "\n" + detail
            raise SystemExit(message)

        inspect_result = executor._ssh(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image],
            timeout_sec=60,
        )
        if inspect_result.exit_code != 0:
            raise SystemExit("built remote image could not be inspected")
        image_id = inspect_result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise SystemExit("remote image inspection returned an invalid image ID")

        payload = {
            "context": args.context.relative_to(ROOT).as_posix(),
            "image": args.image,
            "image_id": image_id,
            "platform": "linux/amd64",
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
    finally:
        executor.close(timeout_sec=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
