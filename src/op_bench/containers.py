from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from op_bench.executor import CommandResult, ensure_text


DockerRunner = Callable[[list[str], int], CommandResult]


@dataclass(frozen=True)
class ContainerRecord:
    container_id: str
    name: str
    state: str
    image: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ContainerManager:
    def __init__(self, runner: DockerRunner | None = None) -> None:
        self.runner = runner or run_docker_command

    def list_managed(self) -> list[ContainerRecord]:
        result = self.runner(
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                "label=op-bench.managed=true",
                "--format",
                "{{json .}}",
            ],
            60,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or "docker ps failed")
        records: list[ContainerRecord] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(
                ContainerRecord(
                    container_id=str(data.get("ID", "")),
                    name=str(data.get("Names", "")),
                    state=str(data.get("State", "")).lower(),
                    image=str(data.get("Image", "")),
                )
            )
        return records

    def prune_stopped(self, execute: bool = False) -> dict[str, object]:
        candidates = [
            record.name
            for record in self.list_managed()
            if record.state in {"created", "dead", "exited"}
        ]
        removed: list[str] = []
        failures: list[dict[str, str]] = []
        if execute:
            for name in candidates:
                result = self.runner(["docker", "rm", "-f", name], 60)
                if result.exit_code == 0:
                    removed.append(name)
                else:
                    failures.append({"name": name, "error": result.stderr or result.stdout})
        return {
            "execute": execute,
            "candidates": candidates,
            "removed": removed,
            "failures": failures,
        }


def run_docker_command(command: list[str], timeout_sec: int) -> CommandResult:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return CommandResult(
            command=command,
            cwd="",
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_sec=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            cwd="",
            exit_code=124,
            stdout=ensure_text(exc.stdout),
            stderr=ensure_text(exc.stderr),
            duration_sec=time.monotonic() - start,
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(
            command=command,
            cwd="",
            exit_code=1,
            stdout="",
            stderr=str(exc),
            duration_sec=time.monotonic() - start,
        )
