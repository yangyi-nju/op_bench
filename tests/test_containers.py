from __future__ import annotations

import json
import unittest

from op_bench.containers import ContainerManager
from op_bench.executor import CommandResult


class FakeDockerRunner:
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], timeout_sec: int) -> CommandResult:
        self.commands.append(command)
        if command[:2] == ["docker", "ps"]:
            return CommandResult(
                command=command,
                cwd="",
                exit_code=0,
                stdout="\n".join(json.dumps(record) for record in self.records),
                stderr="",
                duration_sec=0.1,
            )
        return CommandResult(
            command=command,
            cwd="",
            exit_code=0,
            stdout=command[-1],
            stderr="",
            duration_sec=0.1,
        )


class ContainerManagerTests(unittest.TestCase):
    def test_lists_only_managed_containers_and_normalizes_records(self) -> None:
        runner = FakeDockerRunner(
            [
                {"ID": "one", "Names": "op-bench-one", "State": "running", "Image": "image:one"},
                {"ID": "two", "Names": "op-bench-two", "State": "exited", "Image": "image:two"},
            ]
        )

        records = ContainerManager(runner=runner).list_managed()

        self.assertEqual([record.name for record in records], ["op-bench-one", "op-bench-two"])
        self.assertEqual(records[1].state, "exited")
        self.assertIn("label=op-bench.managed=true", runner.commands[0])

    def test_prune_stopped_defaults_to_preview(self) -> None:
        runner = FakeDockerRunner(
            [
                {"ID": "one", "Names": "op-bench-one", "State": "running", "Image": "image:one"},
                {"ID": "two", "Names": "op-bench-two", "State": "exited", "Image": "image:two"},
            ]
        )
        manager = ContainerManager(runner=runner)

        result = manager.prune_stopped(execute=False)

        self.assertEqual(result["candidates"], ["op-bench-two"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(len(runner.commands), 1)

    def test_prune_stopped_never_removes_running_containers(self) -> None:
        runner = FakeDockerRunner(
            [
                {"ID": "one", "Names": "op-bench-one", "State": "running", "Image": "image:one"},
                {"ID": "two", "Names": "op-bench-two", "State": "exited", "Image": "image:two"},
                {"ID": "three", "Names": "op-bench-three", "State": "dead", "Image": "image:three"},
            ]
        )
        manager = ContainerManager(runner=runner)

        result = manager.prune_stopped(execute=True)

        self.assertEqual(result["removed"], ["op-bench-two", "op-bench-three"])
        self.assertNotIn(["docker", "rm", "-f", "op-bench-one"], runner.commands)
        self.assertIn(["docker", "rm", "-f", "op-bench-two"], runner.commands)


if __name__ == "__main__":
    unittest.main()
