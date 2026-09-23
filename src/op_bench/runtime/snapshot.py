"""Freeze portable task inputs for continued evaluation after source changes."""
from __future__ import annotations

from pathlib import Path
import shutil

from op_bench.runtime.execution import ExecutionError, materialize_source, resolve_image_id
from op_bench.io import write_json
from op_bench.data.task import TaskSpec


def snapshot_task(task: TaskSpec, directory: Path) -> TaskSpec:
    """Create source/, grader/, origin.json and task.json in a new directory.

    Source export preserves its declared revision. Docker image tags are
    resolved to an installed image ID, without pulling or starting containers.
    Task metadata and admission declarations are copied without modification.
    """
    directory = task.validate_output_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    origin = materialize_source(task, directory / "source")
    origin["source_path"] = str(task.source.path)
    data = task.to_dict()
    data["source"] = {"path": "source"}
    if task.grader_dir is not None:
        shutil.copytree(task.grader_dir, directory / "grader")
        data["grader_dir"] = "grader"
    images = {}

    def pin_image(environment: dict) -> tuple[str | None, str | None]:
        if environment["backend"] != "docker":
            return None, None
        declared = environment["image"]
        if declared in images:
            environment["image"] = images[declared]
            return declared, images[declared]
        image_id = resolve_image_id(declared, timeout_sec=30)
        environment["image"] = image_id
        images[declared] = image_id
        return declared, image_id

    declared, image_id = pin_image(data["environment"])
    if image_id is not None:
        origin.update(declared_image=declared, image_id=image_id)
    if data.get("variants"):
        origin["variant_images"] = {}
        for variant in data["variants"]:
            declared, image_id = pin_image(variant["environment"])
            origin["variant_images"][variant["variant_id"]] = {"declared_image": declared, "image_id": image_id}
    write_json(directory / "origin.json", origin)
    write_json(directory / "task.json", data)
    return TaskSpec.load(directory / "task.json")
