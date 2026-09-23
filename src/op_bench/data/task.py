"""Portable task descriptions for the independent patch evaluator.

Task descriptions are trusted benchmark inputs. Agent patches are not. The
visible view deliberately describes the problem and public environment only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path, PurePosixPath
from typing import Any


class TaskInputError(ValueError):
    """A task cannot be evaluated as described."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskInputError(f"{name} must be a nonempty string")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskInputError(f"{name} must be a positive number")
    import math

    if not math.isfinite(value) or value <= 0:
        raise TaskInputError(f"{name} must be a positive finite number")
    return float(value)


def _argv(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskInputError(f"{name} must be a nonempty argv array")
    if any(not isinstance(token, str) or "\0" in token for token in value):
        raise TaskInputError(f"{name} must contain strings without NUL characters")
    _text(value[0], name + "[0]")
    return tuple(value)


def _commands(value: Any, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise TaskInputError(f"{name} must be an array of argv arrays")
    return tuple(_argv(command, f"{name}[{index}]") for index, command in enumerate(value))


def _path(value: Any, name: str, task_dir: Path) -> Path:
    path = Path(_text(value, name)).expanduser()
    return (task_dir / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    revision: str | None = None
    repo_url: str | None = None


@dataclass(frozen=True)
class EnvironmentSpec:
    backend: str = "local"
    image: str | None = None
    python: str = "python3"
    prepare: tuple[tuple[str, ...], ...] = ()
    build: tuple[tuple[str, ...], ...] = ()
    cpus: float = 2.0
    memory: str = "4g"
    pids_limit: int = 512
    gpus: str | None = None
    prepare_timeout_sec: float = 300.0
    build_timeout_sec: float = 900.0
    environment: dict[str, str] = field(default_factory=dict)
    environment_id: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "image": self.image, "python": self.python,
            "prepare": [list(command) for command in self.prepare],
            "build": [list(command) for command in self.build],
            "cpus": self.cpus, "memory": self.memory, "pids_limit": self.pids_limit,
            "gpus": self.gpus, "prepare_timeout_sec": self.prepare_timeout_sec,
            "build_timeout_sec": self.build_timeout_sec,
            "environment": dict(self.environment),
            "environment_id": self.environment_id, "revision": self.revision,
        }


@dataclass(frozen=True)
class TestCase:
    id: str
    group: str
    argv: tuple[str, ...]
    timeout_sec: float = 60.0
    kind: str = "command"
    expected_tests: int = 1
    oracle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "group": self.group, "argv": list(self.argv),
                  "timeout_sec": self.timeout_sec, "kind": self.kind,
                  "expected_tests": self.expected_tests}
        if self.oracle is not None:
            result["oracle"] = self.oracle
        return result


def _environment(environment: Any) -> EnvironmentSpec:
    if not isinstance(environment, dict):
        raise TaskInputError("environment must be an object")
    backend = environment.get("backend", "local")
    if backend not in {"local", "docker"}:
        raise TaskInputError("environment.backend must be local or docker")
    image = environment.get("image")
    if image is not None:
        image = _text(image, "environment.image")
    if backend == "docker" and image is None:
        raise TaskInputError("Docker tasks require environment.image")
    pids_limit = environment.get("pids_limit", 512)
    if isinstance(pids_limit, bool) or not isinstance(pids_limit, int) or pids_limit < 1:
        raise TaskInputError("environment.pids_limit must be a positive integer")
    gpus = environment.get("gpus")
    if gpus is not None:
        gpus = _text(gpus, "environment.gpus")
    variables = environment.get("environment", {})
    if not isinstance(variables, dict) or any(
        not isinstance(key, str) or not key or "=" in key or "\0" in key
        or not isinstance(val, str) or "\0" in val for key, val in variables.items()
    ):
        raise TaskInputError("environment.environment must map valid names to strings")
    return EnvironmentSpec(
        backend=backend, image=image,
        python=_text(environment.get("python", "python3"), "environment.python"),
        prepare=_commands(environment.get("prepare", []), "environment.prepare"),
        build=_commands(environment.get("build", []), "environment.build"),
        cpus=_positive(environment.get("cpus", 2), "environment.cpus"),
        memory=_text(environment.get("memory", "4g"), "environment.memory"),
        pids_limit=pids_limit, gpus=gpus,
        prepare_timeout_sec=_positive(environment.get("prepare_timeout_sec", 300), "prepare_timeout_sec"),
        build_timeout_sec=_positive(environment.get("build_timeout_sec", 900), "build_timeout_sec"),
        environment=dict(variables),
        environment_id=_text(environment.get("environment_id"), "environment.environment_id"),
        revision=_text(environment.get("revision"), "environment.revision"),
    )


def _tests(raw_tests: Any, env: EnvironmentSpec, grader_dir: Path | None) -> tuple[TestCase, ...]:
    if not isinstance(raw_tests, list) or not raw_tests:
        raise TaskInputError("tests must contain at least one required test case")
    tests: list[TestCase] = []
    ids: set[str] = set()
    selectors: set[tuple] = set()
    for index, item in enumerate(raw_tests):
        if not isinstance(item, dict):
            raise TaskInputError(f"tests[{index}] must be an object")
        case_id = _text(item.get("id"), f"tests[{index}].id")
        group = item.get("group")
        if group not in {"fail_to_pass", "pass_to_pass"}:
            raise TaskInputError(f"Invalid test group: {group!r}")
        argv = _argv(item.get("argv"), f"tests[{index}].argv")
        kind = item.get("kind", "command")
        if kind not in {"command", "unittest", "numeric"}:
            raise TaskInputError("Test kind must be command, unittest or numeric")
        expected_tests = item.get("expected_tests", 1)
        if isinstance(expected_tests, bool) or not isinstance(expected_tests, int) or expected_tests < 1:
            raise TaskInputError("expected_tests must be a positive integer")
        if kind == "unittest" and (len(argv) < 2 or argv[0] not in {"{python}", env.python} or argv[1].startswith("-")):
            raise TaskInputError("unittest cases require argv ['{python}', 'test_script.py', ...]")
        oracle = None
        if kind == "numeric":
            oracle = _text(item.get("oracle"), f"tests[{index}].oracle")
            oracle_path = PurePosixPath(oracle)
            if (oracle_path.is_absolute() or ".." in oracle_path.parts
                    or "\\" in oracle or "\0" in oracle or not oracle_path.name
                    or ":" in oracle_path.parts[0]):
                raise TaskInputError("numeric oracle must be a relative path without '..'")
            if grader_dir is None:
                raise TaskInputError("numeric cases require grader_dir")
            if any("{grader}" in token for token in argv):
                raise TaskInputError("numeric worker argv must not reference {grader}")
            if expected_tests != 1:
                raise TaskInputError("numeric expected_tests must be 1 protocol case")
        elif "oracle" in item:
            raise TaskInputError("oracle is only valid for numeric cases")
        selector = (argv, oracle) if kind == "numeric" else (argv, None)
        if case_id in ids or selector in selectors:
            raise TaskInputError("Test case IDs and command selectors must be unique")
        ids.add(case_id)
        selectors.add(selector)
        tests.append(TestCase(case_id, group, argv, _positive(item.get("timeout_sec", 60), "timeout_sec"),
                              kind, expected_tests, oracle))
    if not any(case.group == "fail_to_pass" for case in tests):
        raise TaskInputError("A repair task requires at least one fail_to_pass case")
    return tuple(tests)


@dataclass(frozen=True)
class TaskVariant:
    """A required grading environment for the same source and frozen patch."""
    variant_id: str
    environment: EnvironmentSpec
    tests: tuple[TestCase, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"variant_id": self.variant_id, "environment": self.environment.to_dict(),
                "tests": [case.to_dict() for case in self.tests]}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    statement: str
    source: SourceSpec
    environment: EnvironmentSpec
    tests: tuple[TestCase, ...]
    grader_dir: Path | None
    task_dir: Path
    public_commands: tuple[tuple[str, ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    task_revision: str | None = None
    scope: str | None = None
    defect_group: str | None = None
    scoring_revision: str | None = None
    variants: tuple[TaskVariant, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        path = Path(path).expanduser().resolve()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskInputError(f"Cannot read task description {path}: {exc}") from exc
        if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 2:
            raise TaskInputError("Unsupported task schema_version: v0.8 requires task schema 2 with explicit identities. "
                                 "Migrate older task bundles before execution; historical results remain readable.")
        task_dir = path.parent
        source = value.get("source")
        environment = value.get("environment", {})
        if not isinstance(source, dict) or not isinstance(environment, dict):
            raise TaskInputError("source and environment must be objects")
        source_path = _path(source.get("path"), "source.path", task_dir)
        revision = source.get("revision")
        if revision is not None:
            revision = _text(revision, "source.revision")
        repo_url = source.get("repo_url")
        if repo_url is not None:
            repo_url = _text(repo_url, "source.repo_url")
        if source_path.exists() and not source_path.is_dir():
            raise TaskInputError(f"source.path exists but is not a directory: {source_path}")
        if not source_path.exists() and not (repo_url and revision):
            raise TaskInputError("Missing source.path requires both source.repo_url and source.revision; explicitly prepare it before evaluation")
        grader = value.get("grader_dir")
        grader_dir = None if grader is None else _path(grader, "grader_dir", task_dir)
        if grader_dir is not None and not grader_dir.is_dir():
            raise TaskInputError(f"grader_dir is not a directory: {grader_dir}")
        # A task bundle commonly contains gold patches beside task.json. Neither
        # that bundle nor the private grader may become the source workspace.
        if task_dir.is_relative_to(source_path):
            raise TaskInputError("source.path must not contain the task description directory")
        if grader_dir is not None and (
            grader_dir.is_relative_to(source_path) or source_path.is_relative_to(grader_dir)
        ):
            raise TaskInputError("source.path and grader_dir must not overlap")
        schema_version = value["schema_version"]
        env = _environment(environment)
        tests = _tests(value.get("tests"), env, grader_dir)
        variants = []
        raw_variants = value.get("variants", [])
        if not isinstance(raw_variants, list):
            raise TaskInputError("variants must be an array")
        variant_ids = set()
        for index, item in enumerate(raw_variants):
            if not isinstance(item, dict):
                raise TaskInputError(f"variants[{index}] must be an object")
            if set(item) - {"variant_id", "environment", "tests", "required"}:
                raise TaskInputError("unknown variant fields; variants cannot change task inputs")
            if item.get("required", True) is not True:
                raise TaskInputError("all declared variants must be required")
            variant_id = _text(item.get("variant_id"), f"variants[{index}].variant_id")
            if any(char in variant_id for char in ("/", "\\", "\0")) or variant_id in {".", ".."}:
                raise TaskInputError("variant_id must be a simple identifier without path separators")
            if variant_id in variant_ids:
                raise TaskInputError("variant_id values must be unique")
            variant_ids.add(variant_id)
            # An explicit variant environment is a complete declaration. A
            # partial dict must not silently inherit a different image identity.
            variant_env = (_environment(item["environment"])
                           if "environment" in item else env)
            variant_tests = (_tests(item["tests"], variant_env, grader_dir)
                             if "tests" in item else _tests(value["tests"], variant_env, grader_dir))
            variants.append(TaskVariant(variant_id, variant_env, variant_tests))
        scope = _text(value.get("scope"), "scope")
        if scope not in {"operator", "operator_integration", "framework_support", "uncertain"}:
            raise TaskInputError("unsupported task scope")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TaskInputError("metadata must be an object")
        return cls(
            task_id=_text(value.get("task_id"), "task_id"),
            statement=_text(value.get("statement"), "statement"),
            source=SourceSpec(source_path, revision, repo_url), environment=env,
            tests=tuple(tests), grader_dir=grader_dir, task_dir=task_dir,
            public_commands=_commands(value.get("public_commands", []), "public_commands"),
            metadata=metadata, schema_version=schema_version,
            task_revision=_text(value.get("task_revision"), "task_revision"),
            scope=scope,
            defect_group=_text(value.get("defect_group"), "defect_group"),
            scoring_revision=_text(value.get("scoring_revision"), "scoring_revision"),
            variants=tuple(variants),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "task_id": self.task_id,
            "statement": self.statement,
            "source": {"path": str(self.source.path), "revision": self.source.revision,
                       "repo_url": self.source.repo_url},
            "environment": self.environment.to_dict(),
            "grader_dir": str(self.grader_dir) if self.grader_dir else None,
            "tests": [case.to_dict() for case in self.tests],
            "public_commands": [list(command) for command in self.public_commands],
            "metadata": self.metadata,
            "task_revision": self.task_revision, "scope": self.scope,
            "defect_group": self.defect_group, "scoring_revision": self.scoring_revision,
            "variants": [variant.to_dict() for variant in self.variants],
        }

    def required_variants(self) -> tuple[TaskVariant, ...]:
        return self.variants or (TaskVariant("default", self.environment, self.tests),)

    def validate_output_path(self, output: str | Path) -> Path:
        """Resolve a result destination without writing into task inputs."""
        output = Path(output).expanduser().resolve()
        if output.is_relative_to(self.source.path.resolve()) or (
            self.grader_dir is not None and output.is_relative_to(self.grader_dir.resolve())
        ):
            raise TaskInputError("Output must be outside task source and grader inputs")
        return output

    def as_variant(self, variant_id: str) -> "TaskSpec":
        for variant in self.required_variants():
            if variant.variant_id == variant_id:
                return replace(self, environment=variant.environment, tests=variant.tests, variants=())
        raise TaskInputError(f"unknown required variant: {variant_id}")

    def identity_dict(self) -> dict[str, Any]:
        def environment_identity(env: EnvironmentSpec) -> dict[str, str | None]:
            return {"environment_id": env.environment_id, "revision": env.revision}
        variants = [{"variant_id": variant.variant_id,
                     "environment": environment_identity(variant.environment)}
                    for variant in self.required_variants()]
        complete = all((self.task_revision, self.scope, self.defect_group, self.scoring_revision,
                        self.environment.environment_id, self.environment.revision)) and all(
                            all(item["environment"].values()) for item in variants)
        return {"status": "declared" if complete else "legacy_unidentified",
                "task_id": self.task_id, "task_revision": self.task_revision,
                "scope": self.scope, "defect_group": self.defect_group,
                "scoring_revision": self.scoring_revision,
                "solver_environment": environment_identity(self.environment),
                "required_variants": variants}

    def visible_dict(self) -> dict[str, Any]:
        """Public task input; arbitrary metadata is private unless curated elsewhere."""
        env = self.environment.to_dict()
        # These are evaluator execution recipes. Public build/test entry points
        # are curated separately; candidate builds receive no private grader.
        for key in ("prepare", "build", "prepare_timeout_sec", "build_timeout_sec"):
            env.pop(key)
        return {
            "schema_version": self.schema_version, "task_id": self.task_id,
            "statement": self.statement, "environment": env,
            "task_revision": self.task_revision, "scope": self.scope,
            "required_variants": [
                {"variant_id": variant.variant_id,
                 "environment": {key: val for key, val in variant.environment.to_dict().items()
                                 if key not in {"prepare", "build", "prepare_timeout_sec", "build_timeout_sec"}}}
                for variant in self.required_variants()],
            "public_commands": [list(command) for command in self.public_commands],
        }
