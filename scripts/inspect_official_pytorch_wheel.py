#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.remote import RemoteDockerExecutor, load_hosts_config


_ALLOWED_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
_ALLOWED_BASE_IMAGES = {
    "op-bench/pytorch-boundary-cpu-source-build:py311",
    "op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
}
_WHEEL_NAME = re.compile(
    r"^torch-[0-9][A-Za-z0-9.]*dev[0-9]{8}\+(?:cpu|cu126)-"
    r"cp311-cp311-manylinux_2_28_x86_64\.whl$"
)


_REMOTE_SCRIPT = r'''
import ast
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

allowed_hosts = {"download.pytorch.org", "download-r2.pytorch.org"}
max_bytes = 3_000_000_000
url = sys.argv[1]


class OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise RuntimeError("wheel download redirected outside the official allowlist")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def constant_assignments(source):
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (str, type(None))):
            values[target.id] = value
    return values


request = urllib.request.Request(url, headers={"User-Agent": "op-bench-wheel-audit/1"})
opener = urllib.request.build_opener(OfficialRedirectHandler())
hasher = hashlib.sha256()
size = 0
wheel_path = None
try:
    with opener.open(request, timeout=60) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise RuntimeError("wheel exceeds the inspection size limit")
        with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as wheel:
            wheel_path = wheel.name
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise RuntimeError("wheel exceeds the inspection size limit")
                hasher.update(chunk)
                wheel.write(chunk)

    with zipfile.ZipFile(wheel_path) as archive:
        version_values = constant_assignments(
            archive.read("torch/version.py").decode("utf-8")
        )
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError("wheel contains an unexpected METADATA layout")
        metadata_version = None
        for line in archive.read(metadata_names[0]).decode("utf-8").splitlines():
            if line.startswith("Version: "):
                metadata_version = line.removeprefix("Version: ")
                break
        if metadata_version is None:
            raise RuntimeError("wheel METADATA does not declare Version")

    payload = {
        "artifact_id": urllib.parse.unquote(url.rsplit("/", 1)[-1]),
        "byte_size": size,
        "cuda": version_values.get("cuda"),
        "git_version": version_values.get("git_version"),
        "metadata_version": metadata_version,
        "sha256": hasher.hexdigest(),
        "torch_version": version_values.get("__version__"),
        "url": url,
    }
    if not payload["git_version"] or not payload["torch_version"]:
        raise RuntimeError("wheel version metadata is incomplete")
    print(json.dumps(payload, sort_keys=True))
finally:
    if wheel_path is not None:
        try:
            os.unlink(wheel_path)
        except FileNotFoundError:
            pass
'''


def validated_url(value: str) -> str:
    parsed = urlparse(value)
    wheel_name = unquote(parsed.path.rsplit("/", 1)[-1])
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise argparse.ArgumentTypeError("URL must use the official PyTorch wheel hosts")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("wheel URL must not contain a query or fragment")
    if not parsed.path.startswith("/whl/nightly/") or not _WHEEL_NAME.fullmatch(wheel_name):
        raise argparse.ArgumentTypeError(
            "URL must identify an official CPU/cu126 CPython 3.11 Linux nightly torch wheel"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a fixed official PyTorch nightly wheel in an existing remote Docker "
            "environment without exposing remote connection details."
        )
    )
    parser.add_argument("--url", required=True, type=validated_url)
    parser.add_argument("--host", default="gpu-a10")
    parser.add_argument(
        "--hosts-config",
        default=str(ROOT / "configs/remote_hosts.json"),
    )
    parser.add_argument(
        "--base-image",
        choices=sorted(_ALLOWED_BASE_IMAGES),
        required=True,
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

    executor = RemoteDockerExecutor(host=host, image=args.base_image)
    result = executor._ssh(
        [
            "docker",
            "run",
            "--rm",
            "--label",
            "op-bench.managed=true",
            "--entrypoint",
            "python",
            args.base_image,
            "-c",
            _REMOTE_SCRIPT,
            args.url,
        ],
        timeout_sec=args.timeout_sec,
    )
    if result.exit_code != 0:
        raise SystemExit(f"remote wheel inspection failed (exit={result.exit_code})")

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("remote wheel inspection returned no metadata")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit("remote wheel inspection returned invalid metadata") from exc

    expected_fields = {
        "artifact_id",
        "byte_size",
        "cuda",
        "git_version",
        "metadata_version",
        "sha256",
        "torch_version",
        "url",
    }
    if set(payload) != expected_fields or payload["url"] != args.url:
        raise SystemExit("remote wheel inspection metadata failed validation")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload["sha256"])):
        raise SystemExit("remote wheel inspection returned an invalid SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload["git_version"])):
        raise SystemExit("remote wheel inspection returned an invalid Git commit")

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
