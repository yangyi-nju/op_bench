#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import AgentSpec, ContentIdentity
from op_bench.runtime.legacy import run_manifest_from_v05_dataset
from op_bench.runtime.validation import ContractError
from scripts.validate_dataset import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline OpBench v0.6 RunManifest from checked-in v0.5 data. "
            "This command performs no Agent launch or remote validation."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to a checked-in v0.5 dataset.json.")
    parser.add_argument("--output", required=True, help="Destination for canonical manifest JSON.")
    parser.add_argument("--agent", required=True, help="Public Agent identifier.")
    parser.add_argument("--model", required=True, help="Public model identifier.")
    parser.add_argument("--adapter", required=True, help="Public adapter identifier.")
    parser.add_argument("--repeat", type=int, default=1, help="Attempts per task and Agent.")
    parser.add_argument(
        "--created-at",
        required=True,
        help="Explicit UTC RFC3339 timestamp at second precision, such as 2026-07-17T10:00:00Z.",
    )
    parser.add_argument("--system-prompt-id", default="example-system-prompt-v1")
    parser.add_argument("--task-prompt-id", default="example-task-prompt-v1")
    parser.add_argument("--feedback-policy", choices=("visible", "none"), default="visible")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    try:
        dataset_path = Path(args.dataset).resolve()
        with dataset_path.open("r", encoding="utf-8") as handle:
            dataset_data = json.load(handle)
        if not isinstance(dataset_data, dict):
            raise ContractError("dataset: expected object")
        errors = validate_dataset(
            dataset_data,
            dataset_path.parent,
            require_verified=True,
        )
        if errors:
            raise ContractError("dataset validation failed: " + "; ".join(errors))
        agent = _agent_spec(
            agent_id=args.agent,
            model_id=args.model,
            adapter_id=args.adapter,
            system_prompt_id=args.system_prompt_id,
            task_prompt_id=args.task_prompt_id,
            feedback_policy=args.feedback_policy,
        )
        manifest = run_manifest_from_v05_dataset(
            dataset_path,
            agents=(agent,),
            repeat=args.repeat,
            created_at=args.created_at,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
    except (ContractError, OSError, ValueError) as exc:
        print(f"cannot build RunManifest: {exc}", file=sys.stderr)
        return 1

    print(f"wrote offline run_manifest to {output}")
    return 0


def _agent_spec(
    *,
    agent_id: str,
    model_id: str,
    adapter_id: str,
    system_prompt_id: str,
    task_prompt_id: str,
    feedback_policy: str,
) -> AgentSpec:
    agent = _declared_identity("agent", agent_id)
    model = _declared_identity("model", model_id)
    adapter = _declared_identity("adapter", adapter_id)
    system_prompt = _declared_identity("prompt", system_prompt_id)
    task_prompt = _declared_identity("prompt", task_prompt_id)
    config = ContentIdentity(
        identity_type="agent_config",
        identifier=f"{agent_id}:{model_id}:{adapter_id}:{feedback_policy}",
        digest=canonical_sha256(
            {
                "agent": agent.to_dict(),
                "model": model.to_dict(),
                "adapter": adapter.to_dict(),
                "system_prompt": system_prompt.to_dict(),
                "task_prompt": task_prompt.to_dict(),
                "feedback_policy": feedback_policy,
            }
        ),
        digest_kind="canonical_config",
    )
    return AgentSpec(
        agent=agent,
        model=model,
        adapter=adapter,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        config=config,
        feedback_policy=feedback_policy,
    )


def _declared_identity(identity_type: str, identifier: str) -> ContentIdentity:
    return ContentIdentity(
        identity_type=identity_type,
        identifier=identifier,
        digest=canonical_sha256(
            {"identity_type": identity_type, "identifier": identifier, "source": "explicit-cli"}
        ),
        digest_kind="canonical_config",
    )


if __name__ == "__main__":
    raise SystemExit(main())
