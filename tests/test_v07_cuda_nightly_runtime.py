from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest

from op_bench.runtime import legacy


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_ID = "pytorch-nightly-20260407-torch2.12.0dev-cu126-py311"
PROFILE_ID = "remote-cuda-expansion-nightly-torch2.12.0dev20260407-cu126-py311-v1"
IMAGE = "op-bench/pytorch-nightly-cu126:2.12.0.dev20260407-py311"
WHEEL_ID = (
    "torch-2.12.0.dev20260407+cu126-cp311-cp311-"
    "manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = (
    "676cf93c822752750a4dd78bbaa5b1da"
    "96ea4334e60007704f327e7b5c0156b0"
)
HARDWARE_ID = "linux-amd64-cuda12.6-sm70-16gb-v1"


class V07CudaNightlyRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        environments = json.loads(
            (ROOT / "environments/registry.json").read_text(encoding="utf-8")
        )
        self.environment = {
            item["id"]: item for item in environments["environments"]
        }[ENVIRONMENT_ID]

        profiles = json.loads(
            (ROOT / "configs/runtime_profiles.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.profile = {
            item["profile_id"]: item for item in profiles["profiles"]
        }[PROFILE_ID]

    def test_environment_binds_exact_wheel_image_and_sm70_floor(self) -> None:
        artifact = self.environment["runtime_artifact"]
        self.assertEqual(
            self.environment["docker"]["image"],
            IMAGE,
        )
        self.assertRegex(
            self.environment["docker"]["digest"],
            re.compile(r"^sha256:[0-9a-f]{64}$"),
        )
        self.assertEqual(
            self.environment["docker"]["digest_kind"],
            "local_image_id",
        )
        self.assertEqual(artifact["strategy"], "matched_nightly_wheel")
        self.assertEqual(artifact["artifact_kind"], "official_wheel")
        self.assertEqual(artifact["artifact_id"], WHEEL_ID)
        self.assertEqual(
            artifact["artifact_digest"],
            f"sha256:{WHEEL_SHA256}",
        )
        self.assertEqual(
            artifact["artifact_digest_kind"],
            "wheel_sha256",
        )
        self.assertEqual(
            artifact["torch_version"],
            "2.12.0.dev20260407+cu126",
        )
        self.assertEqual(artifact["python_abi"], "cp311-cp311")
        self.assertEqual(
            self.environment["packages"]["torch_git_version"],
            "6a13e444ee88996ff01cd2bab41d7f2857291646",
        )
        self.assertEqual(
            self.environment["hardware"]["min_compute_capability"],
            "7.0",
        )
        self.assertEqual(
            self.environment["source_loading_modes"],
            ["python_overlay"],
        )

    def test_dockerfile_verifies_the_official_wheel_before_install(self) -> None:
        dockerfile = (
            ROOT
            / "environments"
            / self.environment["docker"]["dockerfile"]
        ).read_text(encoding="utf-8")
        self.assertIn(WHEEL_SHA256, dockerfile)
        self.assertIn(WHEEL_ID, dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)
        self.assertIn("python -m pip install", dockerfile)
        self.assertIn(
            "FROM op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
            dockerfile,
        )

    def test_runtime_profile_and_legacy_projection_match_the_environment(
        self,
    ) -> None:
        image = self.profile["image"]
        hardware = self.profile["hardware"]
        self.assertEqual(self.profile["backend"], "remote_docker")
        self.assertEqual(self.profile["runtime_tier"], "cuda_python_overlay")
        self.assertEqual(self.profile["source_loading_mode"], "python_overlay")
        self.assertEqual(self.profile["platform"], "linux/amd64+cuda12.6")
        self.assertTrue(self.profile["requires_gpu"])
        self.assertEqual(image["identifier"], IMAGE)
        self.assertEqual(
            image["digest"],
            self.environment["docker"]["digest"],
        )
        self.assertEqual(image["digest_kind"], "image_id")
        self.assertEqual(hardware["identifier"], HARDWARE_ID)
        self.assertEqual(
            hardware["digest"],
            "sha256:" + hashlib.sha256(HARDWARE_ID.encode()).hexdigest(),
        )
        self.assertEqual(hardware["digest_kind"], "declared")
        self.assertEqual(
            legacy._PROFILE_BY_ENVIRONMENT[ENVIRONMENT_ID],
            PROFILE_ID,
        )


if __name__ == "__main__":
    unittest.main()
