# OpBench v0.6 M7 Verification

Date: 2026-07-18 (Asia/Shanghai)

Base platform commit: `00f2f30c69307caed284b8c8defc2d2dff3cab62`

Scope: public Demo preparation, bilingual Quickstart, developer/support docs,
representative public Artifact, compatibility wording, and release review.

## 1. Synthetic v1 Demo

Input preparation:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_v0_6_demo.py \
  --output-dir runs/v0.6_m7_demo_input
```

The generated source Git identity is deterministic across independent output
directories. The generated one-task verified Dataset is
`opbench_v0.6_scripted_demo`; the Task is `opbench_demo__normalize`. Preparation
performs no network operation and refuses a non-empty output without deleting
it. The fixed source commit is `294ae026d211a13282202827206a8a66ead3a542`.

Standard v1 execution:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m7_demo_input/dataset/dataset.json \
  --verified-only \
  --agent scripted_canonical \
  --agent-repeat 1 \
  --output-dir runs/v0.6_m7_scripted_demo \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1
```

First invocation: exit `0`, `ran=1`, `skipped=0`.

Second identical invocation: exit `0`, `ran=0`, `skipped=1`.

The complete selected run-tree digest before and after resume was identical:
`9f832defe3f475e8831c44a1cf0a2554aaf384f5477a43c483ee04f459620fc3`.

Result axes: `valid / finished / no_patch`. Totals: expected 1, observed 1,
valid 1, infrastructure-invalid 0, resolved 0. This is the expected outcome for
the deterministic no-edit Adapter and is not a benchmark score.

Resource verification:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_runtime_resources.py \
  --run-root runs/v0.6_m7_scripted_demo
```

Exit `0`; `runtime_resource_ownership=passed`, `runtime_cleanup=passed`.

RunManifest validation:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  runs/v0.6_m7_scripted_demo/run_manifest.json
```

Exit `0`; valid `run_manifest`.

## 2. Representative public Artifact

Tracked file:
`configs/examples/v0.6_scripted_demo_artifact.example.json`.

Frozen identities:

- Cohort: `cohort:v1:24277c28399f44f92025b96900c9142c60034a67d4f9503aaec0c9ecde012f72`;
- Attempt: `attempt:v1:da4a7389ed6b94f7f85e42c230425fbec261376b2c894ce2dd577b00637d6773`;
- Runtime Profile hash:
  `sha256:8a16099f3abfb35db078ba06e414e40f95030bcc0df91301f4e46aa707497738`;
- RunManifest hash:
  `sha256:185d780e3257df46a1ba15046cf0eeb5f549255aaaf95509e7cd29efc70e6ad8`;
- results hash:
  `sha256:8fd5dfb168af31f00c88249c5697d29cc1946ebb94367c38953b09c6c9ff63bd`;
- summary hash:
  `sha256:48c765470f635bd0b3931c8aeb4b1b9a04fa9407234dd2e0ddb72eac69ff103c`;
- Integrity report hash:
  `sha256:c01020288e0608659c2b4bfbe689762cf35baafce45108c9cf7cb3f590d36afe`.

The index records `workspace_list → test_run → vcs_diff → session_finish` and
contains no raw run-root path, machine-local path, credential marker, private
evaluation bytes, or private runtime handle.

## 3. Tests and documentation review

Task 1 focused command:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest \
  tests.test_prepare_v0_6_demo -v
```

Result: 5/5 passed, 0 failures/errors/skips.

Task 2/3 documentation command:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest \
  tests.test_v0_6_release_docs -v
```

The initial RED run failed for the missing Quickstart, guide, and public
Artifact. After implementation, 5/5 link/command/support/wording and public
Artifact checks passed.

## 4. Final release freeze

Clean environment:

```bash
m7_clean_root=$(mktemp -d)
python3 -m venv "$m7_clean_root/venv"
PYTHONPATH=src "$m7_clean_root/venv/bin/python" -m unittest discover \
  -s tests -p 'test_*.py' -q
PYTHONPATH=src "$m7_clean_root/venv/bin/python" scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json --require-verified
```

The environment used Python 3.12.13 and installed no package. Result: 527/527
tests passed with 0 failures/errors/skips; Dataset validation passed 17 tasks.
The system owns and will reclaim the temporary venv.

Release-focused command:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest \
  tests.test_prepare_v0_6_demo \
  tests.test_v0_6_release_docs \
  tests.test_runtime_v1_cli \
  tests.test_runtime_resources_cli \
  tests.test_runtime_integrity -q
```

Result: 25/25 passed with 0 failures/errors/skips.

Additional checks, all exit `0`:

- the public example and executed Demo RunManifest both passed
  `validate_runtime_contract.py`;
- Demo `runtime_resource_ownership` and `runtime_cleanup` passed;
- 85 tracked-or-new versioned JSON files parsed;
- `compileall -q src scripts tests` passed;
- bilingual link, command, support, profile, Artifact, and non-claim tests passed;
- CLI `--help` matched the documented explicit v1 flags;
- `git diff --check` passed;
- changed production inputs contained no probe/discovery/enumeration command;
- public changes contained no machine-local path, credential, private target,
  raw run root, private evaluation bytes, or runtime handle;
- final self-review found no open P0/P1 implementation finding.

## 5. Remote evidence and release decision

M7 did not repeat the M6 stable Remote `connection_timeout`, did not probe the
network, and did not search for a replacement target. M6's exact target evidence
remains authoritative.

Open implementation P0/P1 found so far: none.

Blocked Must items: R-05, R-06, R-07, R-08, R-10.

Release decision: **Blocked**. Local M7 completion cannot override the unified
v0.6 rule that every Must item must be Passed.
