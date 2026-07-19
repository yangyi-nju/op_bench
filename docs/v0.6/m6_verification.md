# OpBench v0.6 M6 Verification Record

Date: 2026-07-18 (Asia/Shanghai)

Branch: `codex/opbench-v0.6`

Commit under test: `86a7e04675c30867706831341703cd892c21c6f4` plus the M6 working tree. The milestone commit is created after this record and is reported in the handoff.

Current release status: **Completed** by the post-M7 exact-target closure in
section 8. The original M6 decision below is preserved as a historical freeze.

Decision: M6 implementation is complete and M7 is next. Deterministic runtime conformance and the required real-Codex local CPU paths passed. The one explicitly configured Remote target was unavailable with a stable connection timeout, so exact Runtime replay and Remote/CUDA canaries remain `Blocked`; they are not reported as successful executions.

## 1. Frozen identities

| Artifact | Identity |
| --- | --- |
| Runtime Profile registry file SHA-256 | `673ba5349912883953742de6de4851eece8b8bb48fdadd79b6cbf3b313b314ea` |
| Example RunManifest file SHA-256 | `1c213da264e508bea60271858daccb602e29c29da3c01184e2e6c3cbef42daf7` |
| Conformance report file SHA-256 | `0dec2ce66ccedf1e4a661a679f0d00c037fd044a0800f23ff72500ed4e77765d` |
| Replay inventory hash | `sha256:9a9d1e7795e8b9f1a182374f588791371b38fa833d9f08940650b589c5ae13a9` |
| Replay manifest file SHA-256 | `3be2c82202d89c09cf22f13412464753dcfaba4e276e6547bb5784eacfc4ab18` |
| Replay results file SHA-256 | `1c20682cc2d502a1182db62778b43c678506927f755c4dea8be7901340352ea5` |
| Real Codex single RunManifest file SHA-256 | `abe0dd323f1b3dcc689e1a65ce71984038b4f9ced9e4f36712affc6a14375fba` |
| Real Codex batch RunManifest file SHA-256 | `71f4a0438ad5b0bacdceec2c2c2b0c95688f67b65aac495ea1d0e28c02e80913` |
| Remote CPU blocked RunManifest file SHA-256 | `79d4051f34b7d1f0276cfff8eff3994977a2680323bec431468b2b081f0c61d7` |

The frozen Profile set is:

| Profile | Content hash |
| --- | --- |
| `local-cpu-process-v1` | `sha256:8a16099f3abfb35db078ba06e414e40f95030bcc0df91301f4e46aa707497738` |
| `remote-cpu-compile-pytorch-2.6-py311-v1` | `sha256:60196e63f93c5652f27518a467118e022d033dcd1ffb0bedb9ee8df790f489bd` |
| `remote-cpu-pytorch-2.6-py311-v1` | `sha256:ae43f0f7111b9e685bcea13c3e4c1f382e92282779155330cee13b7bcac54b7f` |
| `remote-cuda-kernel-pytorch-2.6-cu124-v1` | `sha256:427751235a1e90a9096c7d8b92ed427f6c2ef9d69d7babe681e23e02ca2a6b97` |
| `remote-cuda-overlay-pytorch-2.6-cu124-v1` | `sha256:def6b4bf11c8c555fc65be97485719a98b9b8822543a34fb06889ef2b1b43364` |

## 2. Deterministic conformance

Command:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_runtime_conformance.py \
  --fixture runs/v0.6_m6_conformance_fixture \
  --output-dir runs/v0.6_m6_conformance \
  --profile-registry configs/runtime_profiles.v1.json \
  --profile-id local-cpu-process-v1
```

Exit code: `0`.

Artifact: `runs/v0.6_m6_conformance/runtime_conformance.json`.

All four entries passed and produced the same normalized snapshot hash `sha256:0f03402d610022396c8203dd283db1e2535bb1931081937ff55f51e38fd17a17`:

- CLI + Local Process;
- MCP + Local Process;
- CLI + Scripted Remote semantics;
- MCP + Scripted Remote semantics.

The sequence performs list, read, write, test, diff, and finish. Local `test_run` reads the file changed through the same Runtime lease workspace. Runtime child processes receive a minimal environment and do not inherit controller `PYTHONPATH` or secret variables.

## 3. Frozen 17 + 17 + 51 replay inventory

Command:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_legacy_replay.py \
  --repository-root . \
  --output-root runs/v0.6_m6_legacy_replay
```

Exit code: `0`.

Artifacts are under `runs/v0.6_m6_legacy_replay/replay/`. The manifest freezes exactly 85 unique, sorted, content-bound cases:

| Kind | Cases | Result |
| --- | ---: | --- |
| Baseline | 17 | `Blocked(exact_runtime_unavailable)` |
| Gold | 17 | `Blocked(exact_runtime_unavailable)` |
| Legacy final patch | 51 | `Blocked(exact_runtime_unavailable)` |

The exact-target replay path is implemented and tested through `RuntimeFreshEvaluationBackend`; a connection-level target failure is cached so the controller does not repeat the same unavailable operation 85 times. This offline invocation freezes the complete inventory without pretending that evaluation ran.

The four historical result roots remained byte-identical before and after replay:

| Historical file | SHA-256 |
| --- | --- |
| `runs/v0.5_codex_legacy_cpu/results.jsonl` | `66a91ce646e18f12be065aaac070bb6600f6ffc62ef221534210b8e00a74734d` |
| `runs/v0.5_codex_legacy_cpu/summary.json` | `d4a64795d92179c3636b35530afcd81a5e0ae33e70953c53e7d984625031a535` |
| `runs/v0.5_codex_legacy_cuda/results.jsonl` | `5b0beecd22e0d89dac5c64cc9836704e8edec191f103d93f7a5db9db447b6774` |
| `runs/v0.5_codex_legacy_cuda/summary.json` | `6e552d05bdc07576d6777dc0352059ec9dcdeb00d362dc6798f0a9ed1e43a8e6` |
| `runs/v0.5_precision_codex_cpu/results.jsonl` | `83ef408948d84835235c7f4898bb3015d52d1e20730285e5b215c35c2c03aaf3` |
| `runs/v0.5_precision_codex_cpu/summary.json` | `c1668a30af14ddd4a26d638f095ef31ac68150fba13e8ac12728b6466881bace` |
| `runs/v0.5_precision_codex_gpu/results.jsonl` | `89916e1e929e8d9f15f8edecada53add35c9676f2ba6a6345b2bffd5d2339b0f` |
| `runs/v0.5_precision_codex_gpu/summary.json` | `60ac9b157bc537eacee13fece0ff8aa97d8e557845e097d1a60b82bab343e729` |

## 4. Real Codex standard Adapter

### Single local CPU canary

Input fixture: `runs/v0.6_m6_local_codex_input/dataset/dataset.json`.

Output: `runs/v0.6_m6_codex_local_canary`.

The real `codex_canonical` process used only the generated JSON action client. Its public Action trajectory was:

```text
workspace_list → workspace_read → workspace_read → workspace_search →
test_run → workspace_apply_patch → test_run → vcs_diff → session_finish
```

The Attempt was valid, cleanup passed, all 14 Integrity checks passed, and resource ownership verification passed. The submitted patch only added a docstring, so Fresh Evaluation truthfully classified it as `f2p_failed`; this is an Adapter/runtime canary, not a claim that the task was resolved.

### Two-repeat resume batch

Output: `runs/v0.6_m6_codex_local_batch`.

The first invocation left one infrastructure-invalid retry and one valid timeout/no-patch Attempt. Re-running the exact command executed only the invalid logical Attempt as retry 2 and preserved the already valid retry bytes. A third identical invocation reported `ran=0, skipped=2`; the complete run-tree hash remained `e25fb60f189076989006dd17e7d8ae95549c382840ef15d5340fe61d0d405f45`.

Final summary:

- expected/observed/valid: `2/2/2`;
- infrastructure invalid selected results: `0`;
- retry records: `1`;
- Agent terminals: `finished=1`, `timeout=1`;
- outcomes: `f2p_failed=1`, `no_patch=1`;
- all 14 Integrity checks and both runtime resource checks passed.

The malformed first patch was rejected as the nonterminal Agent error `invalid_request`; it was not attributed as a platform failure after the regression fix.

## 5. Exact Remote target result

One direct attempt used only the single target from the private target configuration. No ping, port scan, host discovery, process list, container list, or target search was run.

The Remote CPU command for `pytorch__149693__lazylinear_init` and `remote-cpu-pytorch-2.6-py311-v1` exited `1` with `platform_error/session_platform_error`. The direct exact workspace-create operation consistently returned exit `255` and was sanitized as `connection_timeout`. Public evidence is under `runs/v0.6_m6_codex_cpu_canary`; it contains two append-only invalid retries, `remote_workspace=create_failed`, exact local workspace release, passing runtime ownership/cleanup checks, and 14 passing Integrity checks.

Because Remote CPU, CUDA Overlay, and CUDA Kernel use that same exact unavailable target, their M6 status is:

| Entry | Status |
| --- | --- |
| Remote CPU | `Blocked(connection_timeout)` |
| CUDA Overlay | `Blocked(connection_timeout)` |
| CUDA Kernel Build | `Blocked(connection_timeout)` |

No replacement host was searched for and the timeout was not repeatedly probed after the stable result.

## 6. Verification commands

Final deterministic results:

| Verification | Result |
| --- | --- |
| M6 focused suite | 84 passed, 0 failed, 0 skipped |
| All `test_runtime*.py` | 348 passed |
| Full `test_*.py` suite | 517 passed |
| Dataset validation | 17 verified tasks |
| Example RunManifest validation | valid |
| `python -m compileall -q src scripts tests` | exit 0 |
| Tracked JSON parsing | exit 0 |
| `git diff --check` | exit 0 |

The resource verifier was run separately on the single local canary, the two-repeat batch, and the Remote blocked run. Both `runtime_resource_ownership` and `runtime_cleanup` passed for all three.

## 7. Safety and review

- Runtime resources have deterministic Attempt/retry/Profile/type/ordinal identities and private exact-handle storage.
- Cleanup addresses only recorded handles. Docker start failure, missing executable, timeout, cleanup command exception, and Remote prepare failure all have fault-injection coverage.
- Local process groups are terminated by their recorded PID group only. There is no global kill, process enumeration, container enumeration, broad label filter, wildcard cleanup, host discovery, port scan, or network probe in the M6 production paths.
- Docker/Remote commands use exact names, paths, target, identity file, and resource labels. Docker networking is `none` for denied Profiles.
- Public artifacts exclude target values and raw handles. CLI validation errors do not print private target paths or values.
- Final review found no open P0/P1 issue. The remaining Remote/CUDA and replay blocks are environmental release evidence gaps, not hidden successes.

M6 does not publish a new Agent ranking and does not relabel v0.5 results as v0.6 results.

## 8. Post-M7 exact-target closure

On 2026-07-19 the same configured target recovered. The closing work did
not search for another host and did not run a network probe, host discovery,
port/service scan, process list, or container list. It invoked only the exact
target and paths already present in the private target binding.

Representative canaries:

| Entry | Artifact root | Result |
| --- | --- | --- |
| Remote CPU | `runs/v0.6_retry_remote_cpu_canary_fixed` | valid/finished/no_patch; 14 Integrity checks Passed; workspace, container, and remote workspace released |
| CUDA Overlay | `runs/v0.6_retry_cuda_overlay_canary` | valid/finished/no_patch; 14 Integrity checks Passed; workspace, container, and remote workspace released |
| CUDA Kernel Build | `runs/v0.6_retry_cuda_kernel_inplace_canary` | expected/observed `f2p_failed`; build exit 0; 1/1 Passed, zero differences |

The kernel path materializes the exact frozen root tree plus recursive Gitlink
submodule commits, copies an optional exact ccache seed into the Attempt-owned
workspace, and executes `setup.py build_ext --inplace` with the workspace on
`PYTHONPATH`. The seed remains read-only shared input; the copied cache,
container, and remote workspace are Attempt-owned cleanup targets.
The remote workspace leaf is created exclusively; a pre-existing leaf fails
closed before rsync/seed/Docker and is never claimed or cleaned by the Attempt.
Each recursive submodule archive is tree-verified against its exact Gitlink
commit, and controller Git (including the incremental rsync fingerprint) runs
under an isolated authority environment. Its fingerprint includes ignored as
well as non-ignored controller-created files. Revision/archive resolution,
conformance, local fresh evaluation, and workspace Git use that same isolation.
Ordinary Remote CPU/Overlay rsync
has no cache exclusion. A seeded inplace-build excludes only root `/.ccache/`,
and a frozen source-owned root `.ccache` fails closed before remote operations.

The final command was:

```bash
PYTHONPATH=src python3.12 scripts/run_legacy_replay.py \
  --repository-root . \
  --output-root runs/v0.6_retry_legacy_replay_exact_complete \
  --target-config configs/remote_hosts.json
```

It exited `0` with `total=85 passed=85 failed=0 blocked=0`. The manifest contains
17 Baseline, 17 Gold, and 51 Legacy cases; all Task authorities are verified.
Expected and observed outcomes match for all 85 cases (31 `f2p_failed`, 54
`resolved`), and `replay_differences.jsonl` is empty.

The exact replay observer creates each case's resource ledger, private lease
store, and cleanup report inside controller-private temporary scratch. Cleanup
is fail-closed: a backend cleanup failure becomes an evaluation failure. Those
per-case files are intentionally ephemeral under the M6 Task 5 persist-only
contract, which retains only the manifest, results, differences, and summary.
Therefore the four replay files are outcome-compatibility evidence, not direct
per-case cleanup artifacts. Persistent R-03/R-10/R-12 resource proof comes
from the v1 Remote CPU/CUDA Overlay canary run roots (all 14 Integrity checks,
including ownership and cleanup) plus backend failure-injection tests.
If the controller process dies after a replay remote leaf is created, that
temporary ledger can be lost; a later exclusive-leaf collision fails closed and
may currently become cached target-global unavailability. Persisted protected
per-case cleanup attestations and collision-specific classification are future
hardening. v0.6 does not infer ownership or run broad cleanup in that case.

| Closing artifact | SHA-256 |
| --- | --- |
| Replay inventory | `sha256:193ef08f68f50a50c67f22b41ca2a31043c78d6b2311d23f16c588a86b80daee` |
| `replay_manifest.json` | `21f85f547b5efde922616a44390b5c07814aaf59c8db27a9863a37a61ac2b424` |
| `replay_results.jsonl` | `3c14d5bd462a633b1c4b7b062d1447d6a575ed62244793d5b93154e43db8c9d1` |
| empty `replay_differences.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `replay_summary.json` | `1f5fa1515f2e93bbdec9a393e9fc07a3ccf4d121e6d33b175fdb7a1b09b03309` |

All eight historical v0.5 result/summary hashes listed in section 3 remained
unchanged. This closes R-05, R-06, R-07, R-08, and R-10 without changing the
historical 37/51 score or claiming a new v0.6 Agent result.
