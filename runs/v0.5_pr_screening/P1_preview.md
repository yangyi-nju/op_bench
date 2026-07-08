# OpBench v0.5 P1 Preview

## Authenticated `gh` Retry Result

After the first preview was pushed, `gh` authentication became available and I reran the P1 verification.

Important correction: the four fallback candidates below are real PyTorch landed commits with PR references in main commit messages, but `gh pr view` reports those PRs as `CLOSED` with `mergedAt: null` and `mergeCommit: null`. They therefore do **not** satisfy the strict `gh pr list --state merged ... is:merged` requirement.

Authenticated commands run:

```bash
gh pr view 108559 --repo pytorch/pytorch --json title,state,closedAt,mergedAt,mergeCommit,baseRefOid,files
gh pr view 141052 --repo pytorch/pytorch --json title,state,closedAt,mergedAt,mergeCommit,baseRefOid,files
gh pr view 147203 --repo pytorch/pytorch --json title,state,closedAt,mergedAt,mergeCommit,baseRefOid,files
gh pr view 140557 --repo pytorch/pytorch --json title,state,closedAt,mergedAt,mergeCommit,baseRefOid,files
```

Observed for all four: `state: CLOSED`, `mergedAt: null`, `mergeCommit: null`.

I also reran P1-style `gh pr list --state merged` searches with the required JSON fields. Exact P1 keyword pack returned 0 results:

```bash
gh pr list --repo pytorch/pytorch --state merged --limit 300 \
  --search '(accumulator OR "Kahan" OR "logsumexp" OR "log_sum_exp" OR "sum precision") is:merged merged:2024-01-01..2025-04-30' \
  --json number,title,url,mergedAt,body,files
```

Expanded authenticated search across `bf16`, `fp16`, `sum`, `mean`, `norm`, `reduce`, `precision`, `accuracy`, and `correctness` found 24-27 unique true-GitHub-merged PRs depending on pack grouping. Hard filtering produced 3 to 6 pass-through PRs, but soft review rejected them for P1 because they were mix-precision copy, MPS/Inductor layout, quantization, performance/storage, or feature support issues rather than numerical accumulation errors.

Conclusion: with strict GitHub `is:merged` semantics, I did not find a compliant P1 pool. The candidates below are retained only as **directional fallback candidates** if OpBench decides that PyTorch ghstack/Phabricator "Pull Request resolved" commits are acceptable despite GitHub API `mergedAt` being null.

## Scope

Subclass: P1, numerical accumulation error.

Target pattern: low-precision reduction, norm, or accumulator path where fp16/bf16 intermediate values are truncated and a focused test can fail on the base commit and pass after the PR patch.

## Search Notes

Initial requested `gh pr list` / `gh pr view` commands could not be executed because this machine's `gh` CLI was not authenticated at the time:

```text
To get started with GitHub CLI, please run:  gh auth login
Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.
```

Fallback used for this preview:

- Pulled `origin/main` so `docs/v0.5/candidate_search.md` and `scripts/screen_candidates.py` are present.
- Cloned `pytorch/pytorch` with `git clone --filter=blob:none --no-checkout --single-branch --branch main --shallow-since=2023-12-01`.
- Searched merged main commits in the required window, using the same P1 keyword intent: `accumulator`, `sum precision`, `lower precision`, `fp16`, `bf16`, `reduction`, `mean`, `norm`, `logsumexp`.
- Built `/tmp/candidates_p1_raw.json` from real PyTorch merged commits and ran:

```bash
PYTHONPATH=src python3 scripts/screen_candidates.py \
  --input /tmp/candidates_p1_raw.json \
  --subclass P1 \
  --output /tmp/candidates_p1_screened.json
```

Hard filter result for the fallback commit-derived preview pool: 9 candidates checked, 7 passed, 2 rejected.

## Fallback Candidate Table

| # | PR | Subclass | Component | Patch Lines | Files | Bug Pattern |
|---:|---:|:---:|---|---:|---:|---|
| 1 | [#108559](https://github.com/pytorch/pytorch/pull/108559) | P1 | `aten/src/ATen/native/ReduceOps.cpp` | 77 | 2 | CPU low-precision `sum` over permuted/non-contiguous reductions truncates intermediate sums. |
| 2 | [#141052](https://github.com/pytorch/pytorch/pull/141052) | P1 | `torch/_inductor/codegen/triton.py` | 71 | 2 | Inductor/Triton fp16/bf16 math reductions accumulate without fp32 upcast. |
| 3 | [#147203](https://github.com/pytorch/pytorch/pull/147203) | P1 | `aten/src/ATen/native/layer_norm.cpp` | 36 | 3 | `rms_norm` downcasts before the final weight multiply, truncating fp16/bf16 computation. |
| 4 | [#140557](https://github.com/pytorch/pytorch/pull/140557) | P1 | `torch/_refs/__init__.py` | 30 | 2 | CPU `native_layer_norm` decomposition loses the CPU-specific low-precision upcast/return-dtype behavior under fake tensor dispatch. |

## Fallback Candidate Schema Preview

```json
[
  {
    "pr_url": "https://github.com/pytorch/pytorch/pull/108559",
    "issue_url": "https://github.com/pytorch/pytorch/issues/83149",
    "title": "Fix permuted sum precision issue for lower precision on CPU",
    "subclass": "P1",
    "problem_dimension": "precision",
    "component": "aten/src/ATen/native/ReduceOps.cpp / test/test_reductions.py",
    "files_changed": [
      "aten/src/ATen/native/ReduceOps.cpp",
      "test/test_reductions.py"
    ],
    "test_files_changed": [
      "test/test_reductions.py"
    ],
    "base_commit": "34e3f6f3c9ed0b8ebdbf37637722a6e53274198a",
    "merge_commit": "412c687e2e014c885ac3dcfb074619856f85eb9c",
    "patch_lines_source": 33,
    "patch_lines_test": 44,
    "requires_kernel_build": false,
    "min_gpu_arch": "cpu",
    "bug_pattern": "Permuted low-precision CPU sum can reduce across two dimensions inside TensorIterator and truncate intermediate values before the full fp32 reference sum is reached.",
    "why_good": "The added `test_sum_noncontig_lowp` constructs bf16/fp16 non-contiguous permuted tensors, compares `torch.sum` against an explicit float reference, and directly exercises the accumulator-buffer fix in `sum_out`."
  },
  {
    "pr_url": "https://github.com/pytorch/pytorch/pull/141052",
    "issue_url": null,
    "title": "[Inductor/Triton] Upcast FP16/BF16 math reductions to FP32",
    "subclass": "P1",
    "problem_dimension": "precision",
    "component": "torch/_inductor/codegen/triton.py / test/inductor/test_op_dtype_prop.py",
    "files_changed": [
      "test/inductor/test_op_dtype_prop.py",
      "torch/_inductor/codegen/triton.py"
    ],
    "test_files_changed": [
      "test/inductor/test_op_dtype_prop.py"
    ],
    "base_commit": "816328fa51382e9b50e60fb928a690d5c1bdadaf",
    "merge_commit": "417d9c3522dc6349b9d79c3822b846ad3c76386c",
    "patch_lines_source": 42,
    "patch_lines_test": 29,
    "requires_kernel_build": false,
    "min_gpu_arch": "sm_70",
    "bug_pattern": "Generated Triton reductions for fp16/bf16 math reductions keep low-precision accumulation instead of promoting reducer inputs to fp32.",
    "why_good": "The new `test_low_precision_reduction` compiles prod/sum/min/max/arg reductions on fp16/bf16 CUDA inputs and asserts the generated Triton code contains `.to(tl.float32)` while matching eager output."
  },
  {
    "pr_url": "https://github.com/pytorch/pytorch/pull/147203",
    "issue_url": "https://github.com/pytorch/pytorch/issues/134106",
    "title": "Fix rms_norm in fp16/bf16",
    "subclass": "P1",
    "problem_dimension": "precision",
    "component": "aten/src/ATen/native/layer_norm.cpp / test/test_nn.py",
    "files_changed": [
      "aten/src/ATen/native/layer_norm.cpp",
      "test/test_nn.py",
      "torch/testing/_internal/common_modules.py"
    ],
    "test_files_changed": [
      "test/test_nn.py",
      "torch/testing/_internal/common_modules.py"
    ],
    "base_commit": "85467ed063d284fa21a2f1d2adfec8fda544923d",
    "merge_commit": "8f71d4563eabbc978940c43b72fa67d770125e78",
    "patch_lines_source": 8,
    "patch_lines_test": 28,
    "requires_kernel_build": true,
    "min_gpu_arch": "cpu",
    "bug_pattern": "The RMSNorm path computes the reduction in opmath dtype but downcasts the normalized result before multiplying by the optional weight, losing fp16/bf16 accuracy.",
    "why_good": "The added `test_rmsnorm_numeric` compares `torch.nn.functional.rms_norm` against a reference that keeps the mean-of-squares reduction and weight multiply in float before the final cast."
  },
  {
    "pr_url": "https://github.com/pytorch/pytorch/pull/140557",
    "issue_url": null,
    "title": "fix layer_norm decomp precision for cpu",
    "subclass": "P1",
    "problem_dimension": "precision",
    "component": "torch/_refs/__init__.py / test/test_decomp.py",
    "files_changed": [
      "test/test_decomp.py",
      "torch/_refs/__init__.py"
    ],
    "test_files_changed": [
      "test/test_decomp.py"
    ],
    "base_commit": "240aa77ad01c4f0cd9b2417748272f2f617c112f",
    "merge_commit": "9ae19ffbedaa7754ee194b5da8d57cc2d2fee20e",
    "patch_lines_source": 5,
    "patch_lines_test": 25,
    "requires_kernel_build": false,
    "min_gpu_arch": "cpu",
    "bug_pattern": "The CPU `native_layer_norm` decomposition needs CPU-specific low-precision handling, but fake tensor/meta dispatch bypasses that branch and changes returned dtype behavior.",
    "why_good": "The added `test_native_layer_norm_cpu_decomp` compares real CPU bf16 native layer norm outputs against fake tensor decomposition outputs and fails if mean/rstd dtype semantics diverge."
  }
]
```

## Soft Review Rejects

| PR | Stage | Reason |
|---:|---|---|
| [#137529](https://github.com/pytorch/pytorch/pull/137529) | soft reject for P1 | Strong precision bug, but it modifies CUDA symmetric-memory `.cu`/`.h` kernel code and requires 4 GPUs with multicast support; better classified as P5 kernel precision, not P1 preview. |
| [#117345](https://github.com/pytorch/pytorch/pull/117345) | soft reject for P1 | Passes hard filter, but the fix is `kaiser_window` scalar formula stability/absolute-value handling, not a reduction accumulator error. |
| [#135174](https://github.com/pytorch/pytorch/pull/135174) | soft reject for P1 | Passes hard filter, but the failure is `mean(..., out=...)` not aliasing/updating the provided output tensor after low-precision temporary promotion; this fits P2 dtype/output semantics better than P1 accumulation. |
| [#134650](https://github.com/pytorch/pytorch/pull/134650) | hard reject | Fails rule 3: 5 files changed, over the 3-file limit. |
| [#142848](https://github.com/pytorch/pytorch/pull/142848) | hard reject | Fails rule 4 and rule 5 in the screened commit: 17 changed lines and no test file modified. |

## Screening Summary

- Raw fallback P1-related candidates checked: 9
- Passed hard filter: 7
- Rejected by hard filter: 2
- Rejected by soft review for P1: 3
- Final fallback P1 preview candidates: 4
- Final strict `gh is:merged` P1 candidates: 0

Important caveat: after authenticated retry, the listed fallback PRs were verified with `gh pr view` to exist, but they are GitHub `CLOSED` PRs with `mergedAt: null`. Their patches were verified against real PyTorch main commits whose commit messages contain the resolved PR URL, with file lists and patch lines taken from local Git history.
