# Builder Workflow

The PR builder is the first step in the dataset construction pipeline.

## What It Does

Given a GitHub pull request URL, the builder:

- parses the repository, PR number, and linked issue
- fetches PR metadata, issue metadata, changed files, and the merged patch
- infers a draft task id and basic operator metadata
- writes a draft task bundle for human review

## What It Produces

Each generated draft lives under `tasks/drafts/<task_id>/` and contains:

- `task.json`: draft manifest
- `issue.md`: agent-visible issue statement draft
- `REVIEW.md`: curation checklist
- `raw/`: raw GitHub API payloads
- `artifacts/gold.patch`: merged PR patch
- `artifacts/test.patch`: placeholder for later evaluator-only test extraction

## Important Limitation

The builder does not automatically prove that the task is benchmark-ready.

You still need to verify:

- the PR actually resolves the issue you want
- the issue reproduces on the base commit
- the environment assumptions are correct
- the fail-to-pass and pass-to-pass tests are real and deterministic

## CLI Usage

Live GitHub mode:

```bash
python3 scripts/build_task_from_pr.py \
  https://github.com/pytorch/pytorch/pull/123456 \
  --output-dir tasks/drafts
```

If the PR body does not contain a closing keyword such as `Fixes #123`, add the issue explicitly:

```bash
python3 scripts/build_task_from_pr.py \
  https://github.com/pytorch/pytorch/pull/123456 \
  --issue-url https://github.com/pytorch/pytorch/issues/123123
```

Offline fixture mode:

```bash
python3 scripts/build_task_from_pr.py \
  https://github.com/pytorch/pytorch/pull/999999 \
  --fixture-dir fixtures/pr_builder/sample_pytorch_pr \
  --output-dir /tmp/op_bench_drafts
```

## Recommended Human Review Loop

1. Generate a draft from the PR URL.
2. Review `REVIEW.md`.
3. Replace placeholder test names and environment assumptions.
4. Replay the task on the base commit.
5. Mark `metadata.curation_status` as `verified` only after replay succeeds.
