"""Dataset builder utilities for creating draft tasks from GitHub PRs."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PR_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$"
)
ISSUE_REF_RE = re.compile(
    r"(?i)\b(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+"
    r"(?:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+))?#(?P<number>\d+)"
)

TEST_FILE_PATTERNS = (
    "/test/",
    "/tests/",
    "_test.py",
    "test_",
)

FRAMEWORK_DEFAULTS = {
    "pytorch": {
        "image": "ghcr.io/op-bench/pytorch-cpu:py310",
        "python_version": "3.10",
        "build_mode": "editable-python",
        "dependencies": ["pytest", "numpy"],
        "setup_commands": ["python -m pip install -r requirements.txt"],
        "test_commands": ["python -m pytest"],
    },
    "tensorflow": {
        "image": "ghcr.io/op-bench/tensorflow-cpu:py310",
        "python_version": "3.10",
        "build_mode": "editable-python",
        "dependencies": ["pytest", "numpy"],
        "setup_commands": ["python -m pip install -r requirements.txt"],
        "test_commands": ["python -m pytest"],
    },
    "default": {
        "image": "ghcr.io/op-bench/default-cpu:py310",
        "python_version": "3.10",
        "build_mode": "editable-python",
        "dependencies": ["pytest"],
        "setup_commands": ["python -m pip install -r requirements.txt"],
        "test_commands": ["python -m pytest"],
    },
}


class BuilderError(RuntimeError):
    """Raised when the builder cannot produce a coherent task draft."""


@dataclass(frozen=True)
class PullRequestRef:
    owner: str
    repo: str
    number: int

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.repo_slug}/pull/{self.number}"


@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int

    @property
    def repo_slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.repo_slug}/issues/{self.number}"


class DataSource(Protocol):
    """Abstraction for live GitHub reads or local fixtures."""

    def fetch_pull(self, ref: PullRequestRef) -> dict[str, Any]:
        ...

    def fetch_pull_files(self, ref: PullRequestRef) -> list[dict[str, Any]]:
        ...

    def fetch_issue(self, ref: IssueRef) -> dict[str, Any]:
        ...

    def fetch_patch(self, ref: PullRequestRef) -> str:
        ...


class GitHubAPISource:
    """Fetches metadata from the GitHub REST API."""

    def __init__(self, token: str | None = None, timeout_sec: int = 30) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._timeout_sec = timeout_sec

    def _request(self, url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
        headers = {
            "Accept": accept,
            "User-Agent": "op_bench-builder/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return urllib.request.Request(url, headers=headers)

    def _get_json(self, url: str) -> Any:
        request = self._request(url)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise BuilderError(f"GitHub API request failed for {url}: {exc.code} {message}") from exc
        except urllib.error.URLError as exc:
            raise BuilderError(f"GitHub API request failed for {url}: {exc.reason}") from exc

    def _get_text(self, url: str, accept: str = "application/vnd.github.patch") -> str:
        request = self._request(url, accept=accept)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise BuilderError(f"GitHub patch request failed for {url}: {exc.code} {message}") from exc
        except urllib.error.URLError as exc:
            raise BuilderError(f"GitHub patch request failed for {url}: {exc.reason}") from exc

    def fetch_pull(self, ref: PullRequestRef) -> dict[str, Any]:
        url = f"https://api.github.com/repos/{ref.repo_slug}/pulls/{ref.number}"
        return self._get_json(url)

    def fetch_pull_files(self, ref: PullRequestRef) -> list[dict[str, Any]]:
        url = f"https://api.github.com/repos/{ref.repo_slug}/pulls/{ref.number}/files?per_page=100"
        return self._get_json(url)

    def fetch_issue(self, ref: IssueRef) -> dict[str, Any]:
        url = f"https://api.github.com/repos/{ref.repo_slug}/issues/{ref.number}"
        return self._get_json(url)

    def fetch_patch(self, ref: PullRequestRef) -> str:
        return self._get_text(ref.html_url + ".patch")


class FixtureSource:
    """Reads builder inputs from a local fixture directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _load_json(self, name: str) -> Any:
        path = self._root / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def fetch_pull(self, ref: PullRequestRef) -> dict[str, Any]:
        return self._load_json("pull.json")

    def fetch_pull_files(self, ref: PullRequestRef) -> list[dict[str, Any]]:
        return self._load_json("files.json")

    def fetch_issue(self, ref: IssueRef) -> dict[str, Any]:
        return self._load_json("issue.json")

    def fetch_patch(self, ref: PullRequestRef) -> str:
        path = self._root / "gold.patch"
        return path.read_text(encoding="utf-8")


def parse_pr_url(url: str) -> PullRequestRef:
    match = PR_URL_RE.match(url.strip())
    if not match:
        raise BuilderError(f"unsupported PR URL: {url}")

    return PullRequestRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        number=int(match.group("number")),
    )


def parse_issue_url(url: str) -> IssueRef:
    match = re.match(
        r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)/?$",
        url.strip(),
    )
    if not match:
        raise BuilderError(f"unsupported issue URL: {url}")

    return IssueRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        number=int(match.group("number")),
    )


def extract_issue_ref(body: str, default_owner: str, default_repo: str) -> IssueRef | None:
    for match in ISSUE_REF_RE.finditer(body or ""):
        owner = match.group("owner") or default_owner
        repo = match.group("repo") or default_repo
        number = int(match.group("number"))
        return IssueRef(owner=owner, repo=repo, number=number)
    return None


def slugify(text: str, max_words: int = 6) -> str:
    pieces = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not pieces:
        return "task"
    return "_".join(pieces[:max_words])


def infer_framework(repo_slug: str) -> str:
    repo_slug_lower = repo_slug.lower()
    if "pytorch" in repo_slug_lower or repo_slug_lower.endswith("/torch"):
        return "pytorch"
    if "tensorflow" in repo_slug_lower:
        return "tensorflow"
    return "unknown"


def infer_component(files: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for file_entry in files:
        filename = file_entry.get("filename", "")
        if is_test_file(filename):
            continue
        parts = filename.split("/")
        if len(parts) >= 2:
            candidates.append("/".join(parts[:2]))
        elif parts:
            candidates.append(parts[0])
    return most_common(candidates) or "unknown"


def infer_operator_name(title: str, body: str, files: list[dict[str, Any]]) -> str:
    for source_text in (title, body):
        for pattern in (r"`([^`]+)`", r"'([^']+)'", r'"([^"]+)"'):
            match = re.search(pattern, source_text or "")
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate

    for file_entry in files:
        filename = file_entry.get("filename", "")
        if filename.endswith(".py"):
            stem = Path(filename).stem
            if stem not in {"test", "tests", "__init__"}:
                return stem

    return "unknown_operator"


def infer_problem_type(title: str, body: str, files: list[dict[str, Any]]) -> str:
    haystack = " ".join([title or "", body or "", " ".join(f.get("filename", "") for f in files)]).lower()
    keyword_map = {
        "dtype": "dtype-handling",
        "broadcast": "shape-handling",
        "shape": "shape-handling",
        "nan": "numerical-semantics",
        "inf": "numerical-semantics",
        "grad": "gradient",
        "cuda": "device-parity",
        "cpu": "device-parity",
        "error": "error-path",
        "exception": "error-path",
    }
    for keyword, label in keyword_map.items():
        if keyword in haystack:
            return label
    return "operator-behavior"


def most_common(items: list[str]) -> str | None:
    if not items:
        return None
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)


def is_test_file(path: str) -> bool:
    lowered = path.lower()
    return any(pattern in lowered for pattern in TEST_FILE_PATTERNS)


def candidate_test_files(files: list[dict[str, Any]]) -> list[str]:
    candidates = [entry.get("filename", "") for entry in files if is_test_file(entry.get("filename", ""))]
    return sorted(set(filter(None, candidates)))


def changed_source_files(files: list[dict[str, Any]]) -> list[str]:
    candidates = [entry.get("filename", "") for entry in files if entry.get("filename") and not is_test_file(entry["filename"])]
    return sorted(set(candidates))


def framework_defaults(framework: str) -> dict[str, Any]:
    return FRAMEWORK_DEFAULTS.get(framework, FRAMEWORK_DEFAULTS["default"])


def build_task_id(repo: str, issue_number: int, issue_title: str) -> str:
    repo_name = repo.split("/")[-1]
    return f"{repo_name}__issue_{issue_number}__{slugify(issue_title)}"


def draft_fail_to_pass_tests(test_files: list[str]) -> list[str]:
    if test_files:
        return [f"TODO: identify fail-to-pass tests from {test_files[0]}"]
    return ["TODO: identify fail-to-pass tests from the PR test diff"]


def draft_pass_to_pass_tests(test_files: list[str]) -> list[str]:
    if len(test_files) >= 2:
        return [f"TODO: identify regression coverage from {test_files[1]}"]
    if test_files:
        return [f"TODO: identify regression coverage from {test_files[0]}"]
    return ["TODO: select regression tests that should remain passing"]


def build_manifest(
    *,
    ref: PullRequestRef,
    issue_ref: IssueRef,
    pull: dict[str, Any],
    issue: dict[str, Any],
    files: list[dict[str, Any]],
    patch_text: str,
    tier: str,
    framework_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_slug = ref.repo_slug
    framework = framework_override or infer_framework(repo_slug)
    defaults = framework_defaults(framework)
    test_files = candidate_test_files(files)
    source_files = changed_source_files(files)
    operator_name = infer_operator_name(pull.get("title", ""), issue.get("body", ""), files)
    component = infer_component(files)
    problem_type = infer_problem_type(pull.get("title", ""), issue.get("body", ""), files)
    issue_title = issue.get("title") or pull.get("title") or f"issue {issue_ref.number}"
    task_id = build_task_id(repo_slug, issue_ref.number, issue_title)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    manifest = {
        "task_id": task_id,
        "version": "v1",
        "source": {
            "pr_url": pull.get("html_url", ref.html_url),
            "issue_url": issue.get("html_url", issue_ref.html_url),
            "repo": repo_slug,
            "issue_number": issue_ref.number,
            "pr_number": ref.number,
            "base_commit": pull["base"]["sha"],
            "merge_commit": pull.get("merge_commit_sha") or pull["head"]["sha"],
        },
        "statement": {
            "title": issue_title,
            "body": issue.get("body") or "",
            "labels": [label["name"] for label in issue.get("labels", []) if "name" in label],
        },
        "operator": {
            "framework": framework,
            "component": component,
            "operator_name": operator_name,
            "problem_type": problem_type,
            "tags": sorted(
                {
                    tag
                    for tag in [
                        framework,
                        problem_type,
                        *[label["name"] for label in issue.get("labels", []) if "name" in label],
                    ]
                    if tag
                }
            ),
        },
        "environment": {
            "tier": tier,
            "image": defaults["image"],
            "python_version": defaults["python_version"],
            "os": "ubuntu22.04",
            "build_mode": defaults["build_mode"],
            "hardware": {
                "device": "cpu" if tier == "cpu-deterministic" else "gpu",
                "min_memory_gb": 8 if tier == "cpu-deterministic" else 16,
            },
            "dependencies": defaults["dependencies"],
        },
        "agent_visible": {
            "repo_setup_commands": defaults["setup_commands"],
            "known_constraints": [
                f"Task was drafted automatically from {pull.get('html_url', ref.html_url)}.",
                "Review environment assumptions before admitting this task into the benchmark.",
            ],
            "allowed_test_commands": defaults["test_commands"],
        },
        "evaluation": {
            "setup_commands": defaults["setup_commands"],
            "fail_to_pass": draft_fail_to_pass_tests(test_files),
            "pass_to_pass": draft_pass_to_pass_tests(test_files),
            "test_command": defaults["test_commands"][0],
            "timeout_sec": 1800,
        },
        "artifacts": {
            "gold_patch": "artifacts/gold.patch",
            "test_patch": "artifacts/test.patch",
        },
        "metadata": {
            "difficulty": "easy",
            "curation_status": "draft",
            "deterministic": True,
            "estimated_runtime_min": 15,
            "notes": (
                f"Auto-generated from PR #{ref.number} on {now}. "
                "Review candidate tests, environment card, and determinism before benchmarking."
            ),
        },
    }

    extra_context = {
        "generated_at": now,
        "patch_line_count": len(patch_text.splitlines()),
        "candidate_test_files": test_files,
        "changed_source_files": source_files,
        "issue_author": issue.get("user", {}).get("login"),
        "pr_author": pull.get("user", {}).get("login"),
    }
    return manifest, extra_context


def render_issue_markdown(issue: dict[str, Any], pull: dict[str, Any], files: list[dict[str, Any]]) -> str:
    labels = ", ".join(label["name"] for label in issue.get("labels", []) if "name" in label) or "none"
    changed_files = "\n".join(f"- {entry.get('filename', '')}" for entry in files) or "- none"
    body = issue.get("body") or "_No issue body provided._"
    return textwrap.dedent(
        f"""\
        # {issue.get('title', pull.get('title', 'Untitled issue'))}

        - Issue URL: {issue.get('html_url', 'unknown')}
        - PR URL: {pull.get('html_url', 'unknown')}
        - Labels: {labels}

        ## Issue Body

        {body}

        ## Changed Files In PR

        {changed_files}
        """
    )


def render_review_checklist(manifest: dict[str, Any], extra: dict[str, Any]) -> str:
    candidate_tests = extra["candidate_test_files"] or ["none detected automatically"]
    changed_sources = extra["changed_source_files"] or ["none detected automatically"]
    checklist = "\n".join(f"- [ ] {item}" for item in candidate_tests)
    sources = "\n".join(f"- {item}" for item in changed_sources)
    return textwrap.dedent(
        f"""\
        # Review Checklist For {manifest['task_id']}

        This task was generated automatically from a PR and still needs human verification.

        ## Candidate Test Files

        {checklist}

        ## Changed Non-Test Files

        {sources}

        ## Required Human Checks

        - [ ] Confirm the linked issue is the right problem statement for this PR.
        - [ ] Verify the issue is reproducible on `source.base_commit`.
        - [ ] Replace placeholder `fail_to_pass` entries with real test names.
        - [ ] Replace placeholder `pass_to_pass` entries with real regression tests.
        - [ ] Confirm the environment image and build mode are realistic.
        - [ ] Split the PR patch into evaluator-only artifacts if hidden tests are needed.
        - [ ] Mark `metadata.curation_status` as `verified` only after replay succeeds.
        """
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def materialize_task_bundle(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    issue_markdown: str,
    review_checklist: str,
    pull: dict[str, Any],
    issue: dict[str, Any],
    files: list[dict[str, Any]],
    patch_text: str,
) -> Path:
    task_dir = output_root / manifest["task_id"]
    raw_dir = task_dir / "raw"
    artifacts_dir = task_dir / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    write_json(task_dir / "task.json", manifest)
    (task_dir / "issue.md").write_text(issue_markdown, encoding="utf-8")
    (task_dir / "REVIEW.md").write_text(review_checklist, encoding="utf-8")
    write_json(raw_dir / "pull.json", pull)
    write_json(raw_dir / "issue.json", issue)
    write_json(raw_dir / "files.json", files)
    (artifacts_dir / "gold.patch").write_text(patch_text, encoding="utf-8")
    (artifacts_dir / "test.patch").write_text(
        "TODO: extract hidden evaluation test patch from the merged PR.\n",
        encoding="utf-8",
    )
    return task_dir


def build_task_from_pr(
    *,
    pr_url: str,
    output_root: Path,
    source: DataSource,
    issue_url: str | None = None,
    tier: str = "cpu-deterministic",
    framework_override: str | None = None,
) -> Path:
    ref = parse_pr_url(pr_url)
    pull = source.fetch_pull(ref)

    if issue_url:
        issue_ref = parse_issue_url(issue_url)
    else:
        issue_ref = extract_issue_ref(pull.get("body", ""), ref.owner, ref.repo)
        if issue_ref is None:
            raise BuilderError(
                "could not infer an issue link from the PR body; rerun with --issue-url"
            )

    issue = source.fetch_issue(issue_ref)
    files = source.fetch_pull_files(ref)
    patch_text = source.fetch_patch(ref)

    manifest, extra = build_manifest(
        ref=ref,
        issue_ref=issue_ref,
        pull=pull,
        issue=issue,
        files=files,
        patch_text=patch_text,
        tier=tier,
        framework_override=framework_override,
    )
    issue_markdown = render_issue_markdown(issue, pull, files)
    review_checklist = render_review_checklist(manifest, extra)

    return materialize_task_bundle(
        output_root=output_root,
        manifest=manifest,
        issue_markdown=issue_markdown,
        review_checklist=review_checklist,
        pull=pull,
        issue=issue,
        files=files,
        patch_text=patch_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an op_bench task draft from a GitHub PR URL.")
    parser.add_argument("pr_url", help="GitHub pull request URL")
    parser.add_argument(
        "--issue-url",
        help="Optional GitHub issue URL override when the PR body does not use closing keywords.",
    )
    parser.add_argument(
        "--output-dir",
        default="tasks/drafts",
        help="Directory where the generated task bundle will be written.",
    )
    parser.add_argument(
        "--tier",
        default="cpu-deterministic",
        choices=["cpu-deterministic", "single-gpu", "kernel-build"],
        help="Environment tier to stamp into the generated manifest.",
    )
    parser.add_argument(
        "--framework",
        help="Optional framework override, for example pytorch or tensorflow.",
    )
    parser.add_argument(
        "--fixture-dir",
        help="Use local fixture files instead of calling GitHub.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir).resolve()
    source: DataSource
    if args.fixture_dir:
        source = FixtureSource(Path(args.fixture_dir).resolve())
    else:
        source = GitHubAPISource()

    try:
        task_dir = build_task_from_pr(
            pr_url=args.pr_url,
            issue_url=args.issue_url,
            output_root=output_root,
            source=source,
            tier=args.tier,
            framework_override=args.framework,
        )
    except BuilderError as exc:
        parser.error(str(exc))
        return 2

    print(task_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
