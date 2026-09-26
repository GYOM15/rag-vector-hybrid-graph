"""Provenance of an eval run: which code, which libraries, when.

A snapshot in `eval/reference/` is only evidence if it can be tied back to the code
that produced it — results from before a retriever fix look exactly like results from
after it. Every eval script stores `config["provenance"] = run_metadata()`.

Light by design (stdlib only): the library versions are read from the installed
distributions' metadata, not by importing them (importing torch alone takes seconds).
"""

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Distributions whose version can change a retrieval score (index, embeddings, NER, BM25).
PACKAGES = ("faiss-cpu", "sentence-transformers", "torch", "transformers", "spacy",
            "en-core-web-sm", "rank-bm25", "numpy", "networkx", "datasets")

# Paths excluded from the "dirty" check: they hold eval *outputs* (snapshots, figures).
# Writing a fresh snapshot into eval/reference/ must not flag the next run as dirty
# code — only uncommitted changes to code or inputs should.
_OUTPUT_PATHS = (":(exclude)eval/reference", ":(exclude)docs")


def _git(*args: str) -> str | None:
    """Output of a git command run at the repo root, or None (no git, not a checkout)."""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_versions() -> dict[str, str]:
    """Installed version of each of `PACKAGES` (absent ones are simply omitted)."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_metadata() -> dict:
    """{git_sha, git_dirty, dirty_files, timestamp, python, platform, packages}.

    `git_sha`/`git_dirty` are None outside a git checkout (e.g. a Docker image);
    `git_dirty` means code or inputs differ from `git_sha`, so the numbers cannot be
    reproduced from the commit alone (`dirty_files` names the culprits, capped at 20).
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".", *_OUTPUT_PATHS) if sha else None
    dirty_files = [line[3:] for line in status.splitlines()] if status else []
    return {
        "git_sha": sha or None,
        "git_dirty": None if status is None else bool(dirty_files),
        "dirty_files": dirty_files[:20],
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": _package_versions(),
    }
