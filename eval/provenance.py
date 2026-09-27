"""Provenance of an eval run: which code, which libraries, when.

A snapshot in `eval/reference/` is only evidence if it can be tied back to the code
that produced it — results from before a retriever fix look exactly like results from
after it. Every eval script takes `start = run_metadata()` *before* loading or indexing
anything, and stores `config["provenance"] = finish_metadata(start)`: the code that
produced the numbers is the code at the start (a multi-hour run keeps what it imported),
and the end state only serves to flag a checkout that moved during the run.

Light by design (stdlib only): the library versions are read from the installed
distributions' metadata, not by importing them (importing torch alone takes seconds).
"""

import hashlib
import platform
import subprocess
import sys
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
    """Output of a git command run at the repo root, or None (no git, not a checkout).

    Only trailing whitespace is stripped: `status --porcelain` lines start with a
    two-column status that may be a space (" M file").
    """
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.rstrip()


def _dirty_files(porcelain: str) -> list[str]:
    """Paths listed by `git status --porcelain` ("XY path" lines, XY = 2 status columns)."""
    return [line[3:] for line in porcelain.splitlines() if line.strip()]


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
    """{git_sha, git_dirty, dirty_files, diff_sha, timestamp, python, platform, packages}.

    `git_sha`/`git_dirty` are None outside a git checkout (e.g. a Docker image);
    `git_dirty` means code or inputs differ from `git_sha`, so the numbers cannot be
    reproduced from the commit alone (`dirty_files` names the culprits, capped at 20).
    `diff_sha` fingerprints the uncommitted changes to tracked files (None when there
    are none): two runs with equal `git_sha` and `diff_sha` ran the same tracked code.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".", *_OUTPUT_PATHS) if sha else None
    dirty_files = _dirty_files(status) if status else []
    diff = _git("diff", "HEAD", "--", ".", *_OUTPUT_PATHS) if dirty_files else None
    return {
        "git_sha": sha or None,
        "git_dirty": None if status is None else bool(dirty_files),
        "dirty_files": dirty_files[:20],
        "diff_sha": hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else None,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "packages": _package_versions(),
    }


def finish_metadata(start: dict) -> dict:
    """`start` (from `run_metadata()` before the run) + when it ended, flagging code drift.

    Provenance taken when the payload is written would describe the checkout at the
    *end* of a possibly multi-hour run — after a commit or checkout in that tree it
    names code that did not produce the numbers. So the start state is kept, and the
    end state is compared with it: `code_changed_during_run` is True (with the end
    sha / dirty files, and a warning) when HEAD or the uncommitted changes moved.
    """
    end = run_metadata()
    fields = ("git_sha", "dirty_files", "diff_sha")
    changed = (None if start.get("git_sha") is None
               else any(start.get(f) != end[f] for f in fields))
    meta = start | {"finished": end["timestamp"], "code_changed_during_run": changed}
    if changed:
        meta |= {"git_sha_at_end": end["git_sha"], "dirty_files_at_end": end["dirty_files"]}
        print(f"⚠️  the checkout changed during the run ({start['git_sha'][:7]} -> "
              f"{(end['git_sha'] or '?')[:7]}, dirty files {end['dirty_files']}): the "
              "recorded provenance is the state at the start.", file=sys.stderr)
    return meta
