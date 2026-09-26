"""Light tests for the eval run provenance (stdlib only, no heavy import)."""

import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import provenance  # noqa: E402


def test_run_metadata_has_the_documented_fields():
    meta = provenance.run_metadata()
    assert {"git_sha", "git_dirty", "timestamp", "python", "packages"} <= set(meta)
    assert meta["python"].count(".") == 2
    assert isinstance(meta["packages"], dict)
    # Timestamp: ISO 8601, timezone-aware, in UTC.
    assert datetime.fromisoformat(meta["timestamp"]).utcoffset() == timedelta(0)


def test_git_fields_describe_this_checkout():
    meta = provenance.run_metadata()
    if meta["git_sha"] is None:  # e.g. a source tarball: nothing to check
        return
    assert re.fullmatch(r"[0-9a-f]{40}", meta["git_sha"])
    assert isinstance(meta["git_dirty"], bool)
    assert meta["git_dirty"] == bool(meta["dirty_files"])
    if not meta["git_dirty"]:
        assert meta["diff_sha"] is None  # nothing uncommitted to fingerprint


def test_without_git_the_fields_are_none(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    meta = provenance.run_metadata()
    assert meta["git_sha"] is None and meta["git_dirty"] is None
    assert meta["dirty_files"] == []


def test_missing_packages_are_omitted(monkeypatch):
    monkeypatch.setattr(provenance, "PACKAGES", ("surely-not-an-installed-dist-xyz",))
    assert provenance._package_versions() == {}


def test_dirty_files_keep_the_whole_path():
    # The first status column is often a space: it must not eat the path's first letter.
    porcelain = " M eval/beir_eval.py\nM  src/pipeline.py\n?? eval/new_tool.py"
    assert provenance._dirty_files(porcelain) == [
        "eval/beir_eval.py", "src/pipeline.py", "eval/new_tool.py"]


# --- finish_metadata: the start state is kept, drift is flagged -------------------

_START = {"git_sha": "a" * 40, "git_dirty": False, "dirty_files": [], "diff_sha": None,
          "timestamp": "2026-01-01T00:00:00+00:00", "python": "3.11.9", "packages": {}}


def test_finish_keeps_the_start_state_and_adds_the_end_time(monkeypatch):
    monkeypatch.setattr(provenance, "run_metadata",
                        lambda: _START | {"timestamp": "2026-01-01T03:00:00+00:00"})
    meta = provenance.finish_metadata(_START)
    assert meta["timestamp"] == _START["timestamp"]
    assert meta["finished"] == "2026-01-01T03:00:00+00:00"
    assert meta["code_changed_during_run"] is False
    assert "git_sha_at_end" not in meta


def test_finish_flags_a_checkout_that_moved_during_the_run(monkeypatch, capsys):
    end = _START | {"git_sha": "b" * 40}
    monkeypatch.setattr(provenance, "run_metadata", lambda: end)
    meta = provenance.finish_metadata(_START)
    assert meta["git_sha"] == "a" * 40  # the code that produced the numbers
    assert meta["code_changed_during_run"] is True
    assert meta["git_sha_at_end"] == "b" * 40
    assert "checkout changed during the run" in capsys.readouterr().err


def test_finish_flags_new_uncommitted_edits(monkeypatch):
    end = _START | {"git_dirty": True, "dirty_files": ["src/pipeline.py"], "diff_sha": "f00"}
    monkeypatch.setattr(provenance, "run_metadata", lambda: end)
    assert provenance.finish_metadata(_START)["code_changed_during_run"] is True


def test_finish_without_git_cannot_tell(monkeypatch):
    start = _START | {"git_sha": None, "git_dirty": None}
    monkeypatch.setattr(provenance, "run_metadata", lambda: start)
    assert provenance.finish_metadata(start)["code_changed_during_run"] is None
