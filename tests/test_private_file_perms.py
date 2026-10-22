"""Files that carry evidence text are owner-only, whatever produced them.

Two discovery artifacts and the research database were written with default
permissions while every comparable surface (last-report.json, last-run.json,
corpus-cache.json, the .env) is 0600/0700. The nominations bundle and pending
report both embed full ``SourceItem`` bodies -- including private corpus text
when a corpus is configured -- and SECURITY.md documents research.db as 0600
even though it was only hardened on the private-corpus path.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from lib import discovery_handoff, private_files


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_nominations_bundle_is_written_owner_only(tmp_path):
    state_dir = tmp_path / "state"
    discovery_handoff.write_nominations_bundle(
        [],
        domain="test",
        tier="deep",
        from_date="2026-02-01",
        to_date="2026-03-01",
        lookback_days=30,
        enrichment_source_boundary=None,
        requested_sources=["reddit"],
        save_dir=state_dir,
    )
    bundle = discovery_handoff.nominations_bundle_path(state_dir)
    assert _mode(bundle) == 0o600


def test_private_write_helper_sets_the_mode_before_writing(tmp_path):
    target = tmp_path / "evidence.json"
    private_files.write_private_text(target, '{"a": 1}')
    assert _mode(target) == 0o600
    assert target.read_text() == '{"a": 1}'


def test_persist_report_hardens_the_database_without_a_corpus(tmp_path, monkeypatch):
    """No corpus in this report, but the DB still holds every finding body."""
    from tests.test_judge_reason_defang import report_with, candidate
    import last30days

    db_path = tmp_path / "research.db"
    monkeypatch.chdir(tmp_path)
    last30days.persist_report(report_with([candidate("c1")]), store_db=db_path)

    for candidate_path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if candidate_path.exists():
            assert _mode(candidate_path) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes only")
def test_db_is_not_group_readable_after_init(tmp_path):
    import store as store_mod

    db_path = tmp_path / "research.db"
    with store_mod.scoped_db(db_path):
        store_mod.init_db()
        store_mod.ensure_private_db_files()
    assert _mode(db_path) == 0o600
