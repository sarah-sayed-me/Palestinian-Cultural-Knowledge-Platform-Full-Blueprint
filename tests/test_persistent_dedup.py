"""
Integration tests for PersistentDuplicationIndex against a real Postgres
instance (docker-compose.yml). Skipped automatically if Postgres isn't
reachable, matching tests/test_rag_integration.py's pattern.
"""

from __future__ import annotations

import pytest

from src.ingestion.deduplication import DuplicationIndex, PersistentDuplicationIndex
from src.rag.db import get_connection


def _postgres_available() -> bool:
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable — run `docker compose up -d`"
)

TEST_TABLE = "ingestion_dedup_index_pytest"


@pytest.fixture
def conn():
    connection = get_connection()
    PersistentDuplicationIndex.TABLE = TEST_TABLE  # isolate from real ingestion data
    yield connection
    with connection.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TEST_TABLE};")
    connection.commit()
    connection.close()


def test_persists_and_reloads_signatures_across_instances(conn):
    text = "فلسطين ثقافة وتاريخ وتراث غني ومتنوع عبر العصور المختلفة." * 5

    first_index = PersistentDuplicationIndex(conn=conn)
    result1 = first_index.check_and_register("doc-1", text, source_id="wikipedia-ar")
    assert not result1.is_duplicate

    # A brand-new instance, same connection — simulates a fresh pipeline run.
    second_index = PersistentDuplicationIndex(conn=conn)
    result2 = second_index.check_and_register("doc-2", text, source_id="semantic-scholar")

    assert result2.is_duplicate  # same text, different run/source — still caught
    assert result2.canonical_id == "doc-1"


def test_distinct_text_is_not_flagged_as_duplicate_across_instances(conn):
    PersistentDuplicationIndex(conn=conn).check_and_register(
        "doc-a", "فلسطين ثقافة وتاريخ." * 10, source_id="wikipedia-ar"
    )

    second_index = PersistentDuplicationIndex(conn=conn)
    result = second_index.check_and_register(
        "doc-b", "نص مختلف تماما عن الوثيقة الاولى بكل المقاييس الممكنة." * 10, source_id="wikipedia-ar"
    )

    assert not result.is_duplicate


def test_without_a_connection_behaves_like_plain_duplication_index():
    index = PersistentDuplicationIndex(conn=None)

    result = index.check_and_register("doc-1", "بعض النصوص هنا" * 10)

    assert not result.is_duplicate
    assert index.stats()["persistent"] is False


def test_stats_reports_persistent_flag(conn):
    index = PersistentDuplicationIndex(conn=conn)

    assert index.stats()["persistent"] is True


def test_plain_duplication_index_unaffected_by_persistent_subclass():
    # Regression guard: PersistentDuplicationIndex must not change the base
    # class's own in-memory-only behavior for existing callers.
    index = DuplicationIndex()

    result = index.check_and_register("doc-1", "نص عربي بسيط للاختبار" * 10)

    assert not result.is_duplicate
    assert "persistent" not in index.stats()
