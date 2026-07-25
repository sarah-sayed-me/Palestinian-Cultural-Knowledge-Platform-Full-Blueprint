"""
Dense retrieval over the pgvector index (ROADMAP.md Section 3.2/3.3).

Filter-then-retrieve is plain SQL: credibility/quality_score/seed_category are
denormalized onto every row (see src/rag/schemas.py::Chunk), so a credibility
floor is a WHERE clause, not a second filtering pass over retrieved results.

Credibility tiers compare correctly as plain strings ("tier_1" <= "tier_2" <=
"tier_3" <= "tier_4") because they share the same "tier_N" format with a single
digit — lexical order equals the intended best-to-worst order.
"""

from __future__ import annotations

from typing import List, Optional

from psycopg2.extensions import connection as PGConnection

from src.rag.config import RagConfig
from src.rag.embedder import Embedder
from src.rag.schemas import Chunk, RetrievedChunk

_SELECT_COLUMNS = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "text",
    "token_count",
    "start_char",
    "end_char",
    "chunking_version",
    "embedding_model",
    "embedding_version",
    "title",
    "source_url",
    "credibility",
    "quality_score",
    "seed_category",
)


class Retriever:
    def __init__(self, conn: PGConnection, embedder: Embedder, config: RagConfig):
        self.conn = conn
        self.embedder = embedder
        self.config = config

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_credibility_tier: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        top_k = top_k if top_k is not None else self.config.retrieval.top_k
        min_tier = (
            min_credibility_tier
            if min_credibility_tier is not None
            else self.config.retrieval.min_credibility_tier
        )
        query_vector = self.embedder.embed_query(query)
        table = self.config.vector_store.table
        columns_sql = ", ".join(_SELECT_COLUMNS)

        # Explicit ::vector casts: unlike an INSERT (where Postgres infers the
        # type from the target column), a bare parameter here has no column to
        # infer from, so psycopg2's array adapter sends it as numeric[] and
        # `vector <=> numeric[]` fails to resolve without the cast.
        sql = f"""
            SELECT {columns_sql}, 1 - (embedding <=> %(query_vector)s::vector) AS score
            FROM {table}
            WHERE %(min_tier)s IS NULL OR credibility <= %(min_tier)s
            ORDER BY embedding <=> %(query_vector)s::vector
            LIMIT %(top_k)s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, {"query_vector": query_vector, "min_tier": min_tier, "top_k": top_k})
            rows = cur.fetchall()
            col_names = [desc.name for desc in cur.description]

        results: List[RetrievedChunk] = []
        for row in rows:
            record = dict(zip(col_names, row))
            score = record.pop("score")
            chunk = Chunk.model_validate(record)
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return results
