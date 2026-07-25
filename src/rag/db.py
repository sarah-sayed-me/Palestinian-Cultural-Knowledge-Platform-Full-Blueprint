"""
Postgres + pgvector connection helper.

Single place that knows how to connect to the vector store (docker-compose.yml)
and register the pgvector type adapter. Everything in src/rag/index.py and
src/rag/retriever.py goes through get_connection() rather than each opening
its own connection — one DATABASE_URL, one place to change it.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PGConnection

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def get_database_url() -> str:
    _ensure_env_loaded()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env (or add DATABASE_URL "
            "directly) and start the vector store with `docker compose up -d`."
        )
    return url


def get_connection(database_url: Optional[str] = None) -> PGConnection:
    """Open a psycopg2 connection with the pgvector type adapter registered.

    Raises a clear RuntimeError (not a raw psycopg2 traceback) if Postgres
    isn't reachable — most commonly because `docker compose up -d` hasn't
    been run yet.
    """
    url = database_url or get_database_url()
    try:
        conn = psycopg2.connect(url)
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            f"Could not connect to Postgres at the configured DATABASE_URL. "
            f"Is it running? Try `docker compose up -d`. Original error: {exc}"
        ) from exc
    # register_vector() requires the extension to already exist in this
    # database — create it here so get_connection() is self-sufficient
    # regardless of whether create_schema() has run yet in this process.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn
