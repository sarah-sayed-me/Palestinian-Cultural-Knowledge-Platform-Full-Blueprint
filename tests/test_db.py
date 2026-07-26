import pytest

from src.rag import db, env


def test_get_database_url_raises_clear_error_when_unset(monkeypatch):
    monkeypatch.setattr(env, "_ENV_LOADED", True)  # skip re-loading .env over our monkeypatch
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        db.get_database_url()


def test_get_connection_raises_clear_error_when_unreachable(monkeypatch):
    monkeypatch.setattr(env, "_ENV_LOADED", True)

    with pytest.raises(RuntimeError, match="Is it running"):
        db.get_connection(database_url="postgresql://nobody:nothing@localhost:1/nowhere")
