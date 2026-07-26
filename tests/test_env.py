from src.rag import env


def test_ensure_env_loaded_only_calls_load_dotenv_once(monkeypatch):
    monkeypatch.setattr(env, "_ENV_LOADED", False)
    calls = []
    monkeypatch.setattr(env, "load_dotenv", lambda: calls.append(1))

    env.ensure_env_loaded()
    env.ensure_env_loaded()

    assert len(calls) == 1
    assert env._ENV_LOADED is True
