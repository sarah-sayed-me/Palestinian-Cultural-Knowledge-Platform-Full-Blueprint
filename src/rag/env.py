"""
Shared .env loading.

Both db.py (DATABASE_URL) and embedder.py (HF_TOKEN, for HuggingFace Hub rate
limits) need .env loaded before they read os.environ. Centralized here instead
of each module rolling its own guard, so any new module that needs .env can't
forget it — and can't get the *ordering* wrong, which was the actual bug this
fixed: Embedder() used to run before the only load_dotenv() call in a given
script (buried in db.py's get_connection()), so HF_TOKEN was never loaded in
time for the HuggingFace Hub download it was meant to authenticate.
"""

from __future__ import annotations

from dotenv import load_dotenv

_ENV_LOADED = False


def ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True
