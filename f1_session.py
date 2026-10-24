import os
import sys
from pathlib import Path

import fastf1
from fastf1.exceptions import F1AuthError
from fastf1.internals.f1account import get_token, token_manager


class F1DataUnavailable(RuntimeError):
    pass


def ensure_f1_auth():
    token = token_manager.get_stored()
    if token is not None and token.is_valid():
        return
    has_env = bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))
    if not has_env and not sys.stdin.isatty():
        raise F1DataUnavailable(
            "No F1 TV login available. Set F1_USERNAME and F1_PASSWORD, or run "
            "'python -m fastf1 auth f1tv --authenticate' once on this machine."
        )
    try:
        get_token()
    except F1AuthError as e:
        raise F1DataUnavailable(f"F1 TV sign-in failed: {e}") from e


def load_session(year, circuit_name, session_type, cache_dir=".f1_cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=cache_dir)
    session = fastf1.get_session(year, circuit_name, session_type)
    session.load()
    if len(session.drivers) == 0 or session.laps.empty:
        raise F1DataUnavailable(
            f"{year} {circuit_name} {session_type} loaded with no drivers or laps. "
            "Either it is not published yet or the F1 TV login was rejected "
            "(check 'python -m fastf1 auth f1tv --status')."
        )
    return session
