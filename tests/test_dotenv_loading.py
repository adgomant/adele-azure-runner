"""Tests for .env file loading semantics."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def test_dotenv_loads_env_file(tmp_path):
    """Values from .env are loaded into os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ADELE_KEY=from_dotenv\n")

    os.environ.pop("TEST_ADELE_KEY", None)
    try:
        load_dotenv(dotenv_path=str(env_file), override=False)
        assert os.environ.get("TEST_ADELE_KEY") == "from_dotenv"
    finally:
        os.environ.pop("TEST_ADELE_KEY", None)


def test_dotenv_local_overrides_env(tmp_path):
    """.env.local values take precedence over .env values."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ADELE_KEY=from_env\n")

    env_local = tmp_path / ".env.local"
    env_local.write_text("TEST_ADELE_KEY=from_local\n")

    os.environ.pop("TEST_ADELE_KEY", None)
    try:
        # Replicate the load order from cli.py:
        # .env.local first (override=False), then .env (override=False).
        # Since the key is set after the first call, the second call is a no-op.
        load_dotenv(dotenv_path=str(env_local), override=False)
        load_dotenv(dotenv_path=str(env_file), override=False)
        assert os.environ.get("TEST_ADELE_KEY") == "from_local"
    finally:
        os.environ.pop("TEST_ADELE_KEY", None)


def test_shell_env_not_overwritten_by_dotenv(tmp_path):
    """A real environment variable is NOT overwritten by .env (override=False)."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ADELE_KEY=from_dotenv\n")

    os.environ["TEST_ADELE_KEY"] = "real_value"
    try:
        load_dotenv(dotenv_path=str(env_file), override=False)
        assert os.environ.get("TEST_ADELE_KEY") == "real_value"
    finally:
        os.environ.pop("TEST_ADELE_KEY", None)


def test_missing_dotenv_file_is_noop(tmp_path):
    """load_dotenv on a nonexistent file does not raise."""
    nonexistent = tmp_path / ".env.nonexistent"
    # Should not raise
    load_dotenv(dotenv_path=str(nonexistent), override=False)
