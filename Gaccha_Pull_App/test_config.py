"""Tests for the environment-variable helpers in config.py."""
import pytest

import config


class TestOpenAIKey:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
        assert config.get_openai_api_key() == "sk-live"

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            config.get_openai_api_key()


class TestModel:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        assert config.get_model() == "gpt-4o"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5-turbo")
        assert config.get_model() == "gpt-5-turbo"


class TestAllowedOrigins:
    def test_default_single_origin(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        assert config.get_allowed_origins() == ["http://localhost:5173"]

    def test_comma_separated_list_is_split_and_trimmed(self, monkeypatch):
        monkeypatch.setenv(
            "ALLOWED_ORIGINS",
            "https://a.vercel.app, https://b.com ,http://localhost:5173",
        )
        assert config.get_allowed_origins() == [
            "https://a.vercel.app",
            "https://b.com",
            "http://localhost:5173",
        ]
