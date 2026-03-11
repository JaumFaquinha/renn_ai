"""
Testes para src/persistence/supabase_client.py

Estratégia: mock completo do supabase.create_client — sem conexão real.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Garante que variáveis de ambiente não interferem entre testes
@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_USER_ID", raising=False)


class TestSupabaseClientDisabled:
    """Comportamento quando SUPABASE_ENABLED=false (default)."""

    def test_is_enabled_false_by_default(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_ENABLED", "false")
        # Recarrega settings para pegar o env atualizado
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        client = m.SupabaseClient()
        assert client.is_enabled is False

    def test_get_client_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_ENABLED", "false")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        client = m.SupabaseClient()
        assert client.get_client() is None

    def test_health_check_returns_false_when_disabled(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_ENABLED", "false")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        client = m.SupabaseClient()
        assert client.health_check() is False


class TestSupabaseClientMissingConfig:
    """Comportamento quando SUPABASE_ENABLED=true mas credenciais faltam."""

    def test_disabled_when_url_missing(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_ENABLED", "true")
        monkeypatch.setenv("SUPABASE_KEY", "some-key")
        # URL não definida
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        client = m.SupabaseClient()
        assert client.is_enabled is False

    def test_disabled_when_key_missing(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_ENABLED", "true")
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        # Força KEY vazia via setenv (evita load_dotenv() reler o .env real)
        monkeypatch.setenv("SUPABASE_KEY", "")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        client = m.SupabaseClient()
        assert client.is_enabled is False


class TestSupabaseClientEnabled:
    """Comportamento com configuração completa e mock do SDK."""

    def _make_client(self, monkeypatch, mock_create):
        monkeypatch.setenv("SUPABASE_ENABLED", "true")
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-service-role-key")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import src.persistence.supabase_client as m
        importlib.reload(m)
        with patch("supabase.create_client", mock_create):
            return m.SupabaseClient()

    def test_is_enabled_true_with_valid_config(self, monkeypatch):
        mock_sdk = MagicMock()
        mock_create = MagicMock(return_value=mock_sdk)
        client = self._make_client(monkeypatch, mock_create)
        assert client.is_enabled is True

    def test_get_client_returns_sdk_instance(self, monkeypatch):
        mock_sdk = MagicMock()
        mock_create = MagicMock(return_value=mock_sdk)
        client = self._make_client(monkeypatch, mock_create)
        assert client.get_client() is mock_sdk

    def test_health_check_success(self, monkeypatch):
        mock_sdk = MagicMock()
        # Simula chain: .table().select().limit().execute()
        mock_sdk.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock()
        mock_create = MagicMock(return_value=mock_sdk)
        client = self._make_client(monkeypatch, mock_create)
        assert client.health_check() is True

    def test_health_check_failure_returns_false(self, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("timeout")
        mock_create = MagicMock(return_value=mock_sdk)
        client = self._make_client(monkeypatch, mock_create)
        assert client.health_check() is False

    def test_create_client_exception_disables_integration(self, monkeypatch):
        mock_create = MagicMock(side_effect=Exception("connection refused"))
        client = self._make_client(monkeypatch, mock_create)
        assert client.is_enabled is False
