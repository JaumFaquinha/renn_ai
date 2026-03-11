"""
supabase_client.py — Singleton de conexão com o Supabase.

Responsabilidades:
    - Ler SUPABASE_URL, SUPABASE_KEY e SUPABASE_ENABLED do environment
    - Criar e expor o cliente supabase-py (thread-safe — o SDK é stateless)
    - Verificar conectividade antes do loop principal (health_check)
    - Retornar None silenciosamente quando SUPABASE_ENABLED=false

Uso:
    client = SupabaseClient()
    if client.is_enabled:
        client.get_client().table("sessions").insert({...}).execute()
"""

import logging
from typing import Optional

from config.settings import SUPABASE_ENABLED, SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Gerencia a conexão com o Supabase.

    Quando SUPABASE_ENABLED=false, todos os métodos retornam None/False
    sem lançar exceções — o projeto funciona identicamente ao estado pré-Fase 7.
    """

    def __init__(self) -> None:
        self._client = None
        self._enabled = SUPABASE_ENABLED

        if not self._enabled:
            logger.debug("Supabase desabilitado (SUPABASE_ENABLED=false)")
            return

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning(
                "SUPABASE_ENABLED=true mas SUPABASE_URL ou SUPABASE_KEY não configurados. "
                "Desabilitando integração."
            )
            self._enabled = False
            return

        try:
            from supabase import create_client  # import lazy — não falha se não instalado
            self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client inicializado", extra={"url": SUPABASE_URL})
        except Exception as exc:
            logger.warning(
                "Falha ao inicializar Supabase client — integração desabilitada",
                extra={"error": str(exc)},
            )
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """True se a integração está ativa e o cliente foi inicializado."""
        return self._enabled and self._client is not None

    def get_client(self):
        """
        Retorna o cliente Supabase ou None se desabilitado.

        Returns:
            supabase.Client | None
        """
        return self._client

    def health_check(self) -> bool:
        """
        Verifica conectividade com o Supabase fazendo uma query leve.

        Returns:
            True se conectado e respondendo, False caso contrário.
        """
        if not self.is_enabled:
            return False

        try:
            # Query mínima: conta sessões (pode retornar 0, só testa a conexão)
            self._client.table("sessions").select("id", count="exact").limit(1).execute()
            logger.info("Supabase health check: OK")
            return True
        except Exception as exc:
            logger.warning("Supabase health check falhou", extra={"error": str(exc)})
            return False
