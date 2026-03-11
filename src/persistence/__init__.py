"""
src/persistence — Persistência cross-session via Supabase.

Módulos:
    supabase_client  — Singleton de conexão com o Supabase
    lap_uploader     — Upload assíncrono de voltas (fila daemon thread)
    query_service    — Consultas históricas (personal best, tendências)
"""

from src.persistence.supabase_client import SupabaseClient
from src.persistence.lap_uploader import LapUploader
from src.persistence.query_service import QueryService

__all__ = ["SupabaseClient", "LapUploader", "QueryService"]
