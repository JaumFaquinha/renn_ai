"""
query_helpers.py — Utilitários para contornar limites do PostgREST/Supabase.

Dois limites do servidor causam bugs silenciosos nas consultas históricas:

  1. ``db-max-rows = 1000`` — qualquer SELECT sem paginação retorna no máximo
     1000 linhas. ``mini_sectors`` já tem ~100k linhas; uma volta tem ~100
     setores, então 10+ voltas já estouram o teto e a query trunca em silêncio.

  2. Comprimento de URL — filtros ``.in_("lap_id", [...])`` com centenas de
     UUIDs geram uma querystring de dezenas de KB. O servidor responde
     HTTP 400 ("Bad Request") e o supabase-py levanta ``APIError``, que era
     engolida pelos ``except Exception`` das funções de consulta — devolvendo
     ``[]`` como se não houvesse dados.

``fetch_all`` resolve (1) paginando via ``.range()``.
``fetch_all_in`` resolve (1) e (2): fatia a lista de ids em blocos e pagina
cada bloco.

Ambas recebem uma *factory* que devolve um query builder novo a cada chamada —
o builder do supabase-py não pode ser reexecutado após ``.execute()``.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Igual ao db-max-rows padrão do Supabase. Páginas deste tamanho minimizam
# o número de round-trips sem nunca serem truncadas pelo servidor.
PAGE_SIZE: int = 1000

# Número de ids por requisição em filtros .in_(). 100 UUIDs ≈ 3.7 KB de
# querystring — folgado abaixo de qualquer limite de URL (8 KB típico).
IN_CHUNK_SIZE: int = 100


def fetch_all(
    make_query: Callable[[], object],
    page_size: int = PAGE_SIZE,
) -> list[dict]:
    """
    Busca todas as linhas de uma query paginando via ``.range()``.

    Args:
        make_query: callable sem argumentos que devolve um query builder NOVO
            (com ``.select()``/``.eq()``/``.order()`` já aplicados). Precisa
            incluir um ``.order()`` determinístico para paginação correta.
        page_size: linhas por página (default: limite do servidor).

    Returns:
        Lista com todas as linhas, sem o teto de 1000 do PostgREST.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        res = make_query().range(offset, offset + page_size - 1).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_all_in(
    make_query: Callable[[list], object],
    ids: list,
    id_chunk: int = IN_CHUNK_SIZE,
    page_size: int = PAGE_SIZE,
) -> list[dict]:
    """
    Executa um filtro ``.in_(coluna, ids)`` com lista grande de ids sem estourar
    o limite de URL nem o teto de linhas do servidor.

    Fatia ``ids`` em blocos de ``id_chunk`` (resolve o 400 de URL longa) e pagina
    cada bloco via ``.range()`` (resolve o teto de 1000 linhas) — necessário
    porque um único bloco de 100 lap_ids pode render ~10k mini-setores.

    Args:
        make_query: callable que recebe uma sublista de ids e devolve um query
            builder NOVO com o ``.in_()`` aplicado e um ``.order()`` determinístico.
        ids: lista completa de ids para o filtro.
        id_chunk: ids por requisição.
        page_size: linhas por página dentro de cada bloco.

    Returns:
        Lista com todas as linhas de todos os blocos.
    """
    rows: list[dict] = []
    unique_ids = list(dict.fromkeys(ids))  # dedup preservando ordem
    for i in range(0, len(unique_ids), id_chunk):
        sub = unique_ids[i:i + id_chunk]
        offset = 0
        while True:
            res = make_query(sub).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
    return rows
