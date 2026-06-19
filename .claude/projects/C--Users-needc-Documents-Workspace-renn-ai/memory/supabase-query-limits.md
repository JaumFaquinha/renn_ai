---
name: supabase-query-limits
description: Lap data lives in Supabase + local data/laps JSON; PostgREST has hard limits that silently truncate queries
metadata:
  type: project
---

Voltas vivem em DOIS lugares: Supabase (banco remoto, 874 voltas em monza após limpeza de 2026-06-17; eram 990) e arquivos JSON locais em `data/laps/`. Um clone novo (ex: o desktop) tem `data/laps/` VAZIO — por isso `train_model.py` carregava muito menos voltas no desktop (só Supabase) que no notebook (Supabase + JSON acumulado).

Em 2026-06-17 removi uma SESSÃO DUPLICADA (`8ce3573b`, dois `run_session.py` concorrentes em 2026-06-10 → 2 sessions a 11ms, 84 voltas sobrepostas). Prevenção adicionada: trava de instância única em `scripts/run_session.py` + índice `UNIQUE(session_id, lap_number)` (migration `database/migration_2026_06_17_unique_session_lap.sql` — rodar no Supabase Studio).

Supabase/PostgREST impõe dois limites de servidor que truncam consultas em silêncio:
- **`db-max-rows = 1000`**: qualquer SELECT sem paginação retorna no máx 1000 linhas. `mini_sectors` tem ~100k linhas.
- **Comprimento de URL**: `.in_("lap_id", [...])` com centenas de UUIDs → HTTP 400 "Bad Request", engolido por `except Exception` → retornava `[]`/`{}`.

**Why:** Explica "no desktop só vêm +100 voltas, no notebook +800" e "erros em funções que não deveriam". Causa primária extra: `load_laps_from_supabase` tinha `limit=100` hardcoded.

**How to apply:** Sempre paginar (`.range()`) e fatiar listas de ids em consultas Supabase — usar `src/persistence/query_helpers.py` (`fetch_all`, `fetch_all_in`). Transitivos do `supabase` (postgrest/httpx) NÃO estão pinados no requirements.txt — risco de drift entre máquinas.
