-- =============================================================================
-- Engenheiro de Corrida IA — Limpeza completa de dados
-- =============================================================================
-- Remove TODOS os dados das tabelas, mantendo schema, índices e RLS intactos.
--
-- Como executar:
--   Supabase Studio → SQL Editor → colar este arquivo → Run
--
-- O que este script FAZ:
--   ✓ Remove todos os registros de mini_sectors, lap_patterns, laps,
--     personal_bests e sessions
--   ✓ Preserva tabelas, colunas, índices, foreign keys e políticas de RLS
--   ✓ Reseta sequences (UUIDs gerados pelo gen_random_uuid não são afetados)
--
-- O que este script NÃO faz:
--   ✗ Não remove tabelas ou colunas
--   ✗ Não altera tipos de dados
--   ✗ Não remove índices ou políticas de RLS
--
-- Gerado em: 2026-03-18
-- =============================================================================

-- TRUNCATE com CASCADE garante a ordem correta independente das FK constraints.
-- RESTART IDENTITY reseta qualquer sequence associada (não se aplica a UUID,
-- mas garante comportamento correto se sequences forem adicionadas futuramente).

TRUNCATE TABLE
    mini_sectors,
    lap_patterns,
    personal_bests,
    laps,
    sessions
RESTART IDENTITY CASCADE;

-- ---------------------------------------------------------------------------
-- Verificação pós-limpeza
-- ---------------------------------------------------------------------------
-- Execute as queries abaixo para confirmar que as tabelas estão vazias.

SELECT
    'sessions'      AS tabela, COUNT(*) AS registros FROM sessions
UNION ALL SELECT
    'laps',                    COUNT(*) FROM laps
UNION ALL SELECT
    'mini_sectors',            COUNT(*) FROM mini_sectors
UNION ALL SELECT
    'lap_patterns',            COUNT(*) FROM lap_patterns
UNION ALL SELECT
    'personal_bests',          COUNT(*) FROM personal_bests;

-- Resultado esperado: todas as linhas com registros = 0
