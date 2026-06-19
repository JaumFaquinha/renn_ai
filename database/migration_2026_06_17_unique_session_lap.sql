-- =============================================================================
-- migration_2026_06_17_unique_session_lap.sql
-- Defesa-em-profundidade contra voltas duplicadas dentro de uma mesma sessão.
-- =============================================================================
-- Execute no Supabase Studio → SQL Editor → New Query → Run.
-- Idempotente: CREATE UNIQUE INDEX IF NOT EXISTS é no-op se já existir.
--
-- Contexto: em 2026-06-10 duas sessões foram criadas a 11ms de distância (dois
-- run_session.py concorrentes) com voltas sobrepostas. A trava de instância
-- única em scripts/run_session.py previne a CAUSA. Este índice é a rede de
-- segurança no banco: impede que a MESMA sessão registre duas voltas com o
-- mesmo lap_number (ex.: um retry de upload reinserindo a volta após o INSERT
-- já ter sido aplicado no servidor).
--
-- Pré-requisito verificado em 2026-06-17: 0 violações de (session_id, lap_number)
-- entre as 990 voltas existentes — seguro de aplicar.
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS uq_laps_session_lap_number
    ON laps (session_id, lap_number);

-- Verificação: deve retornar 1 linha após aplicar.
SELECT indexname
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'laps'
  AND indexname = 'uq_laps_session_lap_number';
