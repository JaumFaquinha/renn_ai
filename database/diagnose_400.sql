-- =============================================================================
-- diagnose_400.sql — Diagnóstico de erro 400 no Supabase
-- =============================================================================
-- Execute este arquivo no Supabase Studio → SQL Editor → New Query → Run
-- Ele verifica o estado real das tabelas e aponta o que está causando o 400.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Verificar quais tabelas existem
-- ---------------------------------------------------------------------------
SELECT
    table_name,
    CASE WHEN table_name IS NOT NULL THEN '✓ existe' END AS status
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('sessions', 'laps', 'mini_sectors', 'lap_patterns', 'personal_bests')
ORDER BY table_name;

-- ---------------------------------------------------------------------------
-- 2. Verificar se a coluna tyre_compound existe em laps
--    (adicionada via migration — pode não ter sido aplicada)
-- ---------------------------------------------------------------------------
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name   = 'laps'
ORDER BY ordinal_position;

-- ---------------------------------------------------------------------------
-- 3. Verificar o tipo de drs_active e drs_available em mini_sectors
--    (migração SMALLINT → INT pode não ter sido aplicada)
-- ---------------------------------------------------------------------------
SELECT
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name   = 'mini_sectors'
  AND column_name  IN ('drs_active', 'drs_available');

-- ---------------------------------------------------------------------------
-- 4. Verificar todos os campos de mini_sectors (comparar com schema.sql)
-- ---------------------------------------------------------------------------
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name   = 'mini_sectors'
ORDER BY ordinal_position;

-- ---------------------------------------------------------------------------
-- 5. Verificar RLS (Row Level Security) — pode bloquear mesmo service_role
-- ---------------------------------------------------------------------------
SELECT
    tablename,
    rowsecurity AS rls_enabled,
    CASE WHEN rowsecurity THEN 'RLS ativo' ELSE 'RLS inativo' END AS status
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('sessions', 'laps', 'mini_sectors', 'lap_patterns', 'personal_bests');

-- ---------------------------------------------------------------------------
-- 6. Verificar políticas RLS existentes
-- ---------------------------------------------------------------------------
SELECT
    tablename,
    policyname,
    cmd,
    qual
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
