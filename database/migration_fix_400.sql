    -- =============================================================================
    -- migration_fix_400.sql — Correção do erro 400 no Supabase
    -- =============================================================================
    -- Execute no Supabase Studio → SQL Editor → New Query → Run
    -- Idempotente: seguro de executar múltiplas vezes sem risco de perda de dados.
    -- =============================================================================

    -- ---------------------------------------------------------------------------
    -- FIX 1: Coluna tyre_compound ausente na tabela laps
    -- Causa: migration adicionada após deploy inicial — pode não ter sido aplicada.
    -- Sintoma: erro 400 com mensagem "column tyre_compound does not exist"
    -- ---------------------------------------------------------------------------
    ALTER TABLE laps
        ADD COLUMN IF NOT EXISTS tyre_compound TEXT;

    -- ---------------------------------------------------------------------------
    -- FIX 2: drs_active / drs_available como SMALLINT (overflow)
    -- Causa: tipo original era SMALLINT (max 32.767).
    -- Misalinhamento de struct no AC pode retornar valores > 32.767,
    -- causando erro de overflow no INSERT de mini_sectors.
    -- Sintoma: erro 400 em bulk INSERT de mini_sectors com valores grandes.
    -- ---------------------------------------------------------------------------
    ALTER TABLE mini_sectors
        ALTER COLUMN drs_active    TYPE INT USING drs_active::INT,
        ALTER COLUMN drs_available TYPE INT USING drs_available::INT;

    -- ---------------------------------------------------------------------------
    -- FIX 3 (emergência): Schema nunca foi aplicado
    -- Se nenhuma das tabelas existir, execute o schema.sql completo primeiro:
    --   database/schema.sql → Supabase Studio → SQL Editor → Run
    -- Depois execute este arquivo para garantir que as migrations estão aplicadas.
    -- ---------------------------------------------------------------------------

    -- Verificação final: exibe estado atual das colunas críticas
    SELECT
        'laps.tyre_compound'           AS campo,
        COUNT(*)                       AS existe
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name   = 'laps'
    AND column_name  = 'tyre_compound'

    UNION ALL

    SELECT
        'mini_sectors.drs_active (INT)' AS campo,
        COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name   = 'mini_sectors'
    AND column_name  = 'drs_active'
    AND udt_name     = 'int4'   -- int4 = INTEGER no Postgres

    UNION ALL

    SELECT
        'mini_sectors.drs_available (INT)' AS campo,
        COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name   = 'mini_sectors'
    AND column_name  = 'drs_available'
    AND udt_name     = 'int4';

    -- Resultado esperado após aplicar este script:
    -- campo                              | existe
    -- -----------------------------------|-------
    -- laps.tyre_compound                 |   1
    -- mini_sectors.drs_active (INT)      |   1
    -- mini_sectors.drs_available (INT)   |   1
