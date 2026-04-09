-- =============================================================================
-- fix_corrupted_clutch_laps.sql
-- =============================================================================
-- Corrige o efeito do bug de desalinhamento do struct SPageFilePhysics
-- identificado no relatório de validação de 2026-04-08.
--
-- CAUSA DO BUG
-- O campo `localVelocity` estava declarado duas vezes no struct ctypes:
--   1ª ocorrência: após `roadTemp`       ← ERRADA (causava drift de +12 bytes)
--   2ª ocorrência: após `brakeBias`      ← CORRETA (posição real na Shared Memory)
-- Com o drift de 12 bytes, o campo `clutch` (esperado: 0.0–1.0) estava
-- lendo os bytes de `brakeTemp[0]` da memória do AC — temperatura de freio
-- em graus Celsius, tipicamente 57–99°C em Monza.
--
-- ESCOPO
-- Voltas com qualquer mini-setor com clutch > 1.0 são corrompidas.
-- Não é possível recuperar o valor real do clutch (era temperatura de freio).
-- A correção marca essas voltas como is_valid = FALSE para excluí-las
-- do treino e das queries históricas, sem remover dados do banco.
--
-- REFERÊNCIA
-- Commit de correção do struct: 01558b4 (full integration with supabase)
-- As voltas corrompidas foram gravadas antes desse commit.
--
-- COMO EXECUTAR
-- Supabase Studio → SQL Editor → colar este arquivo → Run
-- Idempotente: seguro de executar múltiplas vezes.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- PASSO 1 — Diagnóstico (leitura, sem alteração)
-- Execute este bloco primeiro para entender o volume afetado.
-- ---------------------------------------------------------------------------

SELECT
    l.id                                        AS lap_id,
    l.lap_number,
    l.lap_time_ms,
    l.is_valid,
    s.track_id,
    s.car_model,
    s.started_at                                AS session_start,
    COUNT(ms.id)                                AS total_mini_sectors,
    COUNT(ms.id) FILTER (WHERE ms.clutch > 1.0) AS corrupted_sectors,
    ROUND(MAX(ms.clutch)::NUMERIC, 2)           AS clutch_max
FROM laps l
JOIN sessions s ON l.session_id = s.id
JOIN mini_sectors ms ON ms.lap_id = l.id
WHERE s.track_id = 'monza'
GROUP BY l.id, l.lap_number, l.lap_time_ms, l.is_valid, s.track_id, s.car_model, s.started_at
HAVING COUNT(ms.id) FILTER (WHERE ms.clutch > 1.0) > 0
ORDER BY s.started_at, l.lap_number;

-- Resultado esperado: lista das voltas corrompidas com clutch_max entre 57 e 99.


-- ---------------------------------------------------------------------------
-- PASSO 2 — Soft delete: marcar voltas corrompidas como is_valid = FALSE
--
-- Usa a coluna is_valid que o pipeline de treino e as queries históricas
-- já filtram (WHERE is_valid = TRUE), excluindo essas voltas automaticamente
-- de todas as análises futuras sem remover os dados do banco.
-- ---------------------------------------------------------------------------

WITH corrupted_lap_ids AS (
    SELECT DISTINCT ms.lap_id
    FROM mini_sectors ms
    WHERE ms.clutch > 1.0
)
UPDATE laps
SET
    is_valid = FALSE
WHERE id IN (SELECT lap_id FROM corrupted_lap_ids)
  AND is_valid = TRUE;   -- no-op se já estiver marcado

-- Resultado esperado: UPDATE N (onde N = número de voltas corrompidas)


-- ---------------------------------------------------------------------------
-- PASSO 3 — Verificação pós-correção
-- ---------------------------------------------------------------------------

SELECT
    CASE WHEN is_valid THEN 'válidas' ELSE 'inválidas (corrompidas)' END AS status,
    COUNT(*)                                                              AS total_voltas,
    s.track_id
FROM laps l
JOIN sessions s ON l.session_id = s.id
WHERE s.track_id = 'monza'
GROUP BY is_valid, s.track_id
ORDER BY is_valid DESC;

-- Resultado esperado após correção:
--   status                          | total_voltas | track_id
--   --------------------------------|-------------|----------
--   válidas                         |     22       | monza
--   inválidas (corrompidas)         |     10       | monza


-- ---------------------------------------------------------------------------
-- ALTERNATIVA — Hard delete (use apenas se preferir remover os dados)
--
-- O CASCADE garante que mini_sectors e lap_patterns associados
-- também são removidos automaticamente.
-- ATENÇÃO: irreversível. Só execute se não precisar dos dados históricos.
-- ---------------------------------------------------------------------------

-- DELETE FROM laps
-- WHERE id IN (
--     SELECT DISTINCT ms.lap_id
--     FROM mini_sectors ms
--     WHERE ms.clutch > 1.0
-- );


-- ---------------------------------------------------------------------------
-- PASSO 4 — (Opcional) Corrigir personal_bests contaminados
--
-- Se alguma das voltas corrompidas foi marcada como melhor volta histórica,
-- o personal_best aponta para um lap_id inválido.
-- Este bloco remove registros de personal_best que referenciam voltas inválidas.
-- ---------------------------------------------------------------------------

DELETE FROM personal_bests
WHERE lap_id IN (
    SELECT id FROM laps WHERE is_valid = FALSE
);

-- Se a melhor volta real (válida) existir no banco, o sistema irá
-- recriar o personal_best na próxima sessão ao detectar is_alltime_best = TRUE.
