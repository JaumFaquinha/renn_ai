-- =============================================================================
-- Engenheiro de Corrida IA — Schema do Banco de Dados (Supabase / PostgreSQL)
-- =============================================================================
-- Migration idempotente: seguro de executar múltiplas vezes.
-- Executar no Supabase Studio: SQL Editor → colar este arquivo → Run.
--
-- Ordem de execução:
--   1. sessions
--   2. laps
--   3. mini_sectors
--   4. lap_patterns
--   5. personal_bests
--   6. Índices
--   7. RLS (Row Level Security)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. sessions — Uma por execução do run_session.py
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,                  -- piloto (SUPABASE_USER_ID)
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    track_id        TEXT NOT NULL,
    car_model       TEXT NOT NULL DEFAULT 'unknown',
    session_type    TEXT,                           -- 'practice' | 'qualifying' | 'hotlap' | 'race'
    air_temp        FLOAT,
    road_temp       FLOAT,
    total_laps      INT NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- 2. laps — Volta individual gravada pelo LapRecorder
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS laps (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    lap_number          INT NOT NULL,
    lap_time_ms         INT NOT NULL,
    is_valid            BOOLEAN NOT NULL DEFAULT TRUE,
    is_session_best     BOOLEAN NOT NULL DEFAULT FALSE,
    is_alltime_best     BOOLEAN NOT NULL DEFAULT FALSE,
    total_time_lost_s   FLOAT,                      -- soma dos deltas negativos dos top setores
    tyre_compound       TEXT,                       -- composto ativo ao cruzar a linha de chegada
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 3. mini_sectors — Telemetria granular (~100 linhas por volta)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mini_sectors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lap_id          UUID NOT NULL REFERENCES laps(id) ON DELETE CASCADE,

    -- Posição e delta
    track_position   FLOAT NOT NULL,
    delta_vs_best    FLOAT,
    delta_per_sector FLOAT,             -- diff primeiro→último snapshot no mini-setor (target do SectorModel)

    -- Inputs do piloto
    throttle        FLOAT,
    brake           FLOAT,
    steering        FLOAT,
    gear            SMALLINT,
    rpms            INT,
    clutch          FLOAT,

    -- Velocidade
    speed_kmh       FLOAT,
    speed_min       FLOAT,

    -- G-forces
    gforce_x        FLOAT,
    gforce_y        FLOAT,
    gforce_z        FLOAT,

    -- Velocidade angular local
    local_ang_vel_x FLOAT,
    local_ang_vel_y FLOAT,
    local_ang_vel_z FLOAT,

    -- Escorregamento de pneus
    wheel_slip_fl   FLOAT,
    wheel_slip_fr   FLOAT,
    wheel_slip_rl   FLOAT,
    wheel_slip_rr   FLOAT,

    -- Sistemas ativos
    tc_active       FLOAT,
    abs_active      FLOAT,
    drs_active      INT,
    drs_available   INT,

    -- Contexto
    brake_bias      FLOAT,
    surface_grip    FLOAT,
    air_temp        FLOAT,
    road_temp       FLOAT
);

-- ---------------------------------------------------------------------------
-- 4. lap_patterns — Padrões detectados pelo PatternDetector por volta
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lap_patterns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lap_id          UUID NOT NULL REFERENCES laps(id) ON DELETE CASCADE,
    track_position  FLOAT NOT NULL,
    cause           TEXT NOT NULL,
    confidence      FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    evidence        JSONB,                          -- campos de evidência da detecção
    corner_name     TEXT,
    corner_type     TEXT
);

-- ---------------------------------------------------------------------------
-- 5. personal_bests — Melhor volta por pista + carro + piloto
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS personal_bests (
    user_id         UUID NOT NULL,
    track_id        TEXT NOT NULL,
    car_model       TEXT NOT NULL,
    lap_time_ms     INT NOT NULL,
    lap_id          UUID REFERENCES laps(id) ON DELETE SET NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, track_id, car_model)
);

-- ---------------------------------------------------------------------------
-- 6. Índices de performance
-- ---------------------------------------------------------------------------

-- Busca de voltas por sessão (acesso mais comum)
CREATE INDEX IF NOT EXISTS idx_laps_session_id
    ON laps(session_id);

-- Busca de mini-setores por volta, ordenados por posição
CREATE INDEX IF NOT EXISTS idx_mini_sectors_lap_position
    ON mini_sectors(lap_id, track_position);

-- Busca de padrões por volta
CREATE INDEX IF NOT EXISTS idx_lap_patterns_lap_id
    ON lap_patterns(lap_id);

-- Busca de sessões por piloto + pista (queries históricas)
CREATE INDEX IF NOT EXISTS idx_sessions_user_track
    ON sessions(user_id, track_id);

-- ---------------------------------------------------------------------------
-- 7. Row Level Security (RLS)
-- ---------------------------------------------------------------------------
-- O CLI Python usa service_role key → bypassa RLS automaticamente.
-- As políticas protegem dados caso uma anon key seja exposta futuramente.

ALTER TABLE sessions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE laps           ENABLE ROW LEVEL SECURITY;
ALTER TABLE mini_sectors   ENABLE ROW LEVEL SECURITY;
ALTER TABLE lap_patterns   ENABLE ROW LEVEL SECURITY;
ALTER TABLE personal_bests ENABLE ROW LEVEL SECURITY;

-- Política: usuário vê apenas seus próprios dados
-- (sem efeito com service_role key)

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'users_own_sessions' AND tablename = 'sessions'
    ) THEN
        CREATE POLICY users_own_sessions ON sessions
            FOR ALL USING (user_id = auth.uid());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'users_own_laps' AND tablename = 'laps'
    ) THEN
        CREATE POLICY users_own_laps ON laps
            FOR ALL USING (
                session_id IN (SELECT id FROM sessions WHERE user_id = auth.uid())
            );
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'users_own_mini_sectors' AND tablename = 'mini_sectors'
    ) THEN
        CREATE POLICY users_own_mini_sectors ON mini_sectors
            FOR ALL USING (
                lap_id IN (
                    SELECT l.id FROM laps l
                    JOIN sessions s ON l.session_id = s.id
                    WHERE s.user_id = auth.uid()
                )
            );
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'users_own_lap_patterns' AND tablename = 'lap_patterns'
    ) THEN
        CREATE POLICY users_own_lap_patterns ON lap_patterns
            FOR ALL USING (
                lap_id IN (
                    SELECT l.id FROM laps l
                    JOIN sessions s ON l.session_id = s.id
                    WHERE s.user_id = auth.uid()
                )
            );
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE policyname = 'users_own_personal_bests' AND tablename = 'personal_bests'
    ) THEN
        CREATE POLICY users_own_personal_bests ON personal_bests
            FOR ALL USING (user_id = auth.uid());
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 8. Migrations — Correções de tipo aplicadas sobre tabelas já existentes
-- ---------------------------------------------------------------------------
-- Bug fix: drs_active e drs_available eram SMALLINT (max 32.767).
-- A Shared Memory do AC retorna valores de temperatura relidos como int
-- (~1.1×10⁹) quando há misalinhamento de struct, estourando o SMALLINT
-- e abortando silenciosamente o bulk INSERT de mini_sectors.
-- Solução: INT (max ~2.1×10⁹) absorve qualquer garbage value sem erro.
-- Idempotente: ALTER TYPE é no-op se a coluna já for INT.
ALTER TABLE mini_sectors
    ALTER COLUMN drs_active    TYPE INT USING drs_active::INT,
    ALTER COLUMN drs_available TYPE INT USING drs_available::INT;

-- Migration: adiciona tyre_compound à tabela laps (2026-03-10)
-- Contexto: composto de pneu não estava no pipeline — corrige análise de PatternDetector
-- que usava thresholds fixos sem distinguir entre compostos diferentes.
-- Idempotente: ADD COLUMN IF NOT EXISTS é no-op se a coluna já existir.
ALTER TABLE laps ADD COLUMN IF NOT EXISTS tyre_compound TEXT;

-- Migration: adiciona delta_per_sector à tabela mini_sectors (2026-04-14)
-- Contexto: substitui a computação retroativa (diff entre médias de setores consecutivos)
-- pelo valor preciso calculado intra-setor pelo SectorAggregator (dvb_last - dvb_first).
-- Dados históricos sem essa coluna continuam utilizáveis via retrocomputação em train_model.py.
-- Idempotente: ADD COLUMN IF NOT EXISTS é no-op se a coluna já existir.
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS delta_per_sector FLOAT;

-- Migration: estatísticas multi-stat para inputs do piloto (2026-04-25, Proposal P1)
-- Contexto: a média sozinha destrói a dinâmica intra-setor — peak/valley/std
-- são necessárias para o modelo aprender PADRÕES (frenagem tardia, lockup,
-- pulse de TC). Validação empírica: setores p10 vs p90 de delta_per_sector
-- têm médias de inputs <5% diferentes; o sinal causal está nos extremos.
-- Idempotente: ADD COLUMN IF NOT EXISTS é no-op se a coluna já existir.
-- Total: 9 inputs × 3 estatísticas = 27 novas colunas (todas nullable, default NULL).
-- Dados pré-migration: NULL → tratado como 0.0 pelo SectorModel via s.get(f, 0.0).

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS throttle_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS throttle_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS throttle_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS brake_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS brake_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS brake_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS steering_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS steering_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS steering_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fl_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fl_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fl_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fr_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fr_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_fr_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rl_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rl_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rl_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rr_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rr_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS wheel_slip_rr_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS tc_active_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS tc_active_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS tc_active_std FLOAT;

ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS abs_active_max FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS abs_active_min FLOAT;
ALTER TABLE mini_sectors ADD COLUMN IF NOT EXISTS abs_active_std FLOAT;
