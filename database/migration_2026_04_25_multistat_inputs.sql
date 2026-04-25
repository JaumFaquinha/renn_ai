-- =============================================================================
-- Migration 2026-04-25: Multi-stat para inputs do piloto (Proposal P1)
-- =============================================================================
-- Contexto: a média sozinha destrói a dinâmica intra-setor. Threshold-braking
-- de 1.0 por 0.3s e freio constante 0.27 por 1.1s produzem a mesma média
-- (~0.27). Adicionamos peak/valley/variability para que o SectorModel possa
-- aprender padrões de erro do piloto (lockup, pulse de TC, lift abrupto).
--
-- Validação: setores p10 vs p90 de delta_per_sector têm médias de inputs
-- <5% de diferença (sem signal). Após esta migration + retreino com novos
-- dados, esperado: importance dos inputs sobe de <1% para 10–25%.
--
-- Como aplicar:
--   1. Supabase Studio → SQL Editor
--   2. Cole este arquivo inteiro
--   3. Run
--
-- Idempotente: ADD COLUMN IF NOT EXISTS é no-op se a coluna já existir.
-- Reversível: para reverter, ALTER TABLE mini_sectors DROP COLUMN <nome>.
-- =============================================================================

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
