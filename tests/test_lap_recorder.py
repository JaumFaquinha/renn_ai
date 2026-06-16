"""
Testes da FASE 2 — LapRecorder e SectorAggregator.

Executa offline com snapshots sintéticos.
"""

import json
from pathlib import Path

import pytest

from src.memory.graphics_page import AC_LIVE
from src.recording.lap_recorder import LapRecorder
from src.recording.sector_aggregator import SectorAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_snapshot(
    position: float,
    speed: float = 150.0,
    throttle: float = 0.8,
    brake: float = 0.0,
    is_valid: bool = True,
) -> dict:
    """Cria um snapshot sintético no formato de snapshot_to_dict()."""
    return {
        "track_position": position,
        "delta_vs_best": 0.0,
        "throttle": throttle,
        "brake": brake,
        "steering": 0.1,
        "gear": 4,
        "rpms": 7000,
        "clutch": 0.0,
        "speed_kmh": speed,
        "gforce_x": 0.5,
        "gforce_y": -0.3,
        "gforce_z": 1.0,
        "local_ang_vel_x": 0.01,
        "local_ang_vel_y": 0.01,
        "local_ang_vel_z": 0.02,
        "wheel_slip_fl": 0.05,
        "wheel_slip_fr": 0.05,
        "wheel_slip_rl": 0.06,
        "wheel_slip_rr": 0.06,
        "tc_active": 0.0,
        "abs_active": 0.0,
        "drs_active": 0,
        "drs_available": 0,
        "brake_bias": 0.58,
        "surface_grip": 0.97,
        "air_temp": 24.0,
        "road_temp": 31.0,
        "_status": AC_LIVE,
        "_flag": 0,
        "_number_of_tyres_out": 0,
        "_pit_limiter_on": 0 if is_valid else 1,
        "_is_in_pit": 0,
        "_is_in_pit_lane": 0,
        "_penalty_time": 0.0,
        "_car_damage_max": 0.0,
        "_is_ai_controlled": 0,
        "_i_current_time_ms": int(position * 90000),
        "_i_best_time_ms": 90000,
        "_i_last_time_ms": 91000,
        "_last_sector_time_ms": 0,
        "_current_sector_index": int(position * 3),
        "_completed_laps": 1,
    }


def simulate_lap(recorder: LapRecorder, positions: list[float]) -> list:
    """Simula uma sequência de posições na spline, retorna voltas completadas."""
    completed = []
    for pos in positions:
        snap = make_snapshot(pos)
        result = recorder.process_snapshot(snap)
        if result is not None:
            completed.append(result)
    return completed


# ---------------------------------------------------------------------------
# Testes do SectorAggregator
# ---------------------------------------------------------------------------

class TestSectorAggregator:
    def test_aggregate_returns_none_for_empty_list(self):
        agg = SectorAggregator()
        assert agg.aggregate([]) is None

    def test_aggregate_single_snapshot(self):
        agg = SectorAggregator()
        snap = make_snapshot(0.05, speed=150.0, throttle=0.8, brake=0.0)
        result = agg.aggregate([snap])
        assert result is not None
        assert result["throttle"] == pytest.approx(0.8)
        assert result["speed_kmh"] == pytest.approx(150.0)

    def test_aggregate_speed_min_is_minimum(self):
        agg = SectorAggregator()
        snaps = [
            make_snapshot(0.05, speed=200.0),
            make_snapshot(0.05, speed=180.0),
            make_snapshot(0.05, speed=150.0),
        ]
        result = agg.aggregate(snaps)
        assert result["speed_min"] == pytest.approx(150.0)

    def test_aggregate_throttle_is_mean(self):
        agg = SectorAggregator()
        snaps = [
            make_snapshot(0.05, throttle=0.6),
            make_snapshot(0.05, throttle=0.8),
            make_snapshot(0.05, throttle=1.0),
        ]
        result = agg.aggregate(snaps)
        assert result["throttle"] == pytest.approx(0.8)

    def test_aggregate_gear_is_mode(self):
        agg = SectorAggregator()
        snaps = [
            make_snapshot(0.05),
            make_snapshot(0.05),
            make_snapshot(0.05),
        ]
        snaps[0]["gear"] = 3
        snaps[1]["gear"] = 4
        snaps[2]["gear"] = 4
        result = agg.aggregate(snaps)
        assert result["gear"] == 4  # Moda

    def test_aggregate_track_position_is_midpoint(self):
        agg = SectorAggregator()
        snaps = [
            make_snapshot(0.10),
            make_snapshot(0.11),
            make_snapshot(0.12),
        ]
        result = agg.aggregate(snaps)
        assert result["track_position"] == pytest.approx(0.11)

    def test_aggregate_delta_per_sector_is_last_minus_first(self):
        """
        delta_per_sector deve ser a diferença entre o último e o primeiro
        snapshot em delta_vs_best — captura a perda ocorrida no mini-setor.
        """
        agg = SectorAggregator()
        snaps = [
            {**make_snapshot(0.10), "delta_vs_best": 1.0},
            {**make_snapshot(0.10), "delta_vs_best": 1.15},
            {**make_snapshot(0.10), "delta_vs_best": 1.25},
        ]
        result = agg.aggregate(snaps)
        assert "delta_per_sector" in result
        assert result["delta_per_sector"] == pytest.approx(0.25)  # 1.25 - 1.0

    def test_aggregate_delta_per_sector_negative_when_gaining(self):
        """delta_per_sector negativo indica setor onde o piloto ganhou tempo."""
        agg = SectorAggregator()
        snaps = [
            {**make_snapshot(0.20), "delta_vs_best": 2.0},
            {**make_snapshot(0.20), "delta_vs_best": 1.9},  # ganhou 0.1s
        ]
        result = agg.aggregate(snaps)
        assert result["delta_per_sector"] == pytest.approx(-0.1)

    def test_aggregate_delta_per_sector_zero_for_single_snapshot(self):
        """Com apenas um snapshot, delta_per_sector é 0.0 (sem variação)."""
        agg = SectorAggregator()
        snap = {**make_snapshot(0.05), "delta_vs_best": 3.0}
        result = agg.aggregate([snap])
        assert result["delta_per_sector"] == pytest.approx(0.0)

    def test_aggregate_delta_per_sector_absent_without_delta_vs_best(self):
        """Sem delta_vs_best nos snapshots, delta_per_sector não deve aparecer."""
        agg = SectorAggregator()
        snap = {k: v for k, v in make_snapshot(0.05).items() if k != "delta_vs_best"}
        result = agg.aggregate([snap])
        assert "delta_per_sector" not in result


# ---------------------------------------------------------------------------
# Testes do LapRecorder — validação de volta
# ---------------------------------------------------------------------------

class TestLapRecorderValidation:
    def test_rejects_snapshot_with_pit_limiter(self):
        snap = make_snapshot(0.5, is_valid=False)
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_rejects_snapshot_in_pit(self):
        snap = make_snapshot(0.5)
        snap["_is_in_pit"] = 1
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_rejects_snapshot_in_pit_lane(self):
        snap = make_snapshot(0.5)
        snap["_is_in_pit_lane"] = 1
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_rejects_snapshot_with_tyres_out(self):
        snap = make_snapshot(0.5)
        snap["_number_of_tyres_out"] = 2
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_accepts_snapshot_ai_controlled(self):
        # AI-controlled laps são aceitas para permitir registro de tempos da IA.
        snap = make_snapshot(0.5)
        snap["_is_ai_controlled"] = 1
        assert LapRecorder._is_snapshot_invalid(snap) is False

    def test_rejects_snapshot_with_high_damage(self):
        snap = make_snapshot(0.5)
        snap["_car_damage_max"] = 0.5  # Acima do threshold padrão (0.1)
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_rejects_snapshot_with_penalty(self):
        snap = make_snapshot(0.5)
        snap["_penalty_time"] = 5.0
        assert LapRecorder._is_snapshot_invalid(snap) is True

    def test_accepts_valid_snapshot(self):
        snap = make_snapshot(0.5)
        assert LapRecorder._is_snapshot_invalid(snap) is False


# ---------------------------------------------------------------------------
# Testes de fixture de volta real (JSON)
# ---------------------------------------------------------------------------

class TestLapFixture:
    """Valida a fixture de volta de exemplo."""

    def test_fixture_file_exists(self):
        fixture_path = Path(__file__).parent / "fixtures" / "sample_lap.json"
        assert fixture_path.exists(), "Fixture sample_lap.json não encontrada"

    def test_fixture_has_correct_schema(self):
        fixture_path = Path(__file__).parent / "fixtures" / "sample_lap.json"
        with open(fixture_path) as f:
            lap = json.load(f)

        assert "lap_number" in lap
        assert "track_id" in lap
        assert "lap_time_ms" in lap
        assert "mini_sectors" in lap
        assert isinstance(lap["mini_sectors"], list)
        assert len(lap["mini_sectors"]) > 0

    def test_fixture_mini_sectors_have_required_fields(self):
        fixture_path = Path(__file__).parent / "fixtures" / "sample_lap.json"
        with open(fixture_path) as f:
            lap = json.load(f)

        required = [
            "track_position", "delta_vs_best", "throttle", "brake",
            "steering", "gear", "rpms", "speed_kmh", "speed_min",
            "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
            "brake_bias", "surface_grip",
        ]
        for sector in lap["mini_sectors"]:
            for field in required:
                assert field in sector, f"Campo '{field}' ausente no mini-setor"

    def test_fixture_positions_are_in_valid_range(self):
        fixture_path = Path(__file__).parent / "fixtures" / "sample_lap.json"
        with open(fixture_path) as f:
            lap = json.load(f)

        for sector in lap["mini_sectors"]:
            pos = sector["track_position"]
            assert 0.0 <= pos <= 1.0, f"Posição inválida: {pos}"


# ---------------------------------------------------------------------------
# Testes de teleporte / restart de sessão (Proposta A)
# ---------------------------------------------------------------------------

class TestLapRecorderTeleport:
    """
    Cobre o cenário em que o piloto retorna ao box ('Return to pits'),
    reinicia a sessão ou troca de carro. A spline salta abruptamente
    (ex.: 0.7 → 0.0) sem cruzamento limpo da linha de chegada,
    o que antes deixava o flag _lap_invalid preso indefinidamente.
    """

    def test_teleport_resets_invalid_flag(self):
        """Salto negativo grande na spline limpa o flag _lap_invalid."""
        recorder = LapRecorder(track_id="test")
        # Simular passagem pelo pit lane que invalida a volta
        invalid_snap = make_snapshot(0.40)
        invalid_snap["_is_in_pit_lane"] = 1
        recorder.process_snapshot(invalid_snap)
        assert recorder._lap_invalid is True

        # Teleporte: spline salta de 0.40 para 0.01 (return to pits/restart)
        teleport_snap = make_snapshot(0.01)
        recorder.process_snapshot(teleport_snap)

        # O flag deve ter sido limpo pela detecção de teleporte
        assert recorder._lap_invalid is False
        # Buffer da volta antiga descartado
        assert recorder._current_lap == []
        assert recorder._sector_buffer == []

    def test_teleport_clears_lap_buffer(self):
        """Teleporte descarta os mini-setores acumulados da volta em curso."""
        recorder = LapRecorder(track_id="test")
        # Acumular alguns mini-setores válidos
        for pos in [0.05, 0.15, 0.25, 0.35]:
            recorder.process_snapshot(make_snapshot(pos))
        assert len(recorder._current_lap) > 0

        # Teleporte para o início da pista
        recorder.process_snapshot(make_snapshot(0.02))

        assert recorder._current_lap == []
        assert recorder._sector_buffer == []

    def test_small_backward_jitter_does_not_trigger_reset(self):
        """Pequenas oscilações negativas na spline (jitter de leitura) não disparam reset."""
        recorder = LapRecorder(track_id="test")
        recorder.process_snapshot(make_snapshot(0.30))
        # Marca como inválido para detectar se o reset foi disparado
        recorder._lap_invalid = True

        # Oscilação pequena (< 0.1): não deve resetar
        recorder.process_snapshot(make_snapshot(0.28))
        assert recorder._lap_invalid is True  # Flag preservado — não foi reset

    def test_normal_lap_completion_not_treated_as_teleport(self):
        """Cruzamento normal da linha de chegada (0.99 → 0.01) não dispara reset prematuro."""
        recorder = LapRecorder(track_id="test")
        # Simular fim de volta (perto da linha)
        recorder.process_snapshot(make_snapshot(0.99))
        # Cruzamento normal — deve ser tratado por _detect_lap_start, não pelo teleporte
        result = recorder.process_snapshot(make_snapshot(0.01))
        # Não há volta válida (faltam mini-setores), mas o reset deve ter ocorrido
        # via fluxo normal de _finalize_lap → _reset_lap. _last_position deve estar atualizado.
        assert recorder._last_position == pytest.approx(0.01)

    def test_invalid_reason_returns_specific_string(self):
        """_snapshot_invalid_reason retorna a razão específica para diagnóstico (Proposta D)."""
        snap = make_snapshot(0.5)
        snap["_pit_limiter_on"] = 1
        assert LapRecorder._snapshot_invalid_reason(snap) == "pit_limiter_on"

        snap = make_snapshot(0.5)
        snap["_is_in_pit_lane"] = 1
        assert LapRecorder._snapshot_invalid_reason(snap) == "is_in_pit_lane"

        # Snapshot válido retorna None
        assert LapRecorder._snapshot_invalid_reason(make_snapshot(0.5)) is None
