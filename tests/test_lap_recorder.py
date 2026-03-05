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

    def test_rejects_snapshot_ai_controlled(self):
        snap = make_snapshot(0.5)
        snap["_is_ai_controlled"] = 1
        assert LapRecorder._is_snapshot_invalid(snap) is True

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
