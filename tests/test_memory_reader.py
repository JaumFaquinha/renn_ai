"""
Testes da FASE 1 — SharedMemoryReader.

Executa offline sem necessidade do Assetto Corsa.
Verifica estrutura das structs, graceful handling e conversão de snapshot.
"""

import ctypes
import time

import pytest

from src.memory.graphics_page import (
    AC_LIVE,
    AC_NO_FLAG,
    SPageFileGraphic,
)
from src.memory.physics_page import SPageFilePhysics
from src.memory.shared_memory_reader import (
    SharedMemoryReader,
    TelemetrySnapshot,
    snapshot_to_dict,
)
from src.memory.static_page import SPageFileStatic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_physics(**kwargs) -> SPageFilePhysics:
    """Cria SPageFilePhysics com valores padrão mais overrides."""
    p = SPageFilePhysics()
    p.gas = 1.0
    p.brake = 0.0
    p.speedKmh = 200.0
    p.rpms = 7500
    p.gear = 5
    p.performanceMeter = 0.0
    p.airTemp = 24.0
    p.roadTemp = 31.0
    for key, val in kwargs.items():
        setattr(p, key, val)
    return p


def make_graphics(**kwargs) -> SPageFileGraphic:
    """Cria SPageFileGraphic com valores padrão mais overrides."""
    g = SPageFileGraphic()
    g.status = AC_LIVE
    g.normalizedCarPosition = 0.5
    g.iCurrentTime = 50000
    g.iBestTime = 102340
    g.iLastTime = 103000
    g.isInPit = 0
    g.isInPitLane = 0
    g.flag = AC_NO_FLAG
    g.surfaceGrip = 0.97
    for key, val in kwargs.items():
        setattr(g, key, val)
    return g


def make_snapshot(**kwargs) -> TelemetrySnapshot:
    """Cria TelemetrySnapshot sintético para testes."""
    return TelemetrySnapshot(
        timestamp_ns=time.monotonic_ns(),
        physics=make_physics(**{k: v for k, v in kwargs.items() if hasattr(SPageFilePhysics(), k)}),
        graphics=make_graphics(**{k: v for k, v in kwargs.items() if hasattr(SPageFileGraphic(), k)}),
        static=SPageFileStatic(),
        read_ok=True,
    )


# ---------------------------------------------------------------------------
# Testes de estrutura ctypes
# ---------------------------------------------------------------------------

class TestStructSizes:
    """Verifica que as structs têm tamanho não-zero e campos acessíveis."""

    def test_physics_struct_has_positive_size(self):
        assert ctypes.sizeof(SPageFilePhysics) > 0

    def test_graphics_struct_has_positive_size(self):
        assert ctypes.sizeof(SPageFileGraphic) > 0

    def test_static_struct_has_positive_size(self):
        assert ctypes.sizeof(SPageFileStatic) > 0

    def test_physics_key_fields_accessible(self):
        p = SPageFilePhysics()
        assert hasattr(p, "gas")
        assert hasattr(p, "brake")
        assert hasattr(p, "speedKmh")
        assert hasattr(p, "rpms")
        assert hasattr(p, "performanceMeter")
        assert hasattr(p, "localAngularVel")
        assert hasattr(p, "brakeBias")
        assert hasattr(p, "wheelSlip")
        assert hasattr(p, "carDamage")
        assert hasattr(p, "isAIControlled")

    def test_graphics_key_fields_accessible(self):
        g = SPageFileGraphic()
        assert hasattr(g, "normalizedCarPosition")
        assert hasattr(g, "status")
        assert hasattr(g, "iCurrentTime")
        assert hasattr(g, "iBestTime")
        assert hasattr(g, "surfaceGrip")
        assert hasattr(g, "isInPit")
        assert hasattr(g, "flag")

    def test_static_key_fields_accessible(self):
        s = SPageFileStatic()
        assert hasattr(s, "trackSPlineLength")
        assert hasattr(s, "sectorCount")
        assert hasattr(s, "maxRpm")
        assert hasattr(s, "hasDRS")
        assert hasattr(s, "track")
        assert hasattr(s, "carModel")

    def test_physics_array_fields_have_correct_length(self):
        p = SPageFilePhysics()
        assert len(p.wheelSlip) == 4
        assert len(p.accG) == 3
        assert len(p.velocity) == 3
        assert len(p.localVelocity) == 3
        assert len(p.localAngularVel) == 3
        assert len(p.carDamage) == 5

    def test_physics_default_values_are_zero(self):
        p = SPageFilePhysics()
        assert p.gas == pytest.approx(0.0)
        assert p.brake == pytest.approx(0.0)
        assert p.speedKmh == pytest.approx(0.0)
        assert p.rpms == 0


# ---------------------------------------------------------------------------
# Testes de snapshot_to_dict
# ---------------------------------------------------------------------------

class TestSnapshotToDict:
    """Verifica a conversão de TelemetrySnapshot para dict serializável."""

    def test_returns_dict(self):
        snap = make_snapshot()
        result = snapshot_to_dict(snap)
        assert isinstance(result, dict)

    def test_contains_schema_fields(self):
        snap = make_snapshot()
        result = snapshot_to_dict(snap)
        required_fields = [
            "track_position", "delta_vs_best",
            "throttle", "brake", "steering", "gear", "rpms", "clutch",
            "speed_kmh",
            "gforce_x", "gforce_y", "gforce_z",
            "local_ang_vel_x", "local_ang_vel_y", "local_ang_vel_z",
            "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
            "tc_active", "abs_active", "drs_active", "drs_available",
            "brake_bias", "surface_grip", "air_temp", "road_temp",
        ]
        for field in required_fields:
            assert field in result, f"Campo ausente: {field}"

    def test_contains_validation_fields(self):
        snap = make_snapshot()
        result = snapshot_to_dict(snap)
        validation_fields = [
            "_status", "_flag", "_number_of_tyres_out", "_pit_limiter_on",
            "_is_in_pit", "_is_in_pit_lane", "_penalty_time", "_car_damage_max",
            "_is_ai_controlled",
        ]
        for field in validation_fields:
            assert field in result, f"Campo de validação ausente: {field}"

    def test_values_are_propagated_correctly(self):
        p = make_physics(gas=0.75, brake=0.5, speedKmh=150.0, rpms=6500, gear=4)
        g = make_graphics(normalizedCarPosition=0.42, status=AC_LIVE)
        snap = TelemetrySnapshot(
            timestamp_ns=time.monotonic_ns(),
            physics=p,
            graphics=g,
            static=SPageFileStatic(),
            read_ok=True,
        )
        result = snapshot_to_dict(snap)

        assert result["throttle"] == pytest.approx(0.75)
        assert result["brake"] == pytest.approx(0.5)
        assert result["speed_kmh"] == pytest.approx(150.0)
        assert result["rpms"] == 6500
        assert result["gear"] == 4
        assert result["track_position"] == pytest.approx(0.42)
        assert result["_status"] == AC_LIVE

    def test_car_damage_max_is_computed(self):
        p = make_physics()
        p.carDamage[0] = 0.05
        p.carDamage[1] = 0.12
        p.carDamage[2] = 0.03
        g = make_graphics()
        snap = TelemetrySnapshot(
            timestamp_ns=time.monotonic_ns(),
            physics=p, graphics=g, static=SPageFileStatic(), read_ok=True,
        )
        result = snapshot_to_dict(snap)
        assert result["_car_damage_max"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Testes de graceful handling (sem AC rodando)
# ---------------------------------------------------------------------------

class TestSharedMemoryReaderOffline:
    """Verifica comportamento quando o AC não está rodando."""

    def test_connect_returns_false_when_ac_not_running(self):
        reader = SharedMemoryReader()
        # Em ambiente de teste (sem AC), connect() deve retornar False
        result = reader.connect()
        assert result is False
        assert reader.is_connected is False

    def test_read_returns_error_snapshot_when_not_connected(self):
        reader = SharedMemoryReader()
        # Sem conectar
        snapshot = reader.read()
        assert snapshot.read_ok is False
        assert snapshot.error is not None
        assert len(snapshot.error) > 0

    def test_disconnect_is_safe_when_not_connected(self):
        reader = SharedMemoryReader()
        # Não deve lançar exceção
        reader.disconnect()
        assert reader.is_connected is False

    def test_context_manager_handles_ac_not_running(self):
        with SharedMemoryReader() as reader:
            assert reader.is_connected is False
            snapshot = reader.read()
            assert snapshot.read_ok is False

    def test_multiple_reads_without_connection_are_safe(self):
        reader = SharedMemoryReader()
        for _ in range(5):
            snapshot = reader.read()
            assert snapshot.read_ok is False

    def test_invalidate_static_cache_is_safe(self):
        reader = SharedMemoryReader()
        reader.invalidate_static_cache()  # Não deve lançar exceção
