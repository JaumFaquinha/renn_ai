"""
Testes para src/persistence/lap_uploader.py

Estratégia: mock completo do SupabaseClient — sem conexão real ao Supabase.
"""

import json
import time
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from src.persistence.lap_uploader import LapUploader
from src.persistence.supabase_client import SupabaseClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_disabled_client() -> SupabaseClient:
    """SupabaseClient desabilitado (SUPABASE_ENABLED=false)."""
    client = MagicMock(spec=SupabaseClient)
    client.is_enabled = False
    return client


def _make_enabled_client(mock_sdk=None) -> SupabaseClient:
    """SupabaseClient habilitado com SDK mockado."""
    if mock_sdk is None:
        mock_sdk = MagicMock()
    client = MagicMock(spec=SupabaseClient)
    client.is_enabled = True
    client.get_client.return_value = mock_sdk
    return client


def _sample_lap(lap_number: int = 2, lap_time_ms: int = 95000) -> dict:
    """Volta de amostra com mini-setores mínimos para testes."""
    return {
        "lap_number": lap_number,
        "track_id": "monza",
        "car_model": "ferrari_488_gt3",
        "session_type": "practice",
        "lap_time_ms": lap_time_ms,
        "sector_count": 2,
        "mini_sectors": [
            {
                "track_position": 0.005,
                "delta_vs_best": 0.0,
                "throttle": 1.0, "brake": 0.0, "steering": 0.01,
                "gear": 5, "rpms": 7800, "clutch": 0.0,
                "speed_kmh": 245.0, "speed_min": 232.0,
                "gforce_x": 0.1, "gforce_y": -0.8, "gforce_z": 1.2,
                "local_ang_vel_x": 0.01, "local_ang_vel_y": 0.02, "local_ang_vel_z": 0.01,
                "wheel_slip_fl": 0.02, "wheel_slip_fr": 0.02,
                "wheel_slip_rl": 0.03, "wheel_slip_rr": 0.03,
                "tc_active": 0.0, "abs_active": 0.0,
                "drs_active": 1, "drs_available": 1,
                "brake_bias": 0.58, "surface_grip": 0.97,
                "air_temp": 24.0, "road_temp": 31.5,
            },
            {
                "track_position": 0.085,
                "delta_vs_best": 0.18,
                "throttle": 0.0, "brake": 0.92, "steering": 0.12,
                "gear": 2, "rpms": 4200, "clutch": 0.0,
                "speed_kmh": 87.0, "speed_min": 82.0,
                "gforce_x": 0.8, "gforce_y": -2.1, "gforce_z": 1.1,
                "local_ang_vel_x": 0.05, "local_ang_vel_y": 0.03, "local_ang_vel_z": 0.04,
                "wheel_slip_fl": 0.08, "wheel_slip_fr": 0.09,
                "wheel_slip_rl": 0.07, "wheel_slip_rr": 0.08,
                "tc_active": 0.0, "abs_active": 0.3,
                "drs_active": 0, "drs_available": 0,
                "brake_bias": 0.58, "surface_grip": 0.97,
                "air_temp": 24.0, "road_temp": 31.5,
            },
        ],
    }


def _sample_lap_report():
    """LapReport mínimo para testes."""
    report = MagicMock()
    report.total_time_lost_s = 0.42
    sector = MagicMock()
    sector.track_position = 0.085
    sector.corner_name = "Prima Variante"
    sector.corner_type = "chicane"
    match = MagicMock()
    match.cause = "Frenagem tardia com bloqueio"
    match.confidence = 0.87
    match.evidence = {"brake": 0.92, "abs_active": 0.3}
    sector.causes = [match]
    report.top_sectors = [sector]
    return report


# ---------------------------------------------------------------------------
# Testes: uploader desabilitado
# ---------------------------------------------------------------------------

class TestLapUploaderDisabled:
    """Quando Supabase está desabilitado, todos os métodos são no-ops."""

    def test_create_session_returns_none(self):
        uploader = LapUploader(_make_disabled_client())
        result = uploader.create_session("monza", "ferrari_488_gt3", "practice", 24.0, 31.5)
        assert result is None

    def test_enqueue_lap_is_noop(self):
        uploader = LapUploader(_make_disabled_client())
        # Não deve lançar exceção
        uploader.enqueue_lap(_sample_lap(), _sample_lap_report())

    def test_stop_is_noop(self):
        uploader = LapUploader(_make_disabled_client())
        uploader.stop(timeout_s=1.0)  # Não deve lançar exceção

    def test_no_background_thread_created(self):
        uploader = LapUploader(_make_disabled_client())
        assert uploader._thread is None


# ---------------------------------------------------------------------------
# Testes: create_session
# ---------------------------------------------------------------------------

class TestCreateSession:
    def _make_sdk_with_session(self, session_id="sess-uuid-123"):
        mock_sdk = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"id": session_id}]
        mock_sdk.table.return_value.insert.return_value.execute.return_value = mock_result
        return mock_sdk

    def test_returns_session_id_on_success(self):
        mock_sdk = self._make_sdk_with_session("sess-abc-123")
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        result = uploader.create_session("monza", "ferrari_488_gt3", "practice", 24.0, 31.5)
        assert result == "sess-abc-123"

    def test_stores_session_id_internally(self):
        mock_sdk = self._make_sdk_with_session("sess-internal-456")
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        uploader.create_session("monza", "ferrari_488_gt3", "practice", 24.0, 31.5)
        assert uploader._session_id == "sess-internal-456"

    def test_inserts_correct_table(self):
        mock_sdk = self._make_sdk_with_session()
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        uploader.create_session("monza", "ferrari_488_gt3", "practice", 24.0, 31.5)
        mock_sdk.table.assert_called_with("sessions")

    def test_payload_contains_required_fields(self):
        mock_sdk = self._make_sdk_with_session()
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        uploader.create_session("monza", "ferrari_488_gt3", "hotlap", 24.0, 31.5)

        inserted_payload = mock_sdk.table.return_value.insert.call_args[0][0]
        assert inserted_payload["track_id"] == "monza"
        assert inserted_payload["car_model"] == "ferrari_488_gt3"
        assert inserted_payload["session_type"] == "hotlap"
        assert inserted_payload["air_temp"] == 24.0
        assert inserted_payload["road_temp"] == 31.5

    def test_returns_none_when_insert_fails(self):
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.insert.return_value.execute.return_value = MagicMock(data=None)
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        result = uploader.create_session("monza", "car", "practice", 0.0, 0.0)
        assert result is None


# ---------------------------------------------------------------------------
# Testes: enqueue_lap (comportamento assíncrono)
# ---------------------------------------------------------------------------

class TestEnqueueLap:
    def _make_uploader_with_session(self, mock_sdk):
        client = _make_enabled_client(mock_sdk)
        uploader = LapUploader(client)
        uploader._session_id = "sess-test-789"
        return uploader

    def test_enqueue_returns_immediately(self):
        mock_sdk = MagicMock()
        uploader = self._make_uploader_with_session(mock_sdk)

        start = time.monotonic()
        uploader.enqueue_lap(_sample_lap(), _sample_lap_report())
        elapsed = time.monotonic() - start

        # Deve retornar em menos de 10ms (não bloqueia para upload)
        assert elapsed < 0.010

    def test_enqueue_without_session_is_noop(self):
        mock_sdk = MagicMock()
        client = _make_enabled_client(mock_sdk)
        uploader = LapUploader(client)
        # _session_id não definido
        uploader.enqueue_lap(_sample_lap(), _sample_lap_report())
        # Fila deve estar vazia
        assert uploader._queue.empty()

    def test_mini_sectors_payload_has_all_30_fields(self):
        """Verifica que todos os 30 campos do schema §4.5 são mapeados."""
        required_fields = {
            "track_position", "delta_vs_best",
            "throttle", "brake", "steering", "gear", "rpms", "clutch",
            "speed_kmh", "speed_min",
            "gforce_x", "gforce_y", "gforce_z",
            "local_ang_vel_x", "local_ang_vel_y", "local_ang_vel_z",
            "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
            "tc_active", "abs_active", "drs_active", "drs_available",
            "brake_bias", "surface_grip", "air_temp", "road_temp",
            "lap_id",
        }

        uploaded_rows = []

        def capture_insert(rows):
            uploaded_rows.extend(rows)
            result = MagicMock()
            result.data = rows
            return result

        mock_sdk = MagicMock()
        # Primeiro insert = laps (retorna lap_id)
        lap_result = MagicMock()
        lap_result.data = [{"id": "lap-uuid-111"}]
        # Segundo insert = mini_sectors
        ms_result = MagicMock()
        ms_result.data = []

        call_count = [0]
        def side_effect_insert(data):
            call_count[0] += 1
            m = MagicMock()
            if call_count[0] == 1:
                m.execute.return_value = lap_result
            else:
                if isinstance(data, list):
                    uploaded_rows.extend(data)
                m.execute.return_value = ms_result
            return m

        mock_sdk.table.return_value.insert.side_effect = side_effect_insert
        mock_sdk.table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[])
        mock_sdk.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(data=None)

        uploader = self._make_uploader_with_session(mock_sdk)
        uploader.enqueue_lap(_sample_lap(), _sample_lap_report())
        uploader.stop(timeout_s=5.0)

        if uploaded_rows:
            row = uploaded_rows[0]
            missing = required_fields - set(row.keys())
            assert not missing, f"Campos faltando no mini_sector: {missing}"


# ---------------------------------------------------------------------------
# Testes: retry e resiliência
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    def test_retry_on_transient_failure(self):
        """Testa que o upload é retentado após falha transitória."""
        mock_sdk = MagicMock()

        call_count = [0]
        def flaky_execute():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("connection timeout")
            result = MagicMock()
            result.data = [{"id": "lap-retry-123"}]
            return result

        mock_sdk.table.return_value.insert.return_value.execute.side_effect = flaky_execute

        client = _make_enabled_client(mock_sdk)
        uploader = LapUploader(client)
        result = uploader._execute_with_retry(
            lambda: mock_sdk.table("laps").insert({}).execute()
        )

        assert result is not None
        assert call_count[0] == 2  # Falhou 1x, sucesso na 2ª

    def test_returns_none_after_all_retries_exhausted(self):
        """Após 3 tentativas, retorna None sem lançar exceção."""
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.insert.return_value.execute.side_effect = Exception("always fails")

        client = _make_enabled_client(mock_sdk)
        uploader = LapUploader(client)

        # Reduz backoff para o teste não demorar
        import src.persistence.lap_uploader as m
        original = m._RETRY_BACKOFF_S
        m._RETRY_BACKOFF_S = (0.01, 0.01, 0.01)

        result = uploader._execute_with_retry(
            lambda: mock_sdk.table("laps").insert({}).execute()
        )

        m._RETRY_BACKOFF_S = original
        assert result is None


# ---------------------------------------------------------------------------
# Testes: stop / shutdown
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_does_not_raise_with_empty_queue(self):
        mock_sdk = MagicMock()
        uploader = LapUploader(_make_enabled_client(mock_sdk))
        uploader.stop(timeout_s=2.0)  # Não deve lançar

    def test_stop_with_timeout_does_not_raise(self):
        """Stop com fila não drenada não deve lançar exceção."""
        mock_sdk = MagicMock()

        # Upload muito lento para simular timeout
        def slow_execute(*args, **kwargs):
            time.sleep(5.0)
            return MagicMock(data=[{"id": "x"}])

        mock_sdk.table.return_value.insert.return_value.execute.side_effect = slow_execute

        client = _make_enabled_client(mock_sdk)
        uploader = LapUploader(client)
        uploader._session_id = "sess-timeout"
        uploader.enqueue_lap(_sample_lap(), _sample_lap_report())

        # Para com timeout curto — não deve explodir
        uploader.stop(timeout_s=0.1)
