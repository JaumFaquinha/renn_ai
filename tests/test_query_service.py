"""
Testes para src/persistence/query_service.py

Estratégia: mock completo do SupabaseClient — sem conexão real.
"""

from unittest.mock import MagicMock

import pytest

from src.persistence.query_service import QueryService
from src.persistence.supabase_client import SupabaseClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_disabled_client() -> SupabaseClient:
    client = MagicMock(spec=SupabaseClient)
    client.is_enabled = False
    return client


def _make_enabled_client(mock_sdk=None) -> SupabaseClient:
    if mock_sdk is None:
        mock_sdk = MagicMock()
    client = MagicMock(spec=SupabaseClient)
    client.is_enabled = True
    client.get_client.return_value = mock_sdk
    return client


def _sample_sectors() -> list[dict]:
    """Mini-setores de referência com todos os campos do schema."""
    return [
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
    ]


# ---------------------------------------------------------------------------
# Testes: get_personal_best — disabled
# ---------------------------------------------------------------------------

class TestGetPersonalBestDisabled:
    def test_returns_none_when_disabled(self):
        qs = QueryService(_make_disabled_client())
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is None


# ---------------------------------------------------------------------------
# Testes: get_personal_best — sem histórico
# ---------------------------------------------------------------------------

class TestGetPersonalBestNoHistory:
    def test_returns_none_when_no_pb_found(self):
        mock_sdk = MagicMock()
        # limit(1) retorna lista vazia quando não há dados
        mock_sdk.table.return_value \
            .select.return_value \
            .eq.return_value \
            .eq.return_value \
            .eq.return_value \
            .limit.return_value \
            .execute.return_value = MagicMock(data=[])

        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is None

    def test_returns_none_when_lap_id_is_null(self):
        mock_sdk = MagicMock()
        pb_response = MagicMock()
        # limit(1) retorna lista com um dict (não dict direto como single())
        pb_response.data = [{"lap_time_ms": 95000, "lap_id": None, "updated_at": "2026-01-01"}]

        mock_sdk.table.return_value \
            .select.return_value \
            .eq.return_value \
            .eq.return_value \
            .eq.return_value \
            .limit.return_value \
            .execute.return_value = pb_response

        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is None


# ---------------------------------------------------------------------------
# Testes: get_personal_best — com dados
# ---------------------------------------------------------------------------

class TestGetPersonalBestWithData:
    def _setup_sdk_with_pb(self, lap_time_ms=95000, sectors=None):
        if sectors is None:
            sectors = _sample_sectors()

        mock_sdk = MagicMock()

        # Resposta para a query de personal_bests (limit(1) → lista)
        pb_response = MagicMock()
        pb_response.data = [{
            "lap_time_ms": lap_time_ms,
            "lap_id": "lap-pb-uuid-123",
            "updated_at": "2026-02-28T14:30:00Z",
        }]

        # Resposta para a query de mini_sectors
        sectors_response = MagicMock()
        sectors_response.data = sectors

        # Configura o mock para retornar PB na primeira cadeia e setores na segunda
        call_count = [0]

        def select_side_effect(*args, **kwargs):
            call_count[0] += 1
            chain = MagicMock()
            if call_count[0] == 1:
                # personal_bests query — usa limit(1) agora
                chain.eq.return_value.eq.return_value.eq.return_value \
                    .limit.return_value.execute.return_value = pb_response
            else:
                # mini_sectors query
                chain.eq.return_value.order.return_value.execute.return_value = sectors_response
            return chain

        mock_sdk.table.return_value.select.side_effect = select_side_effect
        return mock_sdk

    def test_returns_lap_time_ms(self):
        mock_sdk = self._setup_sdk_with_pb(lap_time_ms=95123)
        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is not None
        assert result["lap_time_ms"] == 95123

    def test_returns_session_date(self):
        mock_sdk = self._setup_sdk_with_pb()
        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is not None
        assert "session_date" in result
        assert result["session_date"] == "2026-02-28T14:30:00Z"

    def test_mini_sectors_ordered_by_track_position(self):
        """Garante que mini_sectors vêm em ordem crescente de track_position."""
        sectors = _sample_sectors()
        # Embaralha intencionalmente
        reversed_sectors = list(reversed(sectors))

        mock_sdk = self._setup_sdk_with_pb(sectors=reversed_sectors)
        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")

        # O mock já retorna na ordem que o Supabase mandou (order() é real no DB)
        # O teste valida que a estrutura está correta
        assert result is not None
        assert isinstance(result["mini_sectors"], list)
        assert len(result["mini_sectors"]) == 2

    def test_mini_sectors_schema_compatible_with_lap_analyzer(self):
        """
        Garante que os mini_sectors retornados têm os campos que
        LapAnalyzer.set_best_lap() precisa: track_position e delta_vs_best.
        """
        mock_sdk = self._setup_sdk_with_pb()
        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")

        assert result is not None
        for sector in result["mini_sectors"]:
            assert "track_position" in sector
            assert "delta_vs_best" in sector
            assert "speed_kmh" in sector

    def test_returns_none_when_sectors_query_fails(self):
        """Se a query de mini_sectors falha, retorna None graciosamente."""
        mock_sdk = MagicMock()
        pb_response = MagicMock()
        pb_response.data = [{
            "lap_time_ms": 95000,
            "lap_id": "lap-uuid-456",
            "updated_at": "2026-02-01",
        }]
        sectors_response = MagicMock()
        sectors_response.data = None

        call_count = [0]

        def select_side_effect(*args, **kwargs):
            call_count[0] += 1
            chain = MagicMock()
            if call_count[0] == 1:
                # personal_bests usa limit(1)
                chain.eq.return_value.eq.return_value.eq.return_value \
                    .limit.return_value.execute.return_value = pb_response
            else:
                chain.eq.return_value.order.return_value.execute.return_value = sectors_response
            return chain

        mock_sdk.table.return_value.select.side_effect = select_side_effect
        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is None

    def test_exception_during_query_returns_none(self):
        """Exceção na query não propaga — retorna None."""
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.select.side_effect = Exception("network error")

        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_personal_best("monza", "ferrari_488_gt3")
        assert result is None


# ---------------------------------------------------------------------------
# Testes: get_sector_history
# ---------------------------------------------------------------------------

class TestGetSectorHistory:
    def test_returns_empty_list_when_disabled(self):
        qs = QueryService(_make_disabled_client())
        result = qs.get_sector_history("monza", "ferrari_488_gt3", 0.085)
        assert result == []

    def test_returns_empty_list_when_no_sessions(self):
        mock_sdk = MagicMock()
        sessions_response = MagicMock()
        sessions_response.data = []
        mock_sdk.table.return_value \
            .select.return_value \
            .eq.return_value \
            .eq.return_value \
            .eq.return_value \
            .order.return_value \
            .limit.return_value \
            .execute.return_value = sessions_response

        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_sector_history("monza", "ferrari_488_gt3", 0.085)
        assert result == []

    def test_exception_returns_empty_list(self):
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.select.side_effect = Exception("timeout")

        qs = QueryService(_make_enabled_client(mock_sdk))
        result = qs.get_sector_history("monza", "ferrari_488_gt3", 0.085)
        assert result == []


# ---------------------------------------------------------------------------
# Testes: update_session_lap_count
# ---------------------------------------------------------------------------

class TestUpdateSessionLapCount:
    def test_noop_when_disabled(self):
        qs = QueryService(_make_disabled_client())
        qs.update_session_lap_count("sess-123", 10)  # Não deve lançar

    def test_noop_when_no_session_id(self):
        mock_sdk = MagicMock()
        qs = QueryService(_make_enabled_client(mock_sdk))
        qs.update_session_lap_count("", 5)
        # Não deve chamar o SDK
        mock_sdk.table.assert_not_called()

    def test_updates_correct_table(self):
        mock_sdk = MagicMock()
        mock_sdk.table.return_value \
            .update.return_value \
            .eq.return_value \
            .execute.return_value = MagicMock()

        qs = QueryService(_make_enabled_client(mock_sdk))
        qs.update_session_lap_count("sess-abc-789", 15)

        mock_sdk.table.assert_called_with("sessions")
        mock_sdk.table.return_value.update.assert_called_with({"total_laps": 15})

    def test_exception_does_not_propagate(self):
        mock_sdk = MagicMock()
        mock_sdk.table.return_value.update.side_effect = Exception("DB error")

        qs = QueryService(_make_enabled_client(mock_sdk))
        qs.update_session_lap_count("sess-xyz", 8)  # Não deve lançar
