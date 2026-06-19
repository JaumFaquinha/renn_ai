"""
query_service.py — Consultas históricas ao Supabase.

Responsabilidades:
    - get_personal_best(): busca a melhor volta histórica por track + car + piloto
    - get_sector_history(): tendência de delta por posição na spline (Fase 7D)
    - update_session_lap_count(): atualiza total_laps no encerramento da sessão

Os mini-setores retornados por get_personal_best() são reconstruídos no mesmo
formato de list[dict] que LapAnalyzer.set_best_lap() espera.
"""

import logging
from typing import Optional

from config.settings import SUPABASE_USER_ID
from src.persistence.query_helpers import fetch_all_in
from src.persistence.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class QueryService:
    """
    Camada de consultas históricas ao Supabase.

    Quando o SupabaseClient está desabilitado, todos os métodos retornam
    None/[] silenciosamente — sem impacto no fluxo principal.
    """

    def __init__(self, supabase_client: SupabaseClient) -> None:
        self._client = supabase_client

    def get_personal_best(
        self, track_id: str, car_model: str
    ) -> Optional[dict]:
        """
        Busca a melhor volta histórica para a combinação track + car + piloto.

        Args:
            track_id: identificador da pista (ex: 'monza')
            car_model: modelo do carro (ex: 'ferrari_488_gt3')

        Returns:
            Dict com {lap_time_ms, session_date, mini_sectors: list[dict]}
            ou None se não houver histórico ou Supabase desabilitado.
        """
        if not self._client.is_enabled:
            return None

        try:
            # Busca personal best + lap_id
            pb_result = (
                self._client.get_client()
                .table("personal_bests")
                .select("lap_time_ms, lap_id, updated_at")
                .eq("user_id", SUPABASE_USER_ID)
                .eq("track_id", track_id)
                .eq("car_model", car_model)
                .limit(1)   # .single() lança 406 quando não há resultado — usar limit(1)
                .execute()
            )

            if not pb_result or not pb_result.data:
                logger.info(
                    "Nenhum personal best encontrado para %s / %s",
                    track_id,
                    car_model,
                )
                return None

            pb = pb_result.data[0]
            lap_id = pb["lap_id"]

            if not lap_id:
                return None

            # Busca mini-setores do PB ordenados por posição
            sectors_result = (
                self._client.get_client()
                .table("mini_sectors")
                .select(
                    "track_position, delta_vs_best, throttle, brake, steering, "
                    "gear, rpms, clutch, speed_kmh, speed_min, "
                    "gforce_x, gforce_y, gforce_z, "
                    "local_ang_vel_x, local_ang_vel_y, local_ang_vel_z, "
                    "wheel_slip_fl, wheel_slip_fr, wheel_slip_rl, wheel_slip_rr, "
                    "tc_active, abs_active, drs_active, drs_available, "
                    "brake_bias, surface_grip, air_temp, road_temp"
                )
                .eq("lap_id", lap_id)
                .order("track_position")
                .execute()
            )

            if not sectors_result or not sectors_result.data:
                logger.warning(
                    "Personal best encontrado mas sem mini-setores (lap_id=%s)", lap_id
                )
                return None

            logger.info(
                "Personal best carregado: %dms (%s / %s)",
                pb["lap_time_ms"],
                track_id,
                car_model,
                extra={"session_date": pb["updated_at"], "sectors": len(sectors_result.data)},
            )

            return {
                "lap_time_ms": pb["lap_time_ms"],
                "session_date": pb["updated_at"],
                "mini_sectors": sectors_result.data,  # list[dict] — formato compatível com LapAnalyzer
            }

        except Exception as exc:
            logger.warning(
                "Falha ao buscar personal best — usando referência da sessão atual",
                extra={"error": str(exc)},
            )
            return None

    def get_sector_history(
        self,
        track_id: str,
        car_model: str,
        track_position: float,
        tolerance: float = 0.005,
        last_n_sessions: int = 10,
    ) -> list[dict]:
        """
        Retorna histórico de delta_vs_best para um setor específico.

        Usado pela Fase 7D para indicadores de tendência no terminal.

        Args:
            track_id: pista
            car_model: carro
            track_position: posição na spline (centro do mini-setor)
            tolerance: margem de posição para considerar o mesmo setor
            last_n_sessions: número de sessões recentes a considerar

        Returns:
            Lista de {session_date, avg_delta, min_speed} por sessão,
            ordenada da mais recente para a mais antiga.
        """
        if not self._client.is_enabled:
            return []

        try:
            # Busca sessões recentes do piloto na pista/carro
            sessions_result = (
                self._client.get_client()
                .table("sessions")
                .select("id, started_at")
                .eq("user_id", SUPABASE_USER_ID)
                .eq("track_id", track_id)
                .eq("car_model", car_model)
                .order("started_at", desc=True)
                .limit(last_n_sessions)
                .execute()
            )

            if not sessions_result or not sessions_result.data:
                return []

            session_ids = [s["id"] for s in sessions_result.data]
            session_dates = {s["id"]: s["started_at"] for s in sessions_result.data}

            pos_min = track_position - tolerance
            pos_max = track_position + tolerance

            # Busca os laps dessas sessões primeiro (define o universo de lap_ids).
            # Antes a query de mini_sectors varria a tabela inteira sem filtro de
            # lap e truncava em 1000 linhas (db-max-rows) — perdia sessões e
            # misturava dados de outros pilotos.
            sb = self._client.get_client()
            laps_data = fetch_all_in(
                lambda ids: (
                    sb.table("laps").select("id, session_id")
                    .in_("session_id", ids).order("id")
                ),
                session_ids,
            )

            if not laps_data:
                return []

            lap_to_session = {l["id"]: l["session_id"] for l in laps_data}
            lap_ids = list(lap_to_session.keys())

            # Mini-setores na faixa de posição, restritos aos laps das sessões.
            # Chunked + paginado: 100 lap_ids podem render ~10k setores.
            sectors_data = fetch_all_in(
                lambda ids: (
                    sb.table("mini_sectors")
                    .select("lap_id, delta_vs_best, speed_min")
                    .in_("lap_id", ids)
                    .gte("track_position", pos_min)
                    .lte("track_position", pos_max)
                    .order("id")
                ),
                lap_ids,
            )

            if not sectors_data:
                return []

            # Agrega delta e speed_min por sessão
            session_data: dict[str, list] = {}
            for sector in sectors_data:
                sid = lap_to_session.get(sector["lap_id"])
                if sid and sector.get("delta_vs_best") is not None:
                    session_data.setdefault(sid, []).append(sector)

            history = []
            for sid, sectors in session_data.items():
                deltas = [s["delta_vs_best"] for s in sectors if s["delta_vs_best"] is not None]
                speeds = [s["speed_min"] for s in sectors if s.get("speed_min") is not None]
                history.append({
                    "session_date": session_dates.get(sid),
                    "avg_delta": sum(deltas) / len(deltas) if deltas else None,
                    "avg_speed_min": sum(speeds) / len(speeds) if speeds else None,
                })

            history.sort(key=lambda x: x["session_date"] or "", reverse=True)
            return history

        except Exception as exc:
            logger.warning(
                "Falha ao buscar histórico do setor",
                extra={"error": str(exc), "track_position": track_position},
            )
            return []

    def get_session_history(
        self,
        track_id: str,
        car_model: str,
        last_n_sessions: int = 10,
    ) -> list[dict]:
        """
        Retorna o histórico de sessões com o melhor tempo de cada uma.

        Args:
            track_id: pista
            car_model: carro
            last_n_sessions: número máximo de sessões a retornar

        Returns:
            Lista de {session_date, best_lap_ms} ordenada da mais recente
            para a mais antiga. Retorna [] se desabilitado ou sem dados.
        """
        if not self._client.is_enabled:
            return []

        try:
            sessions_result = (
                self._client.get_client()
                .table("sessions")
                .select("id, started_at")
                .eq("user_id", SUPABASE_USER_ID)
                .eq("track_id", track_id)
                .eq("car_model", car_model)
                .order("started_at", desc=True)
                .limit(last_n_sessions)
                .execute()
            )

            if not sessions_result or not sessions_result.data:
                return []

            history = []
            for session in sessions_result.data:
                best_lap_result = (
                    self._client.get_client()
                    .table("laps")
                    .select("lap_time_ms")
                    .eq("session_id", session["id"])
                    .eq("is_valid", True)
                    .order("lap_time_ms")
                    .limit(1)
                    .execute()
                )
                if best_lap_result and best_lap_result.data:
                    history.append({
                        "session_date": session["started_at"],
                        "best_lap_ms": best_lap_result.data[0]["lap_time_ms"],
                    })

            return history

        except Exception as exc:
            logger.warning(
                "Falha ao buscar histórico de sessões",
                extra={"error": str(exc)},
            )
            return []

    def get_pattern_frequency(
        self,
        track_id: str,
        car_model: str,
        last_n_sessions: int = 10,
    ) -> list[dict]:
        """
        Retorna os padrões de perda mais frequentes nas últimas N sessões.

        Args:
            track_id: pista
            car_model: carro
            last_n_sessions: número de sessões a considerar

        Returns:
            Lista de {cause, corner_name, count, percentage} ordenada por
            frequência decrescente.
        """
        if not self._client.is_enabled:
            return []

        try:
            # Busca sessões recentes
            sessions_result = (
                self._client.get_client()
                .table("sessions")
                .select("id")
                .eq("user_id", SUPABASE_USER_ID)
                .eq("track_id", track_id)
                .eq("car_model", car_model)
                .order("started_at", desc=True)
                .limit(last_n_sessions)
                .execute()
            )

            if not sessions_result or not sessions_result.data:
                return []

            session_ids = [s["id"] for s in sessions_result.data]
            sb = self._client.get_client()

            # Busca laps dessas sessões (chunked + paginado)
            laps_data = fetch_all_in(
                lambda ids: sb.table("laps").select("id").in_("session_id", ids).order("id"),
                session_ids,
            )

            if not laps_data:
                return []

            lap_ids = [l["id"] for l in laps_data]

            # Busca todos os padrões desses laps.
            # lap_ids pode ter centenas de UUIDs → .in_() direto estoura a URL
            # (HTTP 400). fetch_all_in fatia em blocos e pagina cada um.
            patterns_data = fetch_all_in(
                lambda ids: (
                    sb.table("lap_patterns")
                    .select("cause, corner_name, confidence")
                    .in_("lap_id", ids)
                    .order("id")
                ),
                lap_ids,
            )

            if not patterns_data:
                return []

            # Agrega por causa + curva
            freq: dict[tuple, dict] = {}
            for p in patterns_data:
                key = (p["cause"], p.get("corner_name") or "—")
                if key not in freq:
                    freq[key] = {"cause": p["cause"], "corner_name": key[1], "count": 0}
                freq[key]["count"] += 1

            total = sum(v["count"] for v in freq.values())
            result = []
            for entry in sorted(freq.values(), key=lambda x: x["count"], reverse=True):
                entry["percentage"] = round(entry["count"] / total, 2) if total > 0 else 0.0
                result.append(entry)

            return result

        except Exception as exc:
            logger.warning(
                "Falha ao buscar frequência de padrões",
                extra={"error": str(exc)},
            )
            return []

    def get_sectors_history_batch(
        self,
        track_id: str,
        car_model: str,
        positions: list[float],
        tolerance: float = 0.005,
        last_n_sessions: int = 10,
    ) -> dict[float, list[dict]]:
        """
        Busca histórico de delta para múltiplas posições em uma única chamada.

        Mais eficiente que chamar get_sector_history() N vezes em sequência.

        Args:
            track_id: pista
            car_model: carro
            positions: lista de track_positions a consultar
            tolerance: margem de posição para considerar o mesmo setor
            last_n_sessions: número de sessões a considerar

        Returns:
            Dict {track_position → list[{session_date, avg_delta, avg_speed_min}]}.
            Posições sem histórico não aparecem no dict.
        """
        if not self._client.is_enabled or not positions:
            return {}

        try:
            # Sessões recentes
            sessions_result = (
                self._client.get_client()
                .table("sessions")
                .select("id, started_at")
                .eq("user_id", SUPABASE_USER_ID)
                .eq("track_id", track_id)
                .eq("car_model", car_model)
                .order("started_at", desc=True)
                .limit(last_n_sessions)
                .execute()
            )

            if not sessions_result or not sessions_result.data:
                return {}

            session_ids = [s["id"] for s in sessions_result.data]
            session_dates = {s["id"]: s["started_at"] for s in sessions_result.data}
            sb = self._client.get_client()

            # Laps dessas sessões (chunked + paginado)
            laps_data = fetch_all_in(
                lambda ids: sb.table("laps").select("id, session_id").in_("session_id", ids).order("id"),
                session_ids,
            )

            if not laps_data:
                return {}

            lap_ids = [l["id"] for l in laps_data]
            lap_to_session = {l["id"]: l["session_id"] for l in laps_data}

            # Faixa global cobrindo todas as posições.
            # .in_(lap_ids) com centenas de UUIDs estourava a URL (HTTP 400) e o
            # resultado podia passar de 1000 linhas (db-max-rows) — fetch_all_in
            # fatia e pagina.
            pos_min = min(positions) - tolerance
            pos_max = max(positions) + tolerance

            sectors_data = fetch_all_in(
                lambda ids: (
                    sb.table("mini_sectors")
                    .select("lap_id, track_position, delta_vs_best, speed_min")
                    .in_("lap_id", ids)
                    .gte("track_position", pos_min)
                    .lte("track_position", pos_max)
                    .order("id")
                ),
                lap_ids,
            )

            if not sectors_data:
                return {}

            # Agrupa por posição alvo → sessão → lista de deltas
            pos_session_data: dict[float, dict[str, list]] = {p: {} for p in positions}

            for sector in sectors_data:
                # Encontra qual posição alvo este setor pertence
                sp = sector["track_position"]
                target = next(
                    (p for p in positions if abs(sp - p) <= tolerance), None
                )
                if target is None:
                    continue

                sid = lap_to_session.get(sector["lap_id"])
                if sid is None:
                    continue

                if sector.get("delta_vs_best") is None:
                    continue

                pos_session_data[target].setdefault(sid, []).append(sector)

            # Consolida em histórico por posição
            result: dict[float, list[dict]] = {}
            for pos, session_map in pos_session_data.items():
                history = []
                for sid, sectors in session_map.items():
                    deltas = [s["delta_vs_best"] for s in sectors]
                    speeds = [s["speed_min"] for s in sectors if s.get("speed_min")]
                    history.append({
                        "session_date": session_dates.get(sid),
                        "avg_delta": sum(deltas) / len(deltas) if deltas else None,
                        "avg_speed_min": sum(speeds) / len(speeds) if speeds else None,
                    })
                if history:
                    history.sort(
                        key=lambda x: x["session_date"] or "", reverse=True
                    )
                    result[pos] = history

            return result

        except Exception as exc:
            logger.warning(
                "Falha ao buscar histórico em batch",
                extra={"error": str(exc)},
            )
            return {}

    def update_session_lap_count(self, session_id: str, total_laps: int) -> None:
        """
        Atualiza total_laps na sessão ao encerrar.

        Args:
            session_id: UUID da sessão
            total_laps: número final de voltas completadas
        """
        if not self._client.is_enabled or not session_id:
            return

        try:
            self._client.get_client() \
                .table("sessions") \
                .update({"total_laps": total_laps}) \
                .eq("id", session_id) \
                .execute()
            logger.debug(
                "total_laps atualizado",
                extra={"session_id": session_id, "total_laps": total_laps},
            )
        except Exception as exc:
            logger.warning(
                "Falha ao atualizar total_laps",
                extra={"error": str(exc)},
            )
