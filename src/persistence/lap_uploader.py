"""
lap_uploader.py — Upload assíncrono de voltas para o Supabase.

Responsabilidades:
    - create_session(): INSERT síncrono de uma sessão (chamado uma vez, pré-loop)
    - enqueue_lap(): coloca payload na fila e retorna imediatamente (não bloqueia o loop 20Hz)
    - Daemon thread consome a fila em background
    - Retry automático: 3 tentativas com backoff exponencial (1s, 3s, 9s)
    - Falha após 3 tentativas: loga WARNING, não interrompe a sessão
    - stop(): drena a fila com timeout de 10s antes de encerrar

Fluxo de upload por volta:
    1. INSERT INTO laps → obtém lap_id
    2. INSERT INTO mini_sectors (bulk — 1 chamada para ~100 linhas)
    3. INSERT INTO lap_patterns (bulk — padrões do LapReport)
    4. UPSERT INTO personal_bests (se nova melhor volta da combinação track+car)
    5. UPDATE sessions SET total_laps = total_laps + 1
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import SUPABASE_USER_ID
from src.persistence.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 3.0, 9.0)


@dataclass
class _LapPayload:
    """Payload completo de uma volta para upload."""
    session_id: str
    lap_data: dict       # retorno do LapRecorder (mini_sectors, lap_time_ms, etc.)
    lap_report: object   # LapReport do ReportBuilder (top_sectors com causes)
    is_session_best: bool
    is_alltime_best: bool


class LapUploader:
    """
    Gerencia o upload assíncrono de voltas para o Supabase.

    Quando o SupabaseClient está desabilitado, todos os métodos são no-ops
    e o projeto funciona identicamente ao estado pré-Fase 7.
    """

    def __init__(self, supabase_client: SupabaseClient) -> None:
        self._client = supabase_client
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._session_id: Optional[str] = None

        if self._client.is_enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="LapUploaderWorker",
                daemon=True,
            )
            self._thread.start()
            logger.debug("LapUploader daemon thread iniciada")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def create_session(
        self,
        track_id: str,
        car_model: str,
        session_type: str,
        air_temp: float,
        road_temp: float,
    ) -> Optional[str]:
        """
        Cria registro de sessão no Supabase (chamada síncrona, pré-loop).

        Args:
            track_id: identificador da pista
            car_model: modelo do carro lido da SPageFileStatic
            session_type: tipo de sessão ('practice', 'qualifying', etc.)
            air_temp: temperatura do ar no início da sessão
            road_temp: temperatura da pista no início da sessão

        Returns:
            session_id (UUID str) ou None se desabilitado/falha.
        """
        if not self._client.is_enabled:
            return None

        payload = {
            "user_id": SUPABASE_USER_ID,
            "track_id": track_id,
            "car_model": car_model or "unknown",
            "session_type": session_type or "practice",
            "air_temp": air_temp,
            "road_temp": road_temp,
            "total_laps": 0,
        }

        result = self._execute_with_retry(
            lambda: self._client.get_client()
                .table("sessions")
                .insert(payload)
                .execute()
        )

        if result and result.data:
            self._session_id = result.data[0]["id"]
            logger.info("Sessão criada no Supabase", extra={"session_id": self._session_id})
            return self._session_id

        logger.warning("Falha ao criar sessão no Supabase — continuando sem persistência")
        return None

    def enqueue_lap(
        self,
        completed_lap: dict,
        lap_report,
        is_session_best: bool = False,
        is_alltime_best: bool = False,
    ) -> None:
        """
        Coloca volta na fila de upload (retorna imediatamente — não bloqueia).

        Args:
            completed_lap: dict retornado pelo LapRecorder com mini_sectors
            lap_report: LapReport do ReportBuilder com top_sectors e patterns
            is_session_best: se esta volta é a melhor da sessão atual
            is_alltime_best: se esta volta é o novo personal best histórico
        """
        if not self._client.is_enabled or self._session_id is None:
            return

        payload = _LapPayload(
            session_id=self._session_id,
            lap_data=completed_lap,
            lap_report=lap_report,
            is_session_best=is_session_best,
            is_alltime_best=is_alltime_best,
        )
        self._queue.put(payload)
        logger.debug(
            "Volta enfileirada para upload",
            extra={"lap_number": completed_lap.get("lap_number")},
        )

    def stop(self, timeout_s: float = 10.0) -> None:
        """
        Sinaliza encerramento e aguarda a fila drenar.

        Args:
            timeout_s: segundos máximos de espera antes de forçar encerramento.
        """
        if not self._client.is_enabled or self._thread is None:
            return

        self._queue.put(None)  # sentinela de encerramento

        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            logger.warning(
                "LapUploader não drenou a fila no tempo limite — "
                "dados pendentes estão salvos no JSON local."
            )
        else:
            logger.info("LapUploader encerrado com sucesso")

    # ------------------------------------------------------------------
    # Worker (thread background)
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Loop da daemon thread — consome a fila e faz upload para o Supabase."""
        while True:
            payload = self._queue.get()
            if payload is None:  # sentinela de encerramento
                self._queue.task_done()
                break

            try:
                self._upload_lap(payload)
            except Exception as exc:
                logger.warning(
                    "Erro inesperado no worker de upload — volta ignorada",
                    extra={"error": str(exc)},
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Upload de volta
    # ------------------------------------------------------------------

    def _upload_lap(self, payload: _LapPayload) -> None:
        """Executa o pipeline completo de upload de uma volta."""
        lap_data = payload.lap_data
        lap_report = payload.lap_report

        # 1. INSERT laps
        lap_row = {
            "session_id": payload.session_id,
            "lap_number": lap_data.get("lap_number", 0),
            "lap_time_ms": lap_data.get("lap_time_ms", 0),
            "is_valid": True,
            "is_session_best": payload.is_session_best,
            "is_alltime_best": payload.is_alltime_best,
            "total_time_lost_s": getattr(lap_report, "total_time_lost_s", None),
            "tyre_compound": lap_data.get("tyre_compound", "unknown"),
        }

        result = self._execute_with_retry(
            lambda: self._client.get_client()
                .table("laps")
                .insert(lap_row)
                .execute()
        )

        if not result or not result.data:
            logger.warning(
                "Falha ao inserir volta no Supabase",
                extra={"lap_number": lap_data.get("lap_number")},
            )
            return

        lap_id = result.data[0]["id"]
        logger.debug("Volta inserida", extra={"lap_id": lap_id})

        # 2. INSERT mini_sectors (bulk)
        self._upload_mini_sectors(lap_id, lap_data.get("mini_sectors", []))

        # 3. INSERT lap_patterns (bulk, a partir do LapReport)
        self._upload_patterns(lap_id, lap_report)

        # 4. UPSERT personal_bests (se nova melhor volta)
        if payload.is_alltime_best:
            self._upsert_personal_best(
                lap_id=lap_id,
                lap_data=lap_data,
                session_id=payload.session_id,
            )

        # 5. UPDATE sessions.total_laps
        self._increment_session_laps(payload.session_id)

        logger.info(
            "Volta enviada ao Supabase",
            extra={
                "lap_id": lap_id,
                "lap_number": lap_data.get("lap_number"),
                "lap_time_ms": lap_data.get("lap_time_ms"),
            },
        )

    def _upload_mini_sectors(self, lap_id: str, mini_sectors: list[dict]) -> None:
        """Bulk insert de mini-setores (1 chamada à API para toda a lista)."""
        if not mini_sectors:
            return

        rows = [
            {
                "lap_id": lap_id,
                "track_position": s.get("track_position"),
                "delta_vs_best": s.get("delta_vs_best"),
                "delta_per_sector": s.get("delta_per_sector"),  # None para dados históricos
                "throttle": s.get("throttle"),
                "brake": s.get("brake"),
                "steering": s.get("steering"),
                "gear": s.get("gear"),
                "rpms": s.get("rpms"),
                "clutch": s.get("clutch"),
                "speed_kmh": s.get("speed_kmh"),
                "speed_min": s.get("speed_min"),
                "gforce_x": s.get("gforce_x"),
                "gforce_y": s.get("gforce_y"),
                "gforce_z": s.get("gforce_z"),
                "local_ang_vel_x": s.get("local_ang_vel_x"),
                "local_ang_vel_y": s.get("local_ang_vel_y"),
                "local_ang_vel_z": s.get("local_ang_vel_z"),
                "wheel_slip_fl": s.get("wheel_slip_fl"),
                "wheel_slip_fr": s.get("wheel_slip_fr"),
                "wheel_slip_rl": s.get("wheel_slip_rl"),
                "wheel_slip_rr": s.get("wheel_slip_rr"),
                "tc_active": s.get("tc_active"),
                "abs_active": s.get("abs_active"),
                "drs_active": s.get("drs_active"),
                "drs_available": s.get("drs_available"),
                "brake_bias": s.get("brake_bias"),
                "surface_grip": s.get("surface_grip"),
                "air_temp": s.get("air_temp"),
                "road_temp": s.get("road_temp"),
            }
            for s in mini_sectors
        ]

        result = self._execute_with_retry(
            lambda: self._client.get_client()
                .table("mini_sectors")
                .insert(rows)
                .execute()
        )

        if result:
            logger.debug(
                "Mini-setores inseridos",
                extra={"lap_id": lap_id, "count": len(rows)},
            )
        else:
            logger.warning(
                "Falha ao inserir mini-setores no Supabase — dados preservados no JSON local",
                extra={"lap_id": lap_id, "count": len(rows)},
            )

    def _upload_patterns(self, lap_id: str, lap_report) -> None:
        """Bulk insert de padrões detectados a partir do LapReport."""
        if lap_report is None or not hasattr(lap_report, "top_sectors"):
            return

        rows = []
        for sector in lap_report.top_sectors:
            for match in getattr(sector, "causes", []):
                rows.append({
                    "lap_id": lap_id,
                    "track_position": sector.track_position,
                    "cause": match.cause,
                    "confidence": match.confidence,
                    "evidence": match.evidence,  # dict → jsonb
                    "corner_name": sector.corner_name,
                    "corner_type": sector.corner_type,
                })

        if not rows:
            return

        self._execute_with_retry(
            lambda: self._client.get_client()
                .table("lap_patterns")
                .insert(rows)
                .execute()
        )
        logger.debug("Padrões inseridos", extra={"lap_id": lap_id, "count": len(rows)})

    def _upsert_personal_best(
        self, lap_id: str, lap_data: dict, session_id: str
    ) -> None:
        """UPSERT em personal_bests quando nova melhor volta é registrada."""
        # Busca car_model e track_id do registro de sessão
        result = self._execute_with_retry(
            lambda: self._client.get_client()
                .table("sessions")
                .select("track_id, car_model")
                .eq("id", session_id)
                .single()
                .execute()
        )

        if not result or not result.data:
            return

        session_data = result.data
        self._execute_with_retry(
            lambda: self._client.get_client()
                .table("personal_bests")
                .upsert({
                    "user_id": SUPABASE_USER_ID,
                    "track_id": session_data["track_id"],
                    "car_model": session_data["car_model"],
                    "lap_time_ms": lap_data.get("lap_time_ms", 0),
                    "lap_id": lap_id,
                })
                .execute()
        )
        logger.info(
            "Personal best atualizado no Supabase",
            extra={"lap_time_ms": lap_data.get("lap_time_ms")},
        )

    def _increment_session_laps(self, session_id: str) -> None:
        """Incrementa sessions.total_laps via RPC para evitar race condition."""
        try:
            self._client.get_client().rpc(
                "increment_session_laps",
                {"p_session_id": session_id},
            ).execute()
        except Exception:
            # RPC pode não existir — fallback via SELECT + UPDATE
            try:
                res = self._client.get_client() \
                    .table("sessions") \
                    .select("total_laps") \
                    .eq("id", session_id) \
                    .single() \
                    .execute()
                if res and res.data:
                    new_count = res.data["total_laps"] + 1
                    self._client.get_client() \
                        .table("sessions") \
                        .update({"total_laps": new_count}) \
                        .eq("id", session_id) \
                        .execute()
            except Exception as exc:
                logger.debug(
                    "Falha ao incrementar total_laps",
                    extra={"error": str(exc)},
                )

    # ------------------------------------------------------------------
    # Retry com backoff exponencial
    # ------------------------------------------------------------------

    def _execute_with_retry(self, fn):
        """
        Executa fn() com até _MAX_RETRIES tentativas e backoff exponencial.

        Faz exatamente _MAX_RETRIES tentativas (não _MAX_RETRIES + 1).
        O sleep ocorre apenas entre tentativas — nunca após a última.

        Args:
            fn: callable sem argumentos que chama a API do Supabase

        Returns:
            Resultado da chamada ou None após todas as tentativas falharem.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    wait_s = _RETRY_BACKOFF_S[attempt - 1]
                    logger.warning(
                        "Tentativa %d/%d falhou — aguardando %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        wait_s,
                        extra={"error": str(exc)},
                    )
                    time.sleep(wait_s)
                else:
                    logger.warning(
                        "Todas as %d tentativas falharam — dado preservado no JSON local",
                        _MAX_RETRIES,
                        extra={"error": str(exc)},
                    )
                    return None
