"""
LapRecorder — FASE 2

Grava cada volta em mini-setores (~0.01 na spline normalizada) e persiste
o resultado em JSON ao cruzar a linha de chegada.

Critérios de aceite (CLAUDE.md §6 FASE 2):
- [x] Detecção de início/fim de volta via normalizedCarPosition
- [x] Agrupamento em mini-setores de ~0.01
- [x] JSON gravado automaticamente ao cruzar a linha de chegada
- [x] Schema idêntico ao definido em §4.5
- [x] Descarte de voltas com carDamage > threshold ou isInPit = 1
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from config.settings import (
    CAR_DAMAGE_THRESHOLD,
    CLUTCH_MAX_VALUE,
    LAPS_DIR,
    MAX_TYRES_OUT_ALLOWED,
    MINI_SECTOR_SIZE,
    MIN_SECTORS_PER_LAP,
)
from src.memory.graphics_page import AC_LIVE
from src.recording.sector_aggregator import SectorAggregator

logger = logging.getLogger(__name__)

# Posição na spline que define a linha de chegada
_FINISH_LINE_POSITION: float = 0.0
# Histerese para não detectar o mesmo cruzamento duas vezes
_FINISH_LINE_HYSTERESIS: float = 0.05
# Salto negativo na spline acima deste limiar indica teleporte (return to pits,
# restart de sessão, troca de carro). O detector de cruzamento de linha de chegada
# não captura esses casos porque exige histerese estrita (>0.95 → <0.05).
# Threshold baixo é seguro: a única transição negativa legítima é o cruzamento
# normal da linha de chegada, filtrado por _detect_lap_start() antes desta checagem.
# Valor escolhido (0.1 = 10% da pista) tolera jitter de leitura mas captura
# teleportes a partir de qualquer ponto da pista.
_SPLINE_TELEPORT_THRESHOLD: float = 0.1


class LapRecorder:
    """
    Grava telemetria de mini-setor por mini-setor durante uma sessão.

    Estado interno:
        - _current_lap: lista de snapshots do mini-setor em curso
        - _sector_buffer: snapshots brutos do mini-setor atual
        - _last_position: última posição spline para detectar cruzamento
        - _lap_invalid: se True, a volta atual deve ser descartada
    """

    def __init__(
        self,
        track_id: str = "unknown",
        car_model: str = "unknown",
        session_type: str = "practice",
        on_lap_invalidated: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._track_id = track_id
        self._car_model = car_model
        self._session_type = session_type
        # Callback opcional invocado uma vez quando a volta em curso é
        # invalidada (recebe a razão). Usado para sinal de TTS — sem acoplar
        # o recorder à camada de saída.
        self._on_lap_invalidated = on_lap_invalidated
        self._aggregator = SectorAggregator()
        self._current_lap: list[dict] = []
        self._sector_buffer: list[dict] = []
        self._last_position: float = -1.0
        self._lap_invalid: bool = False
        self._lap_number: int = 0
        self._session_start_ts: int = int(time.time())
        self._current_tyre_compound: str = "unknown"

        LAPS_DIR.mkdir(parents=True, exist_ok=True)

    def process_snapshot(self, snapshot_dict: dict) -> Optional[dict]:
        """
        Processa um snapshot de telemetria e grava a volta ao completar.

        Args:
            snapshot_dict: dicionário retornado por snapshot_to_dict()

        Returns:
            Dados da volta completa (dict) se a volta foi completada e
            válida, None caso contrário.
        """
        if snapshot_dict.get("_status") != AC_LIVE:
            return None

        # Atualiza composto de pneu a cada snapshot (muda após pit stop)
        tyre_compound = snapshot_dict.get("_tyre_compound")
        if tyre_compound:
            self._current_tyre_compound = tyre_compound

        position = snapshot_dict["track_position"]

        # --- Detecção de teleporte / restart de sessão (Proposta A) ---
        # Saltos negativos grandes na spline indicam "Return to pits", restart
        # de sessão ou troca de carro. O detector de cruzamento de linha de
        # chegada (que exige _last_position > 0.95) não captura esses casos,
        # deixando o flag _lap_invalid preso indefinidamente.
        if (
            self._last_position >= 0
            and (self._last_position - position) > _SPLINE_TELEPORT_THRESHOLD
            and not self._detect_lap_start(position)
        ):
            logger.info(
                "Reset de spline detectado — descartando volta em curso",
                extra={
                    "lap_number": self._lap_number,
                    "last_position": round(self._last_position, 4),
                    "current_position": round(position, 4),
                },
            )
            self._reset_lap()
            self._last_position = position
            return None

        # Detectar início de volta (cruzamento da linha de chegada)
        if self._detect_lap_start(position):
            completed_lap = self._finalize_lap()
            self._reset_lap()
            if completed_lap:
                self._lap_number += 1
                self._save_lap(completed_lap)
                return completed_lap

        # Validar snapshot antes de adicionar ao buffer
        invalid_reason = self._snapshot_invalid_reason(snapshot_dict)
        if invalid_reason is not None:
            if not self._lap_invalid:
                # Loga apenas a primeira invalidação da volta para não inundar
                # o log a 20Hz quando múltiplos snapshots seguidos são inválidos.
                logger.info(
                    "Volta marcada como inválida",
                    extra={
                        "lap_number": self._lap_number,
                        "reason": invalid_reason,
                        "track_position": round(position, 4),
                    },
                )
                # Sinal único de invalidação (ex.: TTS). Protegido para que
                # uma falha no callback nunca interrompa a gravação.
                if self._on_lap_invalidated is not None:
                    try:
                        self._on_lap_invalidated(invalid_reason)
                    except Exception as exc:
                        logger.warning(
                            "Callback on_lap_invalidated falhou",
                            extra={"error": str(exc)},
                        )
            self._lap_invalid = True

        # Agrupar snapshot no mini-setor correto
        self._sector_buffer.append(snapshot_dict)

        # Verificar se o mini-setor atual está completo
        sector_index = int(position / MINI_SECTOR_SIZE)
        last_sector_index = int(self._last_position / MINI_SECTOR_SIZE) if self._last_position >= 0 else -1

        if sector_index != last_sector_index and self._sector_buffer:
            aggregated = self._aggregator.aggregate(self._sector_buffer)
            if aggregated:
                self._current_lap.append(aggregated)
            self._sector_buffer = []

        self._last_position = position
        return None

    # ------------------------------------------------------------------
    # Detecção de cruzamento da linha de chegada
    # ------------------------------------------------------------------

    def _detect_lap_start(self, position: float) -> bool:
        """
        Detecta cruzamento da linha de chegada (posição ≈ 0.0).

        Usa histerese para evitar falsos positivos: só detecta quando
        a posição anterior estava na região de chegada (> 1 - hysteresis)
        e a posição atual está na região de saída (< hysteresis).
        """
        if self._last_position < 0:
            return False

        crossed = (
            self._last_position > (1.0 - _FINISH_LINE_HYSTERESIS)
            and position < _FINISH_LINE_HYSTERESIS
        )
        return crossed

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_invalid_reason(snapshot: dict) -> Optional[str]:
        """Retorna a razão de invalidez do snapshot, ou None se válido (§4.2).

        Inclui detecção de clutch corrompido: o campo deve ser 0.0–1.0 conforme
        o AC SDK. Valores acima de CLUTCH_MAX_VALUE indicam offset errado na leitura
        da Shared Memory (bug identificado no relatório de validação 2026-04-08).
        """
        if snapshot.get("_is_in_pit") == 1:
            return "is_in_pit"
        if snapshot.get("_is_in_pit_lane") == 1:
            return "is_in_pit_lane"
        if snapshot.get("_pit_limiter_on") == 1:
            return "pit_limiter_on"
        # Pneus fora da pista — só invalida com MAIS de MAX_TYRES_OUT_ALLOWED
        # pneus fora (padrão 2 → invalida a partir de 3). CLAUDE.md §4.2.
        tyres_out = snapshot.get("_number_of_tyres_out", 0)
        if tyres_out > MAX_TYRES_OUT_ALLOWED:
            return f"tyres_out={tyres_out}"
        damage = snapshot.get("_car_damage_max", 0.0)
        if damage > CAR_DAMAGE_THRESHOLD:
            return f"car_damage={damage:.3f}>{CAR_DAMAGE_THRESHOLD}"
        penalty = snapshot.get("_penalty_time", 0.0)
        if penalty > 0.0:
            return f"penalty_time={penalty:.2f}"
        # Clutch fora do range 0–1 indica leitura corrompida da Shared Memory
        clutch = snapshot.get("clutch", 0.0)
        if clutch > CLUTCH_MAX_VALUE:
            return f"clutch_corrupted={clutch:.2f}>{CLUTCH_MAX_VALUE}"
        return None

    @staticmethod
    def _is_snapshot_invalid(snapshot: dict) -> bool:
        """Wrapper booleano de _snapshot_invalid_reason (mantido por compatibilidade)."""
        return LapRecorder._snapshot_invalid_reason(snapshot) is not None

    # ------------------------------------------------------------------
    # Ciclo de vida da volta
    # ------------------------------------------------------------------

    def _finalize_lap(self) -> Optional[dict]:
        """
        Agrega o buffer final e retorna os dados da volta completa.

        Retorna None se a volta for inválida ou tiver poucos mini-setores.
        """
        # Processar qualquer buffer restante
        if self._sector_buffer:
            aggregated = self._aggregator.aggregate(self._sector_buffer)
            if aggregated:
                self._current_lap.append(aggregated)
            self._sector_buffer = []

        if self._lap_invalid:
            logger.info(
                "Volta descartada — violação de regra de validade",
                extra={"lap_number": self._lap_number, "sectors": len(self._current_lap)},
            )
            return None

        if not self._current_lap:
            return None

        if len(self._current_lap) < MIN_SECTORS_PER_LAP:
            logger.info(
                "Volta descartada — mini-setores insuficientes",
                extra={
                    "lap_number": self._lap_number,
                    "sectors": len(self._current_lap),
                    "min_required": MIN_SECTORS_PER_LAP,
                },
            )
            return None

        # Calcular tempo da volta a partir dos snapshots
        first = self._current_lap[0]
        last = self._current_lap[-1]
        lap_time_ms = last.get("_i_current_time_ms", 0)

        lap_data = {
            "lap_number": self._lap_number,
            "track_id": self._track_id,
            "car_model": self._car_model,
            "session_type": self._session_type,
            "tyre_compound": self._current_tyre_compound,
            "lap_time_ms": lap_time_ms,
            "sector_count": len(self._current_lap),
            "mini_sectors": self._current_lap,
        }

        logger.info(
            "Volta completada",
            extra={
                "lap_number": self._lap_number,
                "lap_time_ms": lap_time_ms,
                "sector_count": len(self._current_lap),
            },
        )
        return lap_data

    def _reset_lap(self) -> None:
        """Reinicia o estado para a próxima volta."""
        self._current_lap = []
        self._sector_buffer = []
        self._lap_invalid = False

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _save_lap(self, lap_data: dict) -> Path:
        """
        Persiste a volta em JSON no diretório LAPS_DIR.

        Nomenclatura: {track_id}_{session_ts}_lap{lap_number:03d}.json
        """
        filename = f"{self._track_id}_{self._session_start_ts}_lap{self._lap_number:03d}.json"
        filepath = LAPS_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(lap_data, f, indent=2)

        logger.info("Volta gravada", extra={"path": str(filepath)})
        return filepath
