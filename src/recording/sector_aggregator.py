"""
SectorAggregator — FASE 2

Agrega múltiplos snapshots brutos de telemetria em um único mini-setor
representativo, aplicando as funções de agregação adequadas por campo.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Campos que usam média simples
_MEAN_FIELDS = [
    "throttle", "brake", "steering", "clutch",
    "speed_kmh",
    "gforce_x", "gforce_y", "gforce_z",
    "local_ang_vel_x", "local_ang_vel_y", "local_ang_vel_z",
    "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
    "tc_active", "abs_active",
    "brake_bias", "surface_grip",
    "air_temp", "road_temp",
    "delta_vs_best",
]

# Campos que usam o valor mínimo (velocidade mínima no setor = indicador de ponto de freio)
_MIN_FIELDS = ["speed_kmh"]

# Campos que usam o valor modal (mais frequente — para inteiros)
_MODE_FIELDS = ["gear", "rpms", "drs_active", "drs_available"]

# Campos de posição que usam o valor central do buffer
_MIDPOINT_FIELDS = ["track_position"]

# Campos de validação que propagam para o mini-setor
_VALIDATION_FIELDS = [
    "_status", "_flag", "_number_of_tyres_out", "_pit_limiter_on",
    "_is_in_pit", "_is_in_pit_lane", "_penalty_time", "_car_damage_max",
    "_is_ai_controlled", "_i_current_time_ms", "_i_best_time_ms",
    "_i_last_time_ms", "_last_sector_time_ms", "_current_sector_index",
    "_completed_laps",
]


class SectorAggregator:
    """
    Agrega uma lista de snapshots brutos em um único mini-setor.

    Cada campo usa a função de agregação mais semanticamente adequada:
    - média: inputs contínuos do piloto e condições
    - mínimo: speed_kmh (velocidade mínima é o ponto de freio)
    - moda: campos discretos (marcha, DRS)
    - midpoint: posição na pista (usa o valor central do buffer)
    """

    def aggregate(self, snapshots: list[dict]) -> Optional[dict]:
        """
        Agrega uma lista de snapshots em um único dict de mini-setor.

        Args:
            snapshots: lista de dicts retornados por snapshot_to_dict()

        Returns:
            Dict do mini-setor agregado, ou None se snapshots estiver vazio.
        """
        if not snapshots:
            return None

        result: dict = {}

        # Posição: valor central do buffer
        positions = [s["track_position"] for s in snapshots]
        result["track_position"] = positions[len(positions) // 2]

        # Campos de média
        for field in _MEAN_FIELDS:
            values = [s[field] for s in snapshots if field in s]
            if values:
                result[field] = sum(values) / len(values)

        # Velocidade mínima (ponto de freio no setor)
        speed_values = [s["speed_kmh"] for s in snapshots if "speed_kmh" in s]
        if speed_values:
            result["speed_min"] = min(speed_values)

        # Campos modais (inteiros)
        for field in _MODE_FIELDS:
            values = [s[field] for s in snapshots if field in s]
            if values:
                result[field] = self._mode(values)

        # Campos de validação: último snapshot (mais recente)
        last = snapshots[-1]
        for field in _VALIDATION_FIELDS:
            if field in last:
                result[field] = last[field]

        return result

    @staticmethod
    def _mode(values: list) -> int:
        """Retorna o valor mais frequente (moda) de uma lista."""
        return max(set(values), key=values.count)
