"""
SectorAggregator — FASE 2

Agrega múltiplos snapshots brutos de telemetria em um único mini-setor
representativo, aplicando as funções de agregação adequadas por campo.

Campos computados (não presentes na Shared Memory, calculados aqui):
    delta_per_sector: variação de delta_vs_best entre o primeiro e último
                      snapshot do mini-setor. Representa a perda de tempo
                      ocorrida DENTRO deste mini-setor específico — é o
                      target correto para o SectorModel (não o delta
                      acumulado desde o início da volta).

    {input}_max, {input}_min, {input}_std (2026-04-25, Proposal P1):
        Para os 9 campos de input do piloto e sistemas ativos, o valor
        médio sozinho destrói a dinâmica intra-setor (~22 snapshots/1.1s
        @20Hz). Threshold-braking de 1.0 por 0.3s e freio constante de
        0.27 por 1.1s produzem a MESMA média (~0.27).
        Capturar peak/valley/variability é prática padrão em telemetria
        motorsport — ver Segers, J. (2014) Analysis Techniques for
        Racecar Data Acquisition §4.
"""

import logging
import statistics
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
    "delta_vs_best",  # mantido para compatibilidade retroativa e debug
]

# Inputs do piloto e sistemas ativos: ALÉM da média (já em _MEAN_FIELDS),
# também emitem _max, _min, _std para preservar a dinâmica intra-setor.
# 9 campos × 3 estatísticas = 27 novas colunas no schema §4.5.
_MULTI_STAT_INPUT_FIELDS = [
    "throttle", "brake", "steering",
    "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
    "tc_active", "abs_active",
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

        # Estatísticas multi-stat para inputs do piloto (Proposal P1):
        # peak/valley/variability são informativas onde a média não é.
        for field in _MULTI_STAT_INPUT_FIELDS:
            values = [s[field] for s in snapshots if field in s]
            if values:
                result[f"{field}_max"] = max(values)
                result[f"{field}_min"] = min(values)
                result[f"{field}_std"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )

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

        # ------------------------------------------------------------------
        # Campo computado: delta_per_sector
        #
        # Diferença entre o delta_vs_best do último e do primeiro snapshot
        # dentro deste buffer. Captura exatamente quanto tempo foi ganho ou
        # perdido durante a passagem por este mini-setor.
        #
        # Por que não usar a média (já em _MEAN_FIELDS)?
        #   A média do delta_vs_best dentro do setor representa o delta
        #   "típico" durante a passagem — mas o que importa para o modelo é
        #   a MUDANÇA: o piloto ficou mais próximo ou mais distante do
        #   melhor tempo enquanto percorria este 1% da pista?
        #
        # Nota: snapshots com delta_vs_best ausente ou inválido (ex: primeira
        # volta sem referência → performanceMeter = -inf) resultam em 0.0.
        # O filtro _DELTA_OUTLIER_THRESHOLD_S no SectorModel descarta outliers.
        # ------------------------------------------------------------------
        dvb_values = [
            s["delta_vs_best"]
            for s in snapshots
            if "delta_vs_best" in s and s["delta_vs_best"] is not None
        ]
        if len(dvb_values) >= 2:
            result["delta_per_sector"] = dvb_values[-1] - dvb_values[0]
        elif len(dvb_values) == 1:
            result["delta_per_sector"] = 0.0
        # Se nenhum snapshot tem delta_vs_best, delta_per_sector é omitido
        # (o SectorModel vai ignorar este setor como sem target válido).

        return result

    @staticmethod
    def _mode(values: list) -> int:
        """Retorna o valor mais frequente (moda) de uma lista."""
        return max(set(values), key=values.count)
