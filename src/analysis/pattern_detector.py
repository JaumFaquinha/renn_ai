"""
PatternDetector — FASE 4

Correlaciona padrões de telemetria com causas específicas de perda de tempo.

Implementa os 5 padrões definidos em CLAUDE.md §5:
1. Frenagem tardia com bloqueio (brake alto + abs ativo + speed_min baixo)
2. Aceleração precoce/agressiva (throttle baixo + tc ativo + wheel_slip alto)
3. Entrada de curva rápida demais (gforce_x alto + steering alto + speed alto)
4. Ponto de troca subótimo (gear incorreta + RPM fora do range)
5. Saída de curva comprometida (throttle parcial em reta longa)

Critérios de aceite (CLAUDE.md §6 FASE 4):
- [x] Todos os 5 padrões implementados
- [x] Cada setor com perda retorna ao menos uma causa identificada
- [x] Confiança expressada como float (0.0–1.0)
- [x] Casos ambíguos retornam lista ordenada por confiança
"""

import logging
from dataclasses import dataclass, field

from config.settings import (
    ABS_ACTIVE_THRESHOLD,
    GFORCE_LATERAL_THRESHOLD,
    RPM_SHIFT_MARGIN,
    STEERING_THRESHOLD,
    TC_ACTIVE_THRESHOLD,
    WHEEL_SLIP_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Thresholds de perda mínima para considerar o setor relevante (segundos)
_MIN_DELTA_TO_ANALYZE: float = 0.05


@dataclass
class PatternMatch:
    """Resultado de detecção de um padrão em um mini-setor."""

    cause: str          # Descrição da causa identificada
    confidence: float   # Confiança da detecção (0.0–1.0)
    evidence: dict = field(default_factory=dict)  # Valores de telemetria que dispararam o padrão


class PatternDetector:
    """
    Detecta padrões de causa de perda de tempo em mini-setores.

    Cada método _detect_* retorna um PatternMatch ou None.
    O método detect() consolida todos os padrões e ordena por confiança.
    """

    def __init__(self, max_rpm: int = 8000) -> None:
        """
        Args:
            max_rpm: RPM máximo do carro da sessão (de SPageFileStatic.maxRpm).
                     Usado para calcular a faixa ótima de troca de marcha.
        """
        self._max_rpm = max_rpm

    def detect(self, sector: dict) -> list[PatternMatch]:
        """
        Aplica todos os detectores de padrão a um mini-setor.

        Args:
            sector: mini-setor enriquecido retornado por LapAnalyzer.analyze()

        Returns:
            Lista de PatternMatch ordenada por confiança decrescente.
            Pode estar vazia se nenhum padrão for detectado.
        """
        delta = sector.get("delta_vs_best", 0.0)
        if delta < _MIN_DELTA_TO_ANALYZE:
            return []

        detectors = [
            self._detect_late_braking,
            self._detect_early_throttle,
            self._detect_fast_corner_entry,
            self._detect_suboptimal_shift,
            self._detect_compromised_exit,
        ]

        matches = []
        for detector in detectors:
            match = detector(sector)
            if match is not None:
                matches.append(match)

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    # ------------------------------------------------------------------
    # Detectores de Padrão
    # ------------------------------------------------------------------

    def _detect_late_braking(self, sector: dict) -> PatternMatch | None:
        """
        Padrão 1: Frenagem tardia com bloqueio.

        Sinais: brake alto + abs ativo + speed_min baixo.
        """
        brake = sector.get("brake", 0.0)
        abs_active = sector.get("abs_active", 0.0)

        if brake < 0.7:
            return None
        if abs_active < ABS_ACTIVE_THRESHOLD:
            return None

        confidence = min(1.0, (brake * 0.5) + (abs_active / ABS_ACTIVE_THRESHOLD * 0.5))

        return PatternMatch(
            cause="Frenagem tardia com bloqueio de rodas",
            confidence=round(confidence, 3),
            evidence={
                "brake": brake,
                "abs_active": abs_active,
                "speed_min_kmh": sector.get("speed_min", 0.0),
            },
        )

    def _detect_early_throttle(self, sector: dict) -> PatternMatch | None:
        """
        Padrão 2: Aceleração precoce/agressiva.

        Sinais: throttle baixo na saída + tc ativo + wheel_slip alto.
        """
        throttle = sector.get("throttle", 0.0)
        tc_active = sector.get("tc_active", 0.0)
        avg_slip = (
            sector.get("wheel_slip_rl", 0.0) + sector.get("wheel_slip_rr", 0.0)
        ) / 2.0

        if tc_active < TC_ACTIVE_THRESHOLD:
            return None
        if avg_slip < WHEEL_SLIP_THRESHOLD:
            return None

        confidence = min(1.0, tc_active * 0.6 + (avg_slip / WHEEL_SLIP_THRESHOLD) * 0.4)

        return PatternMatch(
            cause="Aceleração precoce ou agressiva — TC interveio",
            confidence=round(confidence, 3),
            evidence={
                "throttle": throttle,
                "tc_active": tc_active,
                "avg_rear_slip": round(avg_slip, 4),
            },
        )

    def _detect_fast_corner_entry(self, sector: dict) -> PatternMatch | None:
        """
        Padrão 3: Entrada de curva rápida demais.

        Sinais: gforce_x alto + steering alto + speed alto.
        """
        gforce_x = abs(sector.get("gforce_x", 0.0))
        steering = abs(sector.get("steering", 0.0))
        speed = sector.get("speed_kmh", 0.0)

        if gforce_x < GFORCE_LATERAL_THRESHOLD:
            return None
        if steering < STEERING_THRESHOLD:
            return None

        confidence = min(
            1.0,
            (gforce_x / GFORCE_LATERAL_THRESHOLD * 0.5)
            + (steering / STEERING_THRESHOLD * 0.3)
            + (min(speed, 200.0) / 200.0 * 0.2),
        )

        return PatternMatch(
            cause="Velocidade de entrada na curva elevada — excesso de G lateral",
            confidence=round(confidence, 3),
            evidence={
                "gforce_x": gforce_x,
                "steering": steering,
                "speed_kmh": speed,
            },
        )

    def _detect_suboptimal_shift(self, sector: dict) -> PatternMatch | None:
        """
        Padrão 4: Ponto de troca subótimo.

        Sinais: gear incorreta + RPM fora do range ótimo.
        """
        rpms = sector.get("rpms", 0)
        if rpms == 0:
            return None

        optimal_shift_rpm = self._max_rpm * (1.0 - RPM_SHIFT_MARGIN)
        optimal_min_rpm = self._max_rpm * 0.65  # Banda de potência típica

        too_late = rpms > self._max_rpm * (1.0 - RPM_SHIFT_MARGIN / 2)
        too_early = rpms < optimal_min_rpm and sector.get("gear", 0) > 2

        if not too_late and not too_early:
            return None

        if too_late:
            deviation = (rpms - optimal_shift_rpm) / self._max_rpm
            cause = "Troca de marcha tardia — motor acima da faixa de potência"
        else:
            deviation = (optimal_min_rpm - rpms) / self._max_rpm
            cause = "Troca de marcha antecipada — motor fora da faixa de potência"

        confidence = min(1.0, deviation * 5.0)

        return PatternMatch(
            cause=cause,
            confidence=round(confidence, 3),
            evidence={
                "rpms": rpms,
                "max_rpm": self._max_rpm,
                "gear": sector.get("gear", 0),
                "optimal_shift_rpm": int(optimal_shift_rpm),
            },
        )

    def _detect_compromised_exit(self, sector: dict) -> PatternMatch | None:
        """
        Padrão 5: Saída de curva comprometida.

        Sinais: throttle parcial em reta longa (steering baixo + speed alta + throttle < 1.0).
        """
        throttle = sector.get("throttle", 0.0)
        steering = abs(sector.get("steering", 0.0))
        speed = sector.get("speed_kmh", 0.0)

        # Só relevante em retas (steering baixo) a velocidade alta
        if steering > 0.2:
            return None
        if speed < 150.0:
            return None
        if throttle > 0.95:
            return None

        confidence = min(1.0, (1.0 - throttle) * (speed / 300.0) * 2.0)
        if confidence < 0.15:
            return None

        return PatternMatch(
            cause="Saída de curva comprometida — aceleração incompleta na reta",
            confidence=round(confidence, 3),
            evidence={
                "throttle": throttle,
                "speed_kmh": speed,
                "steering": steering,
            },
        )
