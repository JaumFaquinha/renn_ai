"""
voice_message — FASE 6 (construção da mensagem falada do engenheiro)

Transforma um LapReport na frase curta de rádio enviada ao TTS. Dois tons:

- **Volta com perda**: contexto (perda total da volta) + pior zona (nome e
  tipo da curva) + até 2 causas (a secundária do pior setor, ou a primária
  do segundo setor mais lento).
- **Volta boa**: mensagem positiva quando a volta é a melhor de todos os
  tempos, a melhor da sessão, ou quando o modelo não prevê anomalia relevante
  nos setores reportados (`SectorReport.model_score`). Sem modelo treinado,
  usa a perda total como fallback.

Mantido propositalmente conciso (estilo engenheiro de corrida). O TTS ainda
aplica o truncamento de segurança (`TTS_MAX_MESSAGE_CHARS`).
"""

import logging
from typing import Optional

from config.settings import (
    GOOD_LAP_ANOMALY_MAX,
    GOOD_LAP_TOTAL_LOSS_MAX_S,
    VOICE_HYBRID_MIN_GAIN_S,
    VOICE_SECONDARY_SECTOR_MIN_LOSS_S,
)
from src.analysis.report_builder import LapReport, SectorReport

logger = logging.getLogger(__name__)


# Mensagens positivas por categoria — rotacionadas por lap_number para variar
# e não soar robótico ao repetir voltas boas.
_MSG_ALLTIME_BEST = [
    "Volta excelente! Novo recorde pessoal. Mandou muito bem.",
    "Sensacional! Melhor tempo de todos os tempos. Pilotagem impecável.",
]
_MSG_SESSION_BEST = [
    "Boa! Melhor volta da sessão. Continue nesse ritmo.",
    "Ótima volta! A melhor da sessão até agora. Mantenha o foco.",
]
_MSG_CLEAN_LAP = [
    "Volta limpa, sem perdas significativas. Muito consistente.",
    "Volta sólida, sem erros relevantes. O ritmo está bom.",
]


def build_voice_message(
    lap_report: LapReport,
    *,
    is_session_best: bool = False,
    is_alltime_best: bool = False,
) -> Optional[str]:
    """
    Constrói a mensagem falada a partir do relatório da volta.

    Args:
        lap_report: relatório construído por ReportBuilder.build(). Os
            `top_sectors` já carregam `model_score` quando o SectorModel está
            treinado, e `causes` ordenadas por confiança.
        is_session_best: True se esta é a melhor volta da sessão.
        is_alltime_best: True se esta é a melhor volta de todos os tempos.

    Returns:
        A frase a ser falada, ou None quando não há nada relevante a dizer
        (sem setores reportados e volta não classificada como boa).
    """
    seed = lap_report.lap_number

    # 1) Volta boa → mensagem positiva, com cláusula híbrida apontando a maior
    #    oportunidade restante quando o ganho for relevante (recorde > sessão > limpa)
    if is_alltime_best:
        return _with_hybrid(_pick(_MSG_ALLTIME_BEST, seed), lap_report)
    if is_session_best:
        return _with_hybrid(_pick(_MSG_SESSION_BEST, seed), lap_report)
    if _is_good_lap(lap_report):
        return _with_hybrid(_pick(_MSG_CLEAN_LAP, seed), lap_report)

    # 2) Volta com perda → contexto + pior zona + causas
    if not lap_report.top_sectors:
        return None

    top = lap_report.top_sectors[0]
    parts = [
        f"Volta com {_fmt_s(lap_report.total_time_lost_s)} de perda no total.",
        f"Maior perda {_zone_phrase(top)}: {_fmt_s(top.delta_per_sector_s)} segundos.",
    ]

    cause_bits = _collect_causes(lap_report, top)
    if cause_bits:
        parts.append("Causas: " + "; ".join(cause_bits) + ".")
    else:
        parts.append("Causa não identificada.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_good_lap(lap_report: LapReport) -> bool:
    """
    Classifica a volta como boa.

    Prioriza o sinal do modelo: se os setores reportados (já filtrados por
    posição) têm anomalia máxima abaixo de GOOD_LAP_ANOMALY_MAX, o modelo não
    prevê perda relevante. Sem modelo treinado (model_score None), usa a perda
    total como fallback. Volta sem setores reportados é considerada limpa.
    """
    if not lap_report.top_sectors:
        return True

    scored = [
        s.model_score for s in lap_report.top_sectors if s.model_score is not None
    ]
    if scored:
        return max(scored) < GOOD_LAP_ANOMALY_MAX

    return lap_report.total_time_lost_s < GOOD_LAP_TOTAL_LOSS_MAX_S


def _collect_causes(lap_report: LapReport, top: SectorReport) -> list[str]:
    """
    Reúne até 2 causas: a primária do pior setor mais uma secundária — a
    segunda causa do próprio pior setor, ou (na falta dela) a causa principal
    do segundo setor mais lento, se a perda dele for relevante.
    """
    causes: list[str] = []
    if top.causes:
        causes.append(top.causes[0].cause)

    if len(top.causes) > 1:
        causes.append(top.causes[1].cause)
    elif len(lap_report.top_sectors) > 1:
        second = lap_report.top_sectors[1]
        if (
            second.delta_per_sector_s >= VOICE_SECONDARY_SECTOR_MIN_LOSS_S
            and second.causes
        ):
            cause = second.causes[0].cause
            cause = cause[:1].lower() + cause[1:]  # encadeia naturalmente
            causes.append(f"e {_zone_phrase(second)}, {cause}")

    return causes


def _with_hybrid(base: str, lap_report: LapReport) -> str:
    """Anexa à mensagem positiva a maior oportunidade restante, se relevante."""
    clause = _improvement_clause(lap_report)
    return f"{base} {clause}" if clause else base


def _improvement_clause(lap_report: LapReport) -> Optional[str]:
    """
    Cláusula híbrida: aponta o maior ganho ainda disponível mesmo numa volta
    boa. Retorna None se o ganho do pior setor ficar abaixo do limiar
    (VOICE_HYBRID_MIN_GAIN_S), deixando a mensagem puramente elogiosa.
    """
    if not lap_report.top_sectors:
        return None
    top = lap_report.top_sectors[0]
    if top.delta_per_sector_s < VOICE_HYBRID_MIN_GAIN_S:
        return None

    clause = (
        f"Ainda dá pra ganhar {_fmt_s(top.delta_per_sector_s)} segundos "
        f"{_zone_phrase(top)}"
    )
    if top.causes:
        cause = top.causes[0].cause
        cause = cause[:1].lower() + cause[1:]
        return f"{clause}: {cause}."
    return f"{clause}."


def _zone_phrase(sector: SectorReport) -> str:
    """
    Frase prepositiva da zona, pronta para encaixar após um verbo:
    'em Parabólica, curva rápida' ou 'no trecho 0,50 da pista'.
    """
    if sector.corner_name:
        if sector.corner_type:
            return f"em {sector.corner_name}, {sector.corner_type}"
        return f"em {sector.corner_name}"
    if sector.sector_name:
        return f"no {sector.sector_name}"
    pos = f"{sector.track_position:.2f}".replace(".", ",")
    return f"no trecho {pos} da pista"


def _fmt_s(value: float) -> str:
    """Formata segundos em pt-BR (vírgula decimal). Ex.: 0.30 → '0,30'."""
    return f"{value:.2f}".replace(".", ",")


def _pick(options: list[str], seed: int) -> str:
    """Escolha determinística e rotativa dentro de uma lista de variantes."""
    return options[seed % len(options)]
