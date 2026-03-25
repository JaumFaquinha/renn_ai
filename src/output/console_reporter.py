"""
ConsoleReporter — FASE 5 + FASE 7D

Exibe o relatório de análise de volta no terminal de forma legível,
ordenado por magnitude de perda de tempo.

Fase 7D: indicadores de tendência histórica por setor (↓ melhorou / ↑ piorou / → estável)
quando sector_history é fornecido via QueryService.get_sectors_history_batch().

Saída via logging (sem print()) conforme CLAUDE.md §8.
"""

import logging
from typing import Optional

from src.analysis.report_builder import LapReport, SectorReport

logger = logging.getLogger(__name__)

_SEPARATOR = "─" * 70

# Diferença mínima em segundos para considerar melhoria ou piora (vs média histórica)
_TREND_THRESHOLD_S: float = 0.05


class ConsoleReporter:
    """
    Formata e exibe relatórios de volta no terminal.

    Não usa bibliotecas de UI. Output via sys.stdout através do logging
    configurado em nível INFO.
    """

    def report(
        self,
        lap_report: LapReport,
        sector_history: Optional[dict[float, list[dict]]] = None,
    ) -> None:
        """
        Exibe o relatório completo de uma volta no terminal.

        Args:
            lap_report: LapReport construído pelo ReportBuilder.
            sector_history: dict {track_position → list[{session_date, avg_delta}]}
                            retornado por QueryService.get_sectors_history_batch().
                            Se None, indicadores de tendência são omitidos.
        """
        lines = self._format(lap_report, sector_history or {})
        for line in lines:
            logger.info(line)

    def _format(self, report: LapReport, sector_history: dict) -> list[str]:
        """Formata o relatório em linhas de texto."""
        lines = [
            "",
            _SEPARATOR,
            f"  ANÁLISE DE VOLTA #{report.lap_number}  |  {report.track_id.upper()}",
            _SEPARATOR,
            f"  Tempo da volta : {self._format_time(report.lap_time_ms)}",
            f"  Tempo perdido  : +{report.total_time_lost_s:.3f}s (top {len(report.top_sectors)} setores)",
            _SEPARATOR,
        ]

        if not report.top_sectors:
            lines.append("  Nenhuma perda significativa detectada.")
            lines.append(_SEPARATOR)
            return lines

        for i, sector in enumerate(report.top_sectors, 1):
            history = sector_history.get(sector.track_position, [])
            lines.extend(self._format_sector(i, sector, history))

        lines.append(_SEPARATOR)
        return lines

    def _format_sector(
        self,
        rank: int,
        sector: SectorReport,
        history: list[dict],
    ) -> list[str]:
        """Formata um mini-setor individual com tendência histórica opcional."""
        zone = sector.corner_name or f"Spline {sector.track_position:.3f}"
        corner_type = f" ({sector.corner_type})" if sector.corner_type else ""
        trend = self._trend_indicator(sector.delta_vs_best_s, history)

        lines = [
            f"",
            f"  #{rank}  {zone}{corner_type}",
            f"      Posição    : {sector.track_position:.4f}",
            f"      Perda      : +{sector.delta_vs_best_s:.3f}s{trend}",
            f"      Vel. mín.  : {sector.speed_min_kmh:.1f} km/h",
        ]

        if sector.model_score is not None:
            score_bar = self._confidence_bar(sector.model_score)
            lines.append(
                f"      Modelo IA  : [{score_bar}] {sector.model_score:.0%}  anomalia prevista"
            )

        if sector.causes:
            lines.append(f"      Causas     :")
            for match in sector.causes:
                bar = self._confidence_bar(match.confidence)
                lines.append(f"        [{bar}] {match.confidence:.0%}  {match.cause}")
        else:
            lines.append(f"      Causas     : — (padrão não identificado)")

        return lines

    def _trend_indicator(self, current_delta: float, history: list[dict]) -> str:
        """
        Compara o delta atual com a média histórica e retorna indicador de tendência.

        Args:
            current_delta: delta_vs_best da volta atual neste setor
            history: lista de entradas históricas com avg_delta por sessão

        Returns:
            String formatada com indicador, ou "" se histórico insuficiente.
        """
        if not history:
            return ""

        valid_deltas = [
            h["avg_delta"] for h in history
            if h.get("avg_delta") is not None
        ]
        if len(valid_deltas) < 2:
            return ""

        historical_avg = sum(valid_deltas) / len(valid_deltas)
        diff = current_delta - historical_avg

        if diff < -_TREND_THRESHOLD_S:
            return f"  ↓ -{abs(diff):.2f}s vs média ({len(valid_deltas)} sessões)"
        elif diff > _TREND_THRESHOLD_S:
            return f"  ↑ +{diff:.2f}s vs média ({len(valid_deltas)} sessões)"
        else:
            return f"  → estável vs média ({len(valid_deltas)} sessões)"

    @staticmethod
    def _format_time(ms: int) -> str:
        """Formata milissegundos em mm:ss.mmm."""
        if ms <= 0:
            return "--:--.---"
        minutes = ms // 60000
        seconds = (ms % 60000) / 1000.0
        return f"{minutes:02d}:{seconds:06.3f}"

    @staticmethod
    def _confidence_bar(confidence: float, width: int = 10) -> str:
        """Gera uma barra visual de confiança."""
        filled = round(confidence * width)
        return "█" * filled + "░" * (width - filled)
