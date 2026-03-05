"""
ConsoleReporter — FASE 5

Exibe o relatório de análise de volta no terminal de forma legível,
ordenado por magnitude de perda de tempo.

Saída via logging (sem print()) conforme CLAUDE.md §8.
"""

import logging

from src.analysis.report_builder import LapReport, SectorReport

logger = logging.getLogger(__name__)

_SEPARATOR = "─" * 70


class ConsoleReporter:
    """
    Formata e exibe relatórios de volta no terminal.

    Não usa bibliotecas de UI. Output via sys.stdout através do logging
    configurado em nível INFO.
    """

    def report(self, lap_report: LapReport) -> None:
        """
        Exibe o relatório completo de uma volta no terminal.

        Args:
            lap_report: LapReport construído pelo ReportBuilder.
        """
        lines = self._format(lap_report)
        for line in lines:
            logger.info(line)

    def _format(self, report: LapReport) -> list[str]:
        """Formata o relatório em linhas de texto."""
        lap_time_s = report.lap_time_ms / 1000.0

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
            lines.extend(self._format_sector(i, sector))

        lines.append(_SEPARATOR)
        return lines

    def _format_sector(self, rank: int, sector: SectorReport) -> list[str]:
        """Formata um mini-setor individual."""
        # Localização
        zone = sector.corner_name or f"Spline {sector.track_position:.3f}"
        corner_type = f" ({sector.corner_type})" if sector.corner_type else ""

        lines = [
            f"",
            f"  #{rank}  {zone}{corner_type}",
            f"      Posição    : {sector.track_position:.4f}",
            f"      Perda      : +{sector.delta_vs_best_s:.3f}s",
            f"      Vel. mín.  : {sector.speed_min_kmh:.1f} km/h",
        ]

        if sector.causes:
            lines.append(f"      Causas     :")
            for match in sector.causes:
                bar = self._confidence_bar(match.confidence)
                lines.append(f"        [{bar}] {match.confidence:.0%}  {match.cause}")
        else:
            lines.append(f"      Causas     : — (padrão não identificado)")

        return lines

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
