"""
ReportBuilder — FASE 5

Consolida análise de delta e detecção de padrões em um relatório
estruturado, ordenado por magnitude de perda de tempo.

Critérios de aceite (CLAUDE.md §6 FASE 5):
- [x] Relatório ordenado por magnitude de perda (maior primeiro)
- [x] Exibe: zona da pista, posição na spline, delta em segundos, causa
- [x] Mapeamento posição → nome da curva (via config/track_maps/)
- [x] Output legível em terminal sem bibliotecas de UI
- [x] Latência do relatório pós-volta < 2 segundos
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import TOP_SECTORS_TO_REPORT, TRACK_MAPS_DIR
from src.analysis.pattern_detector import PatternMatch

logger = logging.getLogger(__name__)


@dataclass
class SectorReport:
    """Relatório consolidado de um mini-setor com maior perda."""

    track_position: float
    delta_vs_best_s: float
    speed_min_kmh: float
    causes: list[PatternMatch]
    corner_name: Optional[str] = None
    corner_type: Optional[str] = None
    model_score: Optional[float] = None  # Anomaly score do SectorModel 0.0–1.0


@dataclass
class LapReport:
    """Relatório completo de uma volta analisada."""

    lap_number: int
    lap_time_ms: int
    track_id: str
    top_sectors: list[SectorReport]
    total_time_lost_s: float = 0.0


class ReportBuilder:
    """
    Constrói relatórios estruturados a partir de dados analisados.

    Carrega o mapa de curvas da pista (config/track_maps/{track_id}.json)
    para enriquecer os setores com nome da curva correspondente.
    """

    def __init__(self, track_id: str = "unknown") -> None:
        self._track_id = track_id
        self._track_map: Optional[dict] = self._load_track_map(track_id)

    def build(
        self,
        lap_number: int,
        lap_time_ms: int,
        analyzed_sectors: list[dict],
        pattern_results: dict[int, list[PatternMatch]],
        n_top: int = TOP_SECTORS_TO_REPORT,
        model_scores: Optional[dict[int, float]] = None,
    ) -> LapReport:
        """
        Constrói o relatório completo da volta.

        Args:
            lap_number: número da volta na sessão
            lap_time_ms: tempo total da volta em ms
            analyzed_sectors: lista de mini-setores de LapAnalyzer.analyze()
            pattern_results: mapa sector_index → list[PatternMatch]
            n_top: número de setores a incluir no relatório

        Returns:
            LapReport com os N setores de maior perda, suas causas
            e model_score quando SectorModel disponível.
        """
        # Ordenar por perda e pegar os N maiores
        sorted_sectors = sorted(
            enumerate(analyzed_sectors),
            key=lambda x: x[1].get("delta_vs_best", 0.0),
            reverse=True,
        )[:n_top]

        sector_reports = []
        total_lost = 0.0

        for idx, sector in sorted_sectors:
            position = sector["track_position"]
            delta = sector.get("delta_vs_best", 0.0)
            total_lost += max(0.0, delta)

            corner = self._find_corner(position)
            causes = pattern_results.get(idx, [])

            sector_reports.append(
                SectorReport(
                    track_position=round(position, 4),
                    delta_vs_best_s=round(delta, 3),
                    speed_min_kmh=round(sector.get("speed_min", 0.0), 1),
                    causes=causes,
                    corner_name=corner.get("name") if corner else None,
                    corner_type=corner.get("type") if corner else None,
                    model_score=model_scores.get(idx) if model_scores else None,
                )
            )

        return LapReport(
            lap_number=lap_number,
            lap_time_ms=lap_time_ms,
            track_id=self._track_id,
            top_sectors=sector_reports,
            total_time_lost_s=round(total_lost, 3),
        )

    # ------------------------------------------------------------------
    # Track Map
    # ------------------------------------------------------------------

    def _load_track_map(self, track_id: str) -> Optional[dict]:
        """Carrega o mapa de curvas da pista, se disponível."""
        path = TRACK_MAPS_DIR / f"{track_id}.json"
        if not path.exists():
            logger.debug("Track map não encontrado", extra={"track_id": track_id})
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Track map carregado", extra={"track_id": track_id, "corners": len(data.get("corners", []))})
            return data
        except Exception as exc:
            logger.warning("Falha ao carregar track map", extra={"error": str(exc)})
            return None

    def _find_corner(self, position: float) -> Optional[dict]:
        """
        Encontra a curva mais próxima da posição dada na spline.

        Retorna None se não houver track map ou a posição estiver fora
        de qualquer range de curva definido.
        """
        if not self._track_map:
            return None

        for corner in self._track_map.get("corners", []):
            spline_range = corner.get("spline_range", [])
            if len(spline_range) == 2:
                if spline_range[0] <= position <= spline_range[1]:
                    return corner

        return None
