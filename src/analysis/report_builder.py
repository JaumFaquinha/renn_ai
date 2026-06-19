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
from src.models.sector_model import _TRACK_POSITION_MAX, _TRACK_POSITION_MIN

logger = logging.getLogger(__name__)


@dataclass
class SectorReport:
    """Relatório consolidado de um mini-setor com maior perda."""

    track_position: float
    delta_per_sector_s: float  # Perda dentro do setor (não cumulativo desde a largada)
    speed_min_kmh: float
    causes: list[PatternMatch]
    corner_name: Optional[str] = None
    corner_type: Optional[str] = None
    sector_name: Optional[str] = None  # Setor oficial (ex.: "Setor 2") da posição na spline
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

        Critério de ranking (2026-06-16): `delta_per_sector` (perda dentro
        do mini-setor) em vez de `delta_vs_best` (delta cumulativo desde o
        início da volta). O cumulativo carrega o ruído do reset do
        performanceMeter no cruzamento da linha — gerava falsos +20s em
        spline ≈ 0.000 mesmo em voltas limpas.

        Filtro posicional (2026-06-16): descarta mini-setores com
        track_position fora de [_TRACK_POSITION_MIN, _TRACK_POSITION_MAX]
        (mesmos limites do SectorModel.train). Onde o performanceMeter
        ainda está em transição entre voltas, qualquer ranking é ruído.

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
        # Filtro posicional: descarta extremos da spline (artefatos de reset
        # do performanceMeter). Preserva os índices originais para alinhar
        # com pattern_results / model_scores.
        candidates = [
            (idx, sector)
            for idx, sector in enumerate(analyzed_sectors)
            if _TRACK_POSITION_MIN
            <= float(sector.get("track_position") or 0.0)
            <= _TRACK_POSITION_MAX
        ]

        # Ordena por delta_per_sector (perda local), não pelo cumulativo.
        sorted_sectors = sorted(
            candidates,
            key=lambda x: x[1].get("delta_per_sector", 0.0),
            reverse=True,
        )[:n_top]

        sector_reports = []
        total_lost = 0.0

        for idx, sector in sorted_sectors:
            position = sector["track_position"]
            delta = sector.get("delta_per_sector", 0.0)
            total_lost += max(0.0, delta)

            corner = self._find_corner(position)
            causes = pattern_results.get(idx, [])

            sector_reports.append(
                SectorReport(
                    track_position=round(position, 4),
                    delta_per_sector_s=round(delta, 3),
                    speed_min_kmh=round(sector.get("speed_min", 0.0), 1),
                    causes=causes,
                    corner_name=corner.get("name") if corner else None,
                    corner_type=corner.get("type") if corner else None,
                    sector_name=self._find_sector(position),
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

    def _find_sector(self, position: float) -> Optional[str]:
        """
        Nome do setor oficial em que a posição da spline se encontra.

        Usado como rótulo de zona quando a posição não cai em nenhuma curva
        nomeada (ex.: trechos de reta) — comunica o setor em vez do valor cru
        da spline. Retorna None se não houver track map com setores definidos.
        """
        if not self._track_map:
            return None

        for sector in self._track_map.get("sectors", []):
            start = sector.get("spline_start")
            end = sector.get("spline_end")
            if start is not None and end is not None and start <= position <= end:
                return sector.get("name")

        return None
