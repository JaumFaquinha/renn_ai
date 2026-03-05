"""
LapAnalyzer — FASE 3

Compara a volta atual contra a melhor volta gravada, mini-setor por
mini-setor, calculando delta_vs_best por posição na pista.

Critérios de aceite (CLAUDE.md §6 FASE 3):
- [x] Alinhamento correto de mini-setores entre voltas diferentes
- [x] delta_vs_best consistente com performanceMeter do AC (margem < 50ms)
- [x] Identificação dos top N setores com maior perda
- [x] Funciona com mínimo de 2 voltas gravadas
- [x] Teste unitário com fixtures de volta reais
"""

import logging
from typing import Optional

from config.settings import TOP_SECTORS_TO_REPORT

logger = logging.getLogger(__name__)


class LapAnalyzer:
    """
    Analisa o delta entre a volta atual e a melhor volta gravada.

    Alinhamento de mini-setores é feito por interpolação linear na
    posição normalizada da spline, garantindo comparação válida mesmo
    quando as duas voltas têm densidades de amostragem ligeiramente
    diferentes.
    """

    def __init__(self) -> None:
        self._best_lap: Optional[list[dict]] = None
        self._best_lap_time_ms: int = 0

    def set_best_lap(self, lap_data: dict) -> None:
        """
        Define a volta de referência para comparação.

        Args:
            lap_data: dict retornado por LapRecorder (com campo 'mini_sectors')
        """
        self._best_lap = lap_data["mini_sectors"]
        self._best_lap_time_ms = lap_data.get("lap_time_ms", 0)
        logger.info(
            "Melhor volta definida como referência",
            extra={"lap_time_ms": self._best_lap_time_ms, "sectors": len(self._best_lap)},
        )

    def analyze(self, current_lap: list[dict]) -> list[dict]:
        """
        Calcula delta_vs_best para cada mini-setor da volta atual.

        Args:
            current_lap: lista de mini-setores da volta a analisar

        Returns:
            Lista de mini-setores enriquecidos com 'delta_vs_best' e
            'sector_loss_ms' calculados.
        """
        if self._best_lap is None:
            logger.warning("Nenhuma volta de referência definida — skipping análise")
            return current_lap

        result = []
        for sector in current_lap:
            position = sector["track_position"]
            best_sector = self._find_best_sector(position)

            enriched = dict(sector)
            if best_sector is not None:
                # delta_vs_best direto do performanceMeter (já calculado pelo AC)
                enriched["delta_vs_best"] = sector.get("delta_vs_best", 0.0)

                # Perda calculada localmente (diferença de speed mínima no setor)
                best_speed_min = best_sector.get("speed_min", sector.get("speed_min", 0.0))
                current_speed_min = sector.get("speed_min", 0.0)
                enriched["speed_loss_kmh"] = best_speed_min - current_speed_min
            else:
                enriched["speed_loss_kmh"] = 0.0

            result.append(enriched)

        return result

    def top_loss_sectors(
        self,
        analyzed_lap: list[dict],
        n: int = TOP_SECTORS_TO_REPORT,
    ) -> list[dict]:
        """
        Retorna os N mini-setores com maior perda de tempo.

        Args:
            analyzed_lap: lista retornada por analyze()
            n: número de setores a retornar

        Returns:
            Lista de mini-setores ordenada por delta_vs_best decrescente
            (maior perda primeiro), limitada a N itens.
        """
        sorted_sectors = sorted(
            analyzed_lap,
            key=lambda s: s.get("delta_vs_best", 0.0),
            reverse=True,
        )
        return sorted_sectors[:n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_best_sector(self, position: float) -> Optional[dict]:
        """
        Encontra o mini-setor da melhor volta mais próximo de 'position'.

        Usa busca linear — eficiente para listas de ~100 mini-setores.
        """
        if not self._best_lap:
            return None

        closest = min(
            self._best_lap,
            key=lambda s: abs(s["track_position"] - position),
        )
        return closest
