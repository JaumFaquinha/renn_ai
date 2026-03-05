"""
map_track.py — Mapeamento interativo de curvas por pista.

Lê a posição normalizada na spline em tempo real e permite ao usuário
registrar o nome e tipo de cada curva pressionando teclas.

Gera um arquivo JSON em config/track_maps/{track_id}.json.

Uso:
    python scripts/map_track.py --track monza

Controles durante o mapeamento:
    ENTER  — registra a posição atual como início de uma curva
    Q      — encerra e salva o mapa
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import TRACK_MAPS_DIR
from src.memory.shared_memory_reader import SharedMemoryReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("map_track")


CORNER_TYPES = ["slow_right", "slow_left", "medium_right", "medium_left",
                "fast_right", "fast_left", "chicane", "hairpin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mapeador de curvas de pista para o Engenheiro de Corrida IA")
    parser.add_argument("--track", required=True, help="ID da pista (ex: monza, barcelona)")
    return parser.parse_args()


def prompt_corner(position: float) -> dict | None:
    """Solicita informações de uma curva interativamente."""
    print(f"\n  Posição atual na spline: {position:.4f}")
    name = input("  Nome da curva (ENTER para cancelar): ").strip()
    if not name:
        return None

    print("  Tipos disponíveis:", ", ".join(CORNER_TYPES))
    corner_type = input("  Tipo: ").strip().lower()
    if corner_type not in CORNER_TYPES:
        corner_type = "unknown"

    end_pos = float(input(f"  Posição de fim da curva (atual={position:.4f}): ").strip() or position + 0.05)

    return {
        "name": name,
        "spline_position": round(position, 4),
        "spline_range": [round(position, 4), round(end_pos, 4)],
        "type": corner_type,
        "notes": "",
    }


def run(track_id: str) -> None:
    corners: list[dict] = []
    spline_length = 0.0

    with SharedMemoryReader() as reader:
        if not reader.connect():
            logger.error("Não foi possível conectar ao AC. Certifique-se que o jogo está rodando.")
            return

        static = reader.read_static()
        spline_length = static.trackSPlineLength

        logger.info(
            "Conectado — pista: %s | comprimento: %.1f m",
            static.track,
            spline_length,
        )
        print("\n  === Mapeador de Curvas ===")
        print("  ENTER = registrar curva na posição atual")
        print("  Q     = finalizar e salvar\n")

        while True:
            snapshot = reader.read()
            if not snapshot.read_ok:
                time.sleep(0.1)
                continue

            position = snapshot.graphics.normalizedCarPosition
            print(f"\r  Posição: {position:.4f}  |  Curvas registradas: {len(corners)}", end="", flush=True)

            # Leitura não-bloqueante de tecla (simplificada com input())
            try:
                user_input = input().strip().lower()
            except EOFError:
                break

            if user_input == "q":
                break

            corner = prompt_corner(position)
            if corner:
                corners.append(corner)
                print(f"  ✓ Curva '{corner['name']}' registrada.")

    # Salvar mapa
    TRACK_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRACK_MAPS_DIR / f"{track_id}.json"

    track_map = {
        "track_id": track_id,
        "track_name": track_id.replace("_", " ").title(),
        "layout": "GP",
        "spline_length_m": spline_length,
        "sectors": [],
        "corners": corners,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(track_map, f, indent=2, ensure_ascii=False)

    logger.info("Track map salvo: %s (%d curvas)", output_path, len(corners))


if __name__ == "__main__":
    args = parse_args()
    run(track_id=args.track)
