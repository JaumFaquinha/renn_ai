"""
map_track.py — Mapeamento interativo de curvas por pista.

Lê a posição normalizada na spline em tempo real e permite ao usuário
registrar o nome e tipo de cada curva pressionando teclas.

Gera um arquivo JSON em config/track_maps/{track_id}.json.

Uso:
    python scripts/map_track.py --track monza
    python scripts/map_track.py --track monza --layout combined
    python scripts/map_track.py --track monza --resume       # retoma mapeamento existente

Controles durante o mapeamento:
    ENTER  — registra a posição atual como início de uma curva
    Q      — encerra e salva o mapa

Nota: usa msvcrt (Windows stdlib) para leitura de tecla não-bloqueante.
      Compatível com a plataforma alvo do projeto (Windows — Shared Memory do AC).
"""

import argparse
import json
import logging
import msvcrt
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

CORNER_TYPES = [
    "slow_right", "slow_left",
    "medium_right", "medium_left",
    "fast_right", "fast_left",
    "chicane", "hairpin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mapeador de curvas de pista para o Engenheiro de Corrida IA",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--track",
        required=True,
        help="ID da pista (ex: monza, barcelona, nurburgring_gp)",
    )
    parser.add_argument(
        "--layout",
        default="GP",
        help="Layout da pista quando há múltiplos (ex: GP, combined, junior). Default: GP",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma mapeamento existente sem apagar curvas já registradas",
    )
    return parser.parse_args()


def prompt_corner(position: float) -> dict | None:
    """
    Solicita informações de uma curva interativamente via input bloqueante.

    Chamado apenas após detecção de ENTER — a posição capturada é a do momento
    do keypress, não do input (evita drift por digitação lenta).

    Args:
        position: posição na spline no momento do keypress (0.0–1.0)

    Returns:
        Dict da curva ou None se o usuário cancelar.
    """
    print(f"\n  Posição capturada: {position:.4f}")
    name = input("  Nome da curva (ENTER para cancelar): ").strip()
    if not name:
        return None

    print("  Tipos disponíveis:")
    for i, ct in enumerate(CORNER_TYPES, 1):
        print(f"    {i}. {ct}")
    type_input = input("  Tipo (nome ou número): ").strip().lower()

    # Aceita nome direto ou número do índice
    corner_type = "unknown"
    if type_input in CORNER_TYPES:
        corner_type = type_input
    elif type_input.isdigit():
        idx = int(type_input) - 1
        if 0 <= idx < len(CORNER_TYPES):
            corner_type = CORNER_TYPES[idx]

    # FIX: validação de end_pos > start_pos
    end_pos: float = position + 0.05
    while True:
        raw = input(
            f"  Posição de fim da curva (atual={position:.4f}, deve ser > {position:.4f}): "
        ).strip()
        if not raw:
            end_pos = round(position + 0.05, 4)
            break
        try:
            candidate = float(raw)
            if candidate <= position:
                print(f"  ⚠  Valor deve ser maior que {position:.4f}. Tente novamente.")
                continue
            if candidate > 1.0:
                print("  ⚠  Valor deve ser <= 1.0. Tente novamente.")
                continue
            end_pos = round(candidate, 4)
            break
        except ValueError:
            print("  ⚠  Valor inválido. Digite um número decimal (ex: 0.15).")

    notes = input("  Notas adicionais (opcional): ").strip()

    return {
        "name": name,
        "spline_position": round(position, 4),
        "spline_range": [round(position, 4), end_pos],
        "type": corner_type,
        "notes": notes,
    }


def run(track_id: str, layout: str, resume: bool) -> None:
    """
    Loop principal de mapeamento interativo.

    Usa msvcrt.kbhit() para leitura não-bloqueante — a posição na spline
    é atualizada a cada iteração do loop sem travar esperando input.

    Args:
        track_id: identificador da pista (ex: 'monza')
        layout:   nome do layout (ex: 'GP', 'junior')
        resume:   se True, carrega curvas já registradas do JSON existente
    """
    corners: list[dict] = []
    spline_length = 0.0
    output_path = TRACK_MAPS_DIR / f"{track_id}.json"

    # FIX: modo resume — carrega curvas existentes sem apagar
    if resume and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            corners = existing.get("corners", [])
            logger.info(
                "Modo resume — %d curvas já registradas carregadas",
                len(corners),
            )
        except Exception as exc:
            logger.warning("Falha ao carregar mapa existente: %s", exc)

    with SharedMemoryReader() as reader:
        if not reader.connect():
            logger.error(
                "Não foi possível conectar ao AC. "
                "Certifique-se que o jogo está rodando com Shared Memory habilitada."
            )
            return

        static = reader.read_static()
        spline_length = static.trackSPlineLength
        track_name_ac = static.track  # nome interno do AC

        # Avisa se o track_id passado difere do nome que o AC reporta
        if track_name_ac and track_name_ac.lower() != track_id.lower():
            logger.warning(
                "track_id passado ('%s') difere do nome AC ('%s'). "
                "O arquivo será salvo como '%s.json'.",
                track_id,
                track_name_ac,
                track_id,
            )

        logger.info(
            "Conectado — pista AC: %s | comprimento: %.1f m | layout: %s",
            track_name_ac,
            spline_length,
            layout,
        )
        print("\n  ══════════════════════════════════════════════")
        print(f"  Mapeador de Curvas — {track_id.upper()} ({layout})")
        print("  ══════════════════════════════════════════════")
        print("  ENTER  = registrar curva na posição atual")
        print("  Q      = finalizar e salvar")
        print("  A posição é atualizada em tempo real.\n")

        while True:
            snapshot = reader.read()
            if not snapshot.read_ok:
                time.sleep(0.05)
                continue

            position = snapshot.graphics.normalizedCarPosition

            # Atualiza display em tempo real (não bloqueia)
            print(
                f"\r  Posição: {position:.4f}  |  Curvas: {len(corners):>3}  "
                f"|  [ENTER=registrar  Q=sair]   ",
                end="",
                flush=True,
            )

            # FIX: leitura não-bloqueante com msvcrt — sem travar a posição
            if msvcrt.kbhit():
                key = msvcrt.getch()

                if key in (b"\r", b"\n"):
                    # Captura a posição exata no momento do ENTER
                    captured_position = position
                    corner = prompt_corner(captured_position)
                    if corner:
                        corners.append(corner)
                        print(f"  ✓  Curva '{corner['name']}' registrada em {captured_position:.4f}.")
                    else:
                        print("  —  Cancelado.")

                elif key.lower() == b"q":
                    print("\n  Encerrando mapeamento...")
                    break

            time.sleep(0.05)  # 20Hz — consistente com o sampling rate do projeto

    # Salvar mapa
    if not corners:
        logger.warning("Nenhuma curva registrada — arquivo não salvo.")
        return

    # Ordena curvas por posição na spline para clareza no JSON
    corners.sort(key=lambda c: c["spline_position"])

    TRACK_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # FIX: preserva campos existentes do JSON ao fazer resume
    track_map = {
        "track_id": track_id,
        "track_name": track_id.replace("_", " ").title(),
        "layout": layout,
        "spline_length_m": spline_length,
        "sectors": [],  # preenchido manualmente ou por futura extensão
        "corners": corners,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(track_map, f, indent=2, ensure_ascii=False)

    logger.info(
        "Track map salvo: %s (%d curvas, layout=%s)",
        output_path,
        len(corners),
        layout,
    )


if __name__ == "__main__":
    args = parse_args()
    run(track_id=args.track, layout=args.layout, resume=args.resume)
