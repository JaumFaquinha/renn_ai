"""
query_history.py — CLI de consulta histórica de telemetria.

Exibe personal best, progresso por sessão e padrões de perda mais frequentes
consultando o Supabase diretamente. Funciona sem o AC rodando.

Uso:
    python scripts/query_history.py --track monza
    python scripts/query_history.py --track monza --car ferrari_488_gt3
    python scripts/query_history.py --track monza --sessions 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Força UTF-8 no stdout do terminal Windows (evita UnicodeEncodeError com caracteres box-drawing)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import LOG_LEVEL
from src.persistence.supabase_client import SupabaseClient
from src.persistence.query_service import QueryService

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("query_history")

_SEP = "═" * 64
_SEP_THIN = "─" * 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engenheiro de Corrida IA — Histórico de Telemetria"
    )
    parser.add_argument("--track", required=True, help="ID da pista (ex: monza)")
    parser.add_argument(
        "--car", default=None,
        help="Modelo do carro (ex: ferrari_488_gt3). Se omitido, usa o último registrado."
    )
    parser.add_argument(
        "--sessions", type=int, default=10,
        help="Número de sessões a considerar (default: 10)"
    )
    return parser.parse_args()


def _format_time(ms: int) -> str:
    """Formata milissegundos em mm:ss.mmm."""
    if not ms or ms <= 0:
        return "--:--.---"
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000.0
    return f"{minutes:02d}:{seconds:06.3f}"


def _confidence_bar(percentage: float, width: int = 10) -> str:
    """Gera barra visual de proporção."""
    filled = round(percentage * width)
    return "█" * filled + "░" * (width - filled)


def _resolve_car_model(query_service: QueryService, track_id: str) -> str | None:
    """
    Se --car não for fornecido, busca o modelo do carro mais recente nas sessões.
    Faz query direta na tabela sessions via supabase_client.
    """
    if not query_service._client.is_enabled:
        return None
    try:
        result = (
            query_service._client.get_client()
            .table("sessions")
            .select("car_model")
            .eq("track_id", track_id)
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        if result and result.data:
            return result.data[0]["car_model"]
    except Exception as exc:
        logger.warning("Falha ao resolver car_model", extra={"error": str(exc)})
    return None


def show_history(
    query_service: QueryService,
    track_id: str,
    car_model: str,
    last_n: int,
) -> None:
    """Busca e exibe o histórico completo no terminal."""

    # --- Header ---
    print(f"\n{_SEP}")
    print(f"  HISTÓRICO — {track_id.upper()} / {car_model}")
    print(f"  Últimas {last_n} sessões")
    print(_SEP)

    # --- Personal Best ---
    pb = query_service.get_personal_best(track_id=track_id, car_model=car_model)
    if pb:
        pb_date = (pb.get("session_date") or "")[:10]
        print(f"  Personal Best : {_format_time(pb['lap_time_ms'])}  ({pb_date})")
    else:
        print(f"  Personal Best : sem dados")

    # --- Histórico de Sessões ---
    history = query_service.get_session_history(
        track_id=track_id,
        car_model=car_model,
        last_n_sessions=last_n,
    )

    if history:
        best_ever = min(s["best_lap_ms"] for s in history)
        first_session_ms = history[-1]["best_lap_ms"] if len(history) > 1 else None
        avg_ms = sum(s["best_lap_ms"] for s in history) / len(history)

        print(f"  Média sessões : {_format_time(int(avg_ms))}")

        if first_session_ms and first_session_ms != best_ever:
            improvement_s = (first_session_ms - best_ever) / 1000.0
            print(f"  Melhoria total: -{improvement_s:.3f}s vs primeira sessão")

        print(f"\n{_SEP_THIN}")
        print(f"  PROGRESSO POR SESSÃO")
        print(_SEP_THIN)

        for i, session in enumerate(history):
            date_str = (session.get("session_date") or "")[:10]
            lap_time = session["best_lap_ms"]
            time_str = _format_time(lap_time)

            # Delta vs PB
            pb_ms = pb["lap_time_ms"] if pb else lap_time
            delta_s = (lap_time - pb_ms) / 1000.0

            # Indicador visual
            if lap_time == pb_ms:
                marker = "  ★ PB"
            elif i == 0:
                marker = "  ← sessão mais recente"
            else:
                marker = f"  +{delta_s:.3f}s"

            print(f"  {date_str}  {time_str}{marker}")
    else:
        print(f"  Nenhuma sessão registrada para {track_id} / {car_model}.")

    # --- Padrões Mais Frequentes ---
    patterns = query_service.get_pattern_frequency(
        track_id=track_id,
        car_model=car_model,
        last_n_sessions=last_n,
    )

    if patterns:
        print(f"\n{_SEP_THIN}")
        print(f"  PADRÕES DE PERDA MAIS FREQUENTES (últimas {last_n} sessões)")
        print(_SEP_THIN)

        for p in patterns[:5]:  # top 5
            bar = _confidence_bar(p["percentage"])
            pct = int(p["percentage"] * 100)
            corner = p.get("corner_name") or "—"
            print(f"  [{bar}] {pct:3d}%  {p['cause']}  ({corner})")

    print(f"\n{_SEP}\n")


def main() -> None:
    args = parse_args()

    sb_client = SupabaseClient()
    if not sb_client.is_enabled:
        print("\n[ERRO] SUPABASE_ENABLED=false no .env — histórico não disponível.")
        print("Configure as credenciais do Supabase e defina SUPABASE_ENABLED=true.\n")
        sys.exit(1)

    if not sb_client.health_check():
        print("\n[ERRO] Não foi possível conectar ao Supabase. Verifique as credenciais.\n")
        sys.exit(1)

    query_service = QueryService(sb_client)

    car_model = args.car
    if not car_model:
        car_model = _resolve_car_model(query_service, args.track)
        if not car_model:
            print(f"\n[ERRO] Nenhum carro encontrado para a pista '{args.track}'.")
            print("Use --car para especificar o modelo do carro.\n")
            sys.exit(1)
        print(f"\n  Carro detectado: {car_model}")

    show_history(
        query_service=query_service,
        track_id=args.track,
        car_model=car_model,
        last_n=args.sessions,
    )


if __name__ == "__main__":
    main()
