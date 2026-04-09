"""
train_model.py — Script de treino offline do SectorModel.

Carrega voltas gravadas em JSON (data/laps/) e/ou Supabase,
treina um SectorModel por pista, avalia métricas de confiabilidade
e salva em data/models/{track_id}.pkl.

Uso:
    python scripts/train_model.py --track monza
    python scripts/train_model.py --track monza --car ferrari_488_gt3
    python scripts/train_model.py --track monza --min-laps 10 --verbose

Requer:
    - scikit-learn >= 1.3  (pip install scikit-learn)
    - Voltas gravadas em data/laps/ ou Supabase habilitado em .env
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import LAPS_DIR, MODELS_DIR, LOG_LEVEL, SUPABASE_ENABLED
from src.models.sector_model import SectorModel

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")

# Número mínimo de voltas para iniciar treino
_DEFAULT_MIN_LAPS: int = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina SectorModel com voltas gravadas",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--track",
        required=True,
        help="ID da pista (ex: monza, spa, nurburgring_gp)",
    )
    parser.add_argument(
        "--car",
        default=None,
        help="Filtrar por modelo de carro (opcional)",
    )
    parser.add_argument(
        "--min-laps",
        type=int,
        default=_DEFAULT_MIN_LAPS,
        help=f"Mínimo de voltas para treinar (default: {_DEFAULT_MIN_LAPS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe feature importance detalhada após treino",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

def load_laps_from_json(
    track_id: str,
    car_model: str | None = None,
) -> list[dict]:
    """
    Carrega voltas gravadas em JSON de LAPS_DIR.

    Filtra por track_id e opcionalmente por car_model.
    Ignora arquivos com formato inesperado sem interromper o carregamento.

    Args:
        track_id: identificador da pista
        car_model: filtro opcional de carro

    Returns:
        Lista de dicts de volta compatível com SectorModel.train()
    """
    laps: list[dict] = []
    pattern = f"{track_id}_*.json"
    files = sorted(LAPS_DIR.glob(pattern))

    if not files:
        logger.info(
            "Nenhum arquivo JSON encontrado",
            extra={"dir": str(LAPS_DIR), "pattern": pattern},
        )
        return laps

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                lap = json.load(f)

            if lap.get("track_id") != track_id:
                continue
            if car_model and lap.get("car_model") != car_model:
                continue
            if not lap.get("mini_sectors"):
                continue

            laps.append(lap)
        except Exception as exc:
            logger.warning(
                "Falha ao carregar arquivo de volta",
                extra={"file": path.name, "error": str(exc)},
            )

    logger.info(
        "Voltas carregadas do JSON",
        extra={"count": len(laps), "track_id": track_id, "dir": str(LAPS_DIR)},
    )
    return laps


def load_laps_from_supabase(
    track_id: str,
    car_model: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Carrega voltas do Supabase (se habilitado).

    Reconstrói o formato list[dict] compatível com SectorModel.train()
    a partir das tabelas laps + mini_sectors.

    Args:
        track_id: identificador da pista
        car_model: filtro opcional de carro
        limit: máximo de voltas a carregar

    Returns:
        Lista de dicts de volta.
    """
    if not SUPABASE_ENABLED:
        return []

    try:
        from src.persistence.supabase_client import SupabaseClient
        client = SupabaseClient()
        if not client.is_enabled:
            return []

        sb = client.get_client()

        # Busca sessões da pista
        sessions_q = (
            sb.table("sessions")
            .select("id")
            .eq("track_id", track_id)
        )
        if car_model:
            sessions_q = sessions_q.eq("car_model", car_model)
        sessions_result = sessions_q.execute()

        if not sessions_result or not sessions_result.data:
            return []

        session_ids = [s["id"] for s in sessions_result.data]

        # Busca voltas válidas dessas sessões
        laps_result = (
            sb.table("laps")
            .select("id, lap_number, lap_time_ms, tyre_compound")
            .in_("session_id", session_ids)
            .eq("is_valid", True)
            .order("lap_time_ms")
            .limit(limit)
            .execute()
        )

        if not laps_result or not laps_result.data:
            return []

        # Para cada volta, busca os mini-setores
        laps: list[dict] = []
        for lap_row in laps_result.data:
            sectors_result = (
                sb.table("mini_sectors")
                .select(
                    "track_position, delta_vs_best, throttle, brake, steering, "
                    "gear, rpms, clutch, speed_kmh, speed_min, "
                    "gforce_x, gforce_y, gforce_z, "
                    "local_ang_vel_x, local_ang_vel_y, local_ang_vel_z, "
                    "wheel_slip_fl, wheel_slip_fr, wheel_slip_rl, wheel_slip_rr, "
                    "tc_active, abs_active, drs_active, drs_available, "
                    "brake_bias, surface_grip, air_temp, road_temp"
                )
                .eq("lap_id", lap_row["id"])
                .order("track_position")
                .execute()
            )

            if sectors_result and sectors_result.data:
                laps.append({
                    "lap_number": lap_row["lap_number"],
                    "lap_time_ms": lap_row["lap_time_ms"],
                    "tyre_compound": lap_row.get("tyre_compound", "unknown"),
                    "mini_sectors": sectors_result.data,
                })

        logger.info(
            "Voltas carregadas do Supabase",
            extra={"count": len(laps), "track_id": track_id},
        )
        return laps

    except Exception as exc:
        logger.warning(
            "Falha ao carregar voltas do Supabase",
            extra={"error": str(exc)},
        )
        return []


# ---------------------------------------------------------------------------
# Avaliação de confiabilidade
# ---------------------------------------------------------------------------

def evaluate_model(model: SectorModel, lap_data: list[dict]) -> dict:
    """
    Avalia a confiabilidade do modelo treinado.

    Métricas calculadas:
        - mae: Mean Absolute Error na predição de delta_vs_best
        - r2: coeficiente de determinação (R²)
        - correlation: correlação de Pearson entre score e delta real
        - high_loss_precision: % dos top-10% setores por delta real
                               que o modelo ranqueia no top-10% por score

    Args:
        model: SectorModel já treinado
        lap_data: mesmas voltas usadas no treino (in-sample)

    Returns:
        Dict com métricas de confiabilidade.
    """
    try:
        from sklearn.metrics import mean_absolute_error, r2_score
        import numpy as np
    except ImportError:
        return {}

    all_sectors = []
    for lap in lap_data:
        all_sectors.extend(lap.get("mini_sectors", []))

    if not all_sectors:
        return {}

    scores = model.predict_batch(all_sectors)
    deltas = [float(s.get("delta_vs_best", 0.0)) for s in all_sectors]

    scores_arr = np.array(scores)
    deltas_arr = np.array(deltas)

    # Predição bruta = score * max_delta (desfaz normalização)
    predicted_deltas = scores_arr * model._max_delta

    mae = float(mean_absolute_error(deltas_arr, predicted_deltas))
    r2 = float(r2_score(deltas_arr, predicted_deltas))

    # Correlação de Pearson entre score e delta
    if np.std(scores_arr) > 0 and np.std(deltas_arr) > 0:
        corr = float(np.corrcoef(scores_arr, deltas_arr)[0, 1])
    else:
        corr = 0.0

    # Precision@10%: dos setores com maior delta real, quantos o modelo detecta?
    n_top = max(1, len(deltas_arr) // 10)
    top_by_delta = set(np.argsort(deltas_arr)[-n_top:])
    top_by_score = set(np.argsort(scores_arr)[-n_top:])
    precision_at_10 = len(top_by_delta & top_by_score) / n_top

    return {
        "n_sectors": len(all_sectors),
        "mae_s": round(mae, 4),
        "r2": round(r2, 4),
        "pearson_correlation": round(corr, 4),
        "precision_at_10pct": round(precision_at_10, 4),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    track_id: str = args.track
    car_model: str | None = args.car
    min_laps: int = args.min_laps

    logger.info(
        "Iniciando treino",
        extra={"track_id": track_id, "car_model": car_model or "any"},
    )

    # 1. Carregar voltas (JSON + Supabase)
    laps = load_laps_from_json(track_id, car_model)
    laps += load_laps_from_supabase(track_id, car_model)

    # Deduplica por lap_number + lap_time_ms (evita duplicatas JSON/Supabase)
    seen: set[tuple] = set()
    unique_laps = []
    for lap in laps:
        key = (lap.get("lap_number"), lap.get("lap_time_ms"))
        if key not in seen:
            seen.add(key)
            unique_laps.append(lap)
    laps = unique_laps

    if len(laps) < min_laps:
        logger.error(
            "Voltas insuficientes para treino",
            extra={
                "loaded": len(laps),
                "min_required": min_laps,
                "track_id": track_id,
            },
        )
        print(
            f"\n[ERRO] Apenas {len(laps)} volta(s) encontrada(s) para '{track_id}'.\n"
            f"Mínimo necessário: {min_laps}.\n"
            f"Grave mais voltas com run_session.py antes de treinar."
        )
        sys.exit(1)

    logger.info("Total de voltas para treino: %d", len(laps))

    # 2. Treinar modelo
    model = SectorModel(track_id=track_id)
    success = model.train(laps)

    if not success:
        logger.error("Treino falhou — verifique os logs acima")
        sys.exit(1)

    # 3. Avaliar confiabilidade — usa apenas voltas limpas (sem clutch corrompido)
    #    para evitar métricas distorcidas por dados fora do range do scaler.
    _CLUTCH_FIELD = "clutch"
    _CLUTCH_MAX = 1.0
    clean_laps_for_eval = [
        lap for lap in laps
        if not any(
            float(s.get(_CLUTCH_FIELD, 0.0)) > _CLUTCH_MAX
            for s in lap.get("mini_sectors", [])
        )
    ]
    metrics = evaluate_model(model, clean_laps_for_eval)

    n_discarded_clutch = model.n_discarded_clutch_laps
    n_discarded_outliers = model.n_discarded_outlier_sectors

    print("\n" + "─" * 60)
    print(f"  SECTOR MODEL — {track_id.upper()}")
    print("─" * 60)
    print(f"  Voltas carregadas  : {len(laps)}")
    if n_discarded_clutch > 0:
        print(f"  Voltas descartadas : {n_discarded_clutch} (clutch corrompido)")
    print(f"  Voltas usadas      : {len(clean_laps_for_eval)}")
    if n_discarded_outliers > 0:
        print(f"  Setores descartados: {n_discarded_outliers} (|delta| > 60s)")
    print(f"  Mini-setores       : {model.n_training_sectors}")
    print(f"  MAE (delta)        : {metrics.get('mae_s', '—')}s")
    print(f"  R²                 : {metrics.get('r2', '—')}")
    print(f"  Correlação Pearson : {metrics.get('pearson_correlation', '—')}")
    print(f"  Precision@10%      : {metrics.get('precision_at_10pct', '—')}")
    print("─" * 60)

    if args.verbose and model.feature_importance:
        print("\n  FEATURE IMPORTANCE (top 10):")
        for i, (feat, imp) in enumerate(model.feature_importance.items()):
            bar = "█" * int(imp * 50)
            print(f"  {i+1:>2}. {feat:<25} {imp:.4f}  {bar}")
            if i >= 9:
                break
        print()

    # Avisos de qualidade do modelo
    corr = metrics.get("pearson_correlation", 0.0)
    if corr < 0.5:
        print(
            f"  ⚠  Correlação baixa ({corr:.2f}) — colete mais voltas para melhorar o modelo."
        )
    elif corr >= 0.7:
        print(f"  ✓  Correlação boa ({corr:.2f}) — modelo pronto para uso.")

    # 4. Salvar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{track_id}.pkl"
    saved = model.save(str(model_path))

    if saved:
        print(f"\n  Modelo salvo em: {model_path}")
    else:
        print("\n  [ERRO] Falha ao salvar o modelo.")
        sys.exit(1)

    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
