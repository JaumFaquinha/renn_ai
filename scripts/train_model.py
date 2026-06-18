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
from src.models.sector_model import (
    SectorModel,
    _DELTA_OUTLIER_THRESHOLD_S,
    _TRACK_POSITION_MIN,
    _TRACK_POSITION_MAX,
)

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
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Habilita tracking de experimentos via MLflow (requer: pip install mlflow)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=None,
        dest="mlflow_experiment",
        help="Nome do experimento MLflow (default: sector_model_{track_id})",
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
                # SELECT * para resiliência ao schema:
                # - Pré-migration P1: colunas multi-stat ainda não existem;
                #   listar explicitamente quebraria o query.
                # - Pós-migration P1: colunas vêm como NULL para dados
                #   antigos, automaticamente absorvidas como 0.0 pelo
                #   sector_model via s.get(f, 0.0).
                # Custo de bandwidth marginal — colunas extras (id, lap_id,
                # created_at) são <1% do payload total.
                .select("*")
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
        - mae: Mean Absolute Error na predição de delta_per_sector
        - r2: coeficiente de determinação (R²)
        - correlation: correlação de Pearson entre score e delta real
        - high_loss_precision: % dos top-10% setores por delta real
                               que o modelo ranqueia no top-10% por score

    O target de comparação é "delta_per_sector" (perda por mini-setor).
    Para dados históricos sem esse campo, o valor é computado retroativamente
    como a diferença entre delta_vs_best de setores consecutivos na mesma volta
    — mesma lógica do SectorModel.train().

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

    # Extrair setores com target correto (delta_per_sector)
    # Aplica a mesma lógica retroativa do SectorModel.train() para garantir
    # que os deltas comparados são os mesmos usados durante o treino.
    all_sectors: list[dict] = []
    all_targets: list[float] = []
    all_cars: list[str | None] = []
    n_outliers_filtered: int = 0

    for lap in lap_data:
        lap_sectors = lap.get("mini_sectors", [])
        lap_car = (lap.get("car_model") or "").strip() or None
        prev_dvb: float | None = None

        for sector in lap_sectors:
            # Filtro posicional (2026-06-15) — mesmo do SectorModel.train(),
            # mantém in-sample/CV consistente com a amostra de treino.
            pos = sector.get("track_position")
            if pos is not None:
                try:
                    pos_f = float(pos)
                    if pos_f < _TRACK_POSITION_MIN or pos_f > _TRACK_POSITION_MAX:
                        continue
                except (TypeError, ValueError):
                    pass

            # Usa `is not None` em vez de `in sector`: dados do Supabase carregam
            # todas as colunas selecionadas mesmo com valor NULL no banco
            # (mini-setores gravados antes da migração que introduziu
            # `delta_per_sector`). Mesma lógica aplicada em sector_model.train().
            if sector.get("delta_per_sector") is not None:
                target = float(sector["delta_per_sector"])
            elif sector.get("delta_vs_best") is not None:
                curr_dvb = float(sector["delta_vs_best"])
                if prev_dvb is None:
                    prev_dvb = curr_dvb
                    continue
                target = curr_dvb - prev_dvb
                prev_dvb = curr_dvb
            else:
                prev_dvb = None
                continue

            # Bug A (playbook §"Known issues"): aplicar o MESMO filtro de outliers
            # que SectorModel.train() aplica. Sem isso, mini-setores com target
            # ±160s (artefatos de reset do performanceMeter, primeira volta sem
            # iBestTime, ou snapshots fora de AC_LIVE) destroem R²/Pearson —
            # geram a métrica ilusória de R²≈0 mesmo em modelos saudáveis.
            if abs(target) > _DELTA_OUTLIER_THRESHOLD_S:
                n_outliers_filtered += 1
                # Sincroniza prev_dvb se possível, igual a SectorModel.train()
                dvb_raw = sector.get("delta_vs_best")
                if dvb_raw is not None:
                    prev_dvb = float(dvb_raw)
                continue

            all_sectors.append(sector)
            all_targets.append(target)
            all_cars.append(lap_car)

    if not all_sectors:
        return {}

    deltas_arr = np.array(all_targets, dtype=float)

    # IMPORTANTE: usa a saída BRUTA do GBR (model._model.predict) em vez de
    # model.predict_batch(). predict_batch() clipa o score em [0, 1] e o
    # multiplica por _max_delta — interface de anomaly-score para produção.
    # Para avaliar a *qualidade da regressão* (MAE/R²/Pearson), precisamos
    # da predição contínua, sem clipping. Avaliar a saída clipada subestima
    # severamente o ajuste real do modelo (ex: monza.pkl real R²=0.85,
    # mas clipped R²=0.17 — porque _max_delta=0.25s não consegue cobrir
    # deltas reais até 5s).
    # Usa o helper do próprio modelo — ele inclui as colunas one-hot do
    # car_model (2026-06-15) na ordem correta esperada pelo scaler.
    X_eval = np.array(
        [model._row_for_sector(s, c) for s, c in zip(all_sectors, all_cars)],
        dtype=float,
    )
    # _scaler é None para modelos novos (HistGBR, tree-based, escala-livre).
    if model._scaler is not None:
        X_eval = model._scaler.transform(X_eval)
    predicted_deltas = model._model.predict(X_eval)

    # scores_arr ainda é necessário para precision@10% (anomaly-score ranking)
    scores_arr = np.clip(predicted_deltas / max(model._max_delta, 1e-9), 0.0, 1.0)

    mae = float(mean_absolute_error(deltas_arr, predicted_deltas))
    r2 = float(r2_score(deltas_arr, predicted_deltas))

    # Correlação de Pearson entre predição bruta e delta real
    # (mais informativa que score clipped vs delta — captura a qualidade
    # da regressão na faixa completa de valores)
    if np.std(predicted_deltas) > 0 and np.std(deltas_arr) > 0:
        corr = float(np.corrcoef(predicted_deltas, deltas_arr)[0, 1])
    else:
        corr = 0.0

    # Precision@10%: dos setores com maior delta real, quantos o modelo detecta?
    n_top = max(1, len(deltas_arr) // 10)
    top_by_delta = set(np.argsort(deltas_arr)[-n_top:])
    top_by_score = set(np.argsort(scores_arr)[-n_top:])
    precision_at_10 = len(top_by_delta & top_by_score) / n_top

    # Baseline trivial: predizer sempre a média do target.
    # Se o GBR não bate este MAE, o modelo não está aprendendo nada útil.
    baseline_pred = float(np.mean(deltas_arr))
    baseline_mae = float(mean_absolute_error(deltas_arr, np.full_like(deltas_arr, baseline_pred)))

    return {
        "n_sectors": len(all_sectors),
        "n_outliers_filtered": n_outliers_filtered,
        "mae_s": round(mae, 4),
        "baseline_mae_s": round(baseline_mae, 4),
        "mae_improvement_vs_baseline": round((baseline_mae - mae) / baseline_mae, 4) if baseline_mae > 0 else 0.0,
        "r2": round(r2, 4),
        "pearson_correlation": round(corr, 4),
        "precision_at_10pct": round(precision_at_10, 4),
    }


# ---------------------------------------------------------------------------
# Cross-validation por volta (resolve avaliação in-sample)
# ---------------------------------------------------------------------------

def cross_validate_model(
    lap_data: list[dict],
    track_id: str,
    n_splits: int = 5,
) -> dict:
    """
    K-fold cross-validation agrupando por VOLTA (não por mini-setor).

    Por que por volta: mini-setores adjacentes da mesma volta são quase
    idênticos (track_position e dinâmica do carro variam suavemente).
    Misturar setores entre folds vaza o sinal e infla R²/Pearson.
    Referência: Roberts et al. (2017), *Cross-validation strategies for
    data with hierarchical structure*. Ecography 40.

    Em cada fold:
      - Treina um SectorModel novo com (k-1) grupos de voltas.
      - Avalia em todos os mini-setores das voltas separadas.
      - O filtro de outliers do train()/evaluate_model() é aplicado em ambos.

    O modelo final (salvo em data/models/{track}.pkl) ainda é treinado em
    100% dos dados — o CV serve apenas para reportar métrica honesta de
    generalização, não para selecionar o modelo final.

    Args:
        lap_data: voltas limpas (já filtradas por clutch corrompido).
        track_id: identificador da pista (passado ao SectorModel).
        n_splits: número de folds (default 5).

    Returns:
        Dict com mean ± std de MAE, R² e Pearson cross-fold.
        {} se voltas insuficientes para CV ou sklearn indisponível.
    """
    try:
        from sklearn.model_selection import KFold
        import numpy as np
    except ImportError:
        return {}

    if len(lap_data) < n_splits:
        logger.info(
            "Voltas insuficientes para CV %d-fold (%d disponíveis) — pulando CV",
            n_splits, len(lap_data),
        )
        return {}

    laps_arr = np.array(lap_data, dtype=object)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(laps_arr), start=1):
        train_laps = [laps_arr[i] for i in train_idx]
        test_laps = [laps_arr[i] for i in test_idx]

        fold_model = SectorModel(track_id=track_id)
        if not fold_model.train(train_laps):
            logger.warning(
                "Fold %d/%d falhou no treino — pulando", fold_idx, n_splits,
            )
            continue

        m = evaluate_model(fold_model, test_laps)
        if m and m.get("n_sectors", 0) > 0:
            fold_metrics.append(m)
            logger.debug(
                "Fold %d/%d: MAE=%.4f R²=%.4f Pearson=%.4f n=%d",
                fold_idx, n_splits,
                m["mae_s"], m["r2"], m["pearson_correlation"], m["n_sectors"],
            )

    if not fold_metrics:
        return {}

    def _agg(key: str) -> tuple[float, float]:
        vals = [fm[key] for fm in fold_metrics if key in fm]
        if not vals:
            return (0.0, 0.0)
        return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))

    mae_mean, mae_std = _agg("mae_s")
    r2_mean, r2_std = _agg("r2")
    corr_mean, corr_std = _agg("pearson_correlation")
    p10_mean, p10_std = _agg("precision_at_10pct")
    base_mean, _ = _agg("baseline_mae_s")

    return {
        "n_folds": len(fold_metrics),
        "n_splits_requested": n_splits,
        "mae_s_mean": mae_mean,
        "mae_s_std": mae_std,
        "baseline_mae_s_mean": base_mean,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "pearson_mean": corr_mean,
        "pearson_std": corr_std,
        "precision_at_10pct_mean": p10_mean,
        "precision_at_10pct_std": p10_std,
    }


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------

def _log_to_mlflow(
    mlflow,
    track_id: str,
    car_model: str | None,
    n_laps: int,
    n_clean_laps: int,
    model: "SectorModel",
    metrics: dict,
    cv_metrics: dict,
    model_path: str,
) -> None:
    """
    Registra um run no MLflow com params, métricas, feature importance e artefato.

    Chamado apenas quando --mlflow está ativo e o modelo foi salvo com sucesso.
    Falhas de tracking não interrompem o script (o modelo já foi salvo localmente).
    """
    try:
        run_name = f"{track_id}__{car_model or 'any'}"
        with mlflow.start_run(run_name=run_name):
            # --- Parâmetros fixos do modelo ---
            mlflow.log_params({
                "track_id": track_id,
                "car_model": car_model or "any",
                "target_field": "delta_per_sector",
                "n_laps_total": n_laps,
                "n_laps_clean": n_clean_laps,
                # Hiperparâmetros do GBR (espelham os valores hardcoded em sector_model.py)
                "gbr_n_estimators": 100,
                "gbr_max_depth": 4,
                "gbr_learning_rate": 0.1,
                "gbr_subsample": 0.8,
                "gbr_min_samples_leaf": 5,
                "gbr_loss": "huber",
                "gbr_alpha": 0.9,
            })

            # --- Tags descritivas ---
            mlflow.set_tags({
                "n_training_sectors": model.n_training_sectors,
                "n_discarded_clutch_laps": model.n_discarded_clutch_laps,
                "n_discarded_outlier_sectors": model.n_discarded_outlier_sectors,
            })

            # --- Métricas in-sample ---
            if metrics:
                in_sample = {
                    f"insample_{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and k != "n_sectors"
                }
                mlflow.log_metrics(in_sample)
                mlflow.log_metric("insample_n_sectors", metrics.get("n_sectors", 0))

            # --- Métricas de cross-validation ---
            if cv_metrics:
                cv_loggable = {
                    f"cv_{k}": v
                    for k, v in cv_metrics.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(cv_loggable)

            # --- Feature importance (top 20 como métricas individuais) ---
            for i, (feat, imp) in enumerate(model.feature_importance.items()):
                mlflow.log_metric(f"feat__{feat}", imp)
                if i >= 19:
                    break

            # --- Artefato: o .pkl do modelo ---
            mlflow.log_artifact(model_path, artifact_path="model")

        logger.info(
            "Run registrado no MLflow",
            extra={"run_name": run_name, "model_path": model_path},
        )
        print(f"  MLflow run registrado: '{run_name}' — execute 'mlflow ui' para visualizar.")

    except Exception as exc:
        logger.warning(
            "Falha ao registrar run no MLflow — modelo salvo localmente normalmente",
            extra={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # Windows: console default cp1252 não renderiza U+2500 (─) usado nos
    # separadores. Reconfiguração silenciosa para UTF-8 — não afeta Linux/macOS.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, Exception):
        pass

    args = parse_args()
    track_id: str = args.track
    car_model: str | None = args.car
    min_laps: int = args.min_laps

    # MLflow: configurar experimento se flag ativada
    _mlflow = None
    if args.mlflow:
        try:
            import mlflow
            _mlflow = mlflow
            experiment_name = args.mlflow_experiment or f"sector_model_{track_id}"
            _mlflow.set_experiment(experiment_name)
            logger.info("MLflow habilitado", extra={"experiment": experiment_name})
        except ImportError:
            logger.warning(
                "mlflow não instalado — tracking desabilitado. "
                "Execute: pip install mlflow"
            )
            _mlflow = None

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
            float(s.get(_CLUTCH_FIELD) or 0.0) > _CLUTCH_MAX
            for s in lap.get("mini_sectors", [])
        )
    ]
    metrics = evaluate_model(model, clean_laps_for_eval)

    # Cross-validation por volta — métrica honesta de generalização.
    # Roda APÓS o treino final (não substitui o modelo salvo) para reportar
    # ao usuário o quão bem o modelo generaliza para voltas não vistas.
    cv_metrics = cross_validate_model(clean_laps_for_eval, track_id, n_splits=5)

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
        print(f"  Setores descartados: {n_discarded_outliers} (|delta| > {_DELTA_OUTLIER_THRESHOLD_S:.0f}s)")
    print(f"  Mini-setores       : {model.n_training_sectors}")
    print()
    print(f"  ── In-sample (treino, otimista) ──")
    if metrics.get("n_outliers_filtered", 0) > 0:
        print(f"  Outliers filtrados : {metrics['n_outliers_filtered']} (|delta| > {_DELTA_OUTLIER_THRESHOLD_S:.0f}s)")
    print(f"  MAE (delta/setor)  : {metrics.get('mae_s', '—')}s")
    print(f"  Baseline MAE (mean): {metrics.get('baseline_mae_s', '—')}s")
    if metrics.get("baseline_mae_s"):
        improvement_pct = metrics.get("mae_improvement_vs_baseline", 0.0) * 100
        print(f"  Ganho vs baseline  : {improvement_pct:.1f}%")
    print(f"  R²                 : {metrics.get('r2', '—')}")
    print(f"  Correlação Pearson : {metrics.get('pearson_correlation', '—')}")
    print(f"  Precision@10%      : {metrics.get('precision_at_10pct', '—')}")
    print()
    if cv_metrics:
        print(f"  ── Cross-validation por volta ({cv_metrics['n_folds']}-fold, métrica honesta) ──")
        print(f"  MAE (delta/setor)  : {cv_metrics['mae_s_mean']}s ± {cv_metrics['mae_s_std']}")
        print(f"  R²                 : {cv_metrics['r2_mean']} ± {cv_metrics['r2_std']}")
        print(f"  Correlação Pearson : {cv_metrics['pearson_mean']} ± {cv_metrics['pearson_std']}")
        print(f"  Precision@10%      : {cv_metrics['precision_at_10pct_mean']} ± {cv_metrics['precision_at_10pct_std']}")
    else:
        print(f"  ── Cross-validation pulado (voltas insuficientes para 5-fold) ──")
    print("─" * 60)

    if args.verbose and model.feature_importance:
        print("\n  FEATURE IMPORTANCE (top 10):")
        for i, (feat, imp) in enumerate(model.feature_importance.items()):
            bar = "█" * int(imp * 50)
            print(f"  {i+1:>2}. {feat:<25} {imp:.4f}  {bar}")
            if i >= 9:
                break
        print()

    # Avisos de qualidade do modelo — preferir CV (métrica honesta) sobre
    # in-sample. In-sample só é usado se CV não rodou (poucas voltas).
    if cv_metrics:
        corr_for_judgment = cv_metrics.get("pearson_mean", 0.0)
        source_label = "CV"
    else:
        corr_for_judgment = metrics.get("pearson_correlation", 0.0)
        source_label = "in-sample"

    if corr_for_judgment < 0.5:
        print(
            f"  ⚠  Correlação {source_label} baixa ({corr_for_judgment:.2f}) — "
            f"colete mais voltas para melhorar a generalização."
        )
    elif corr_for_judgment >= 0.7:
        print(f"  ✓  Correlação {source_label} boa ({corr_for_judgment:.2f}) — modelo pronto para uso.")
    else:
        print(f"  •  Correlação {source_label} moderada ({corr_for_judgment:.2f}) — modelo utilizável, com ressalvas.")

    # Sanity-check: se o ganho do GBR sobre o baseline (predizer média) é
    # marginal, o modelo não está extraindo sinal real das features.
    improvement = metrics.get("mae_improvement_vs_baseline", 0.0)
    if improvement < 0.10:
        print(
            f"  ⚠  Ganho marginal sobre baseline ({improvement*100:.1f}%) — "
            f"o modelo está aprendendo pouco além da média do delta."
        )

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

    # 5. MLflow tracking (opt-in via --mlflow)
    if _mlflow is not None and saved:
        _log_to_mlflow(
            mlflow=_mlflow,
            track_id=track_id,
            car_model=car_model,
            n_laps=len(laps),
            n_clean_laps=len(clean_laps_for_eval),
            model=model,
            metrics=metrics,
            cv_metrics=cv_metrics,
            model_path=str(model_path),
        )


if __name__ == "__main__":
    main()
