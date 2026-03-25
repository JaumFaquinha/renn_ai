"""
SectorModel — Modelo de recompensa por setor (ML leve).

Aprende a relação entre inputs de telemetria e perda de tempo (delta_vs_best)
usando GradientBoostingRegressor. Treinado por pista com as próprias voltas
gravadas — auto-supervisionado via reward = -delta_vs_best.

Interface pública:
    model = SectorModel(track_id="monza")
    ok    = model.train(lap_data_list)      # list[dict] do LapRecorder
    score = model.predict(sector)           # float 0.0 (normal) → 1.0 (severo)
    model.save(path)                        # persiste joblib
    model.load(path)                        # carrega joblib

Referência arquitetural: CLAUDE.md §2, §4.5
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Features usadas pelo modelo — subset estável do schema CLAUDE.md §4.5
# track_position é crítico: normaliza o comportamento esperado por zona da pista
# ---------------------------------------------------------------------------
_FEATURE_FIELDS: list[str] = [
    "track_position",
    "throttle", "brake", "steering", "clutch",
    "gear", "rpms",
    "speed_kmh", "speed_min",
    "gforce_x", "gforce_y", "gforce_z",
    "local_ang_vel_x", "local_ang_vel_y", "local_ang_vel_z",
    "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
    "tc_active", "abs_active",
    "brake_bias", "surface_grip",
]

_TARGET_FIELD: str = "delta_vs_best"

# Mínimo de setores para garantir generalização mínima
_MIN_SECTORS_TO_TRAIN: int = 200   # ≈ 2 voltas completas


class SectorModel:
    """
    Modelo de recompensa por setor treinado por pista.

    Conceito de reward:
        reward(setor) = -delta_vs_best   →   setor sem perda = reward alto
        O modelo aprende a prever o delta esperado dado os inputs do piloto.
        A difere0nça entre delta previsto e delta real detecta padrões
        que os 5 heurísticos fixos do PatternDetector não capturam.

    Anomaly score:
        predict() retorna score 0.0–1.0, onde:
        - 0.0 = setor dentro do padrão aprendido (sem perda prevista)
        - 1.0 = maior perda prevista observada durante o treino (p95)

    Dependência: scikit-learn >= 1.3 (GradientBoostingRegressor, StandardScaler)
                 joblib (incluído no scikit-learn)
    """

    def __init__(self, track_id: str) -> None:
        """
        Args:
            track_id: identificador da pista (ex: 'monza', 'spa').
                      Usado como chave de persistência e validação.
        """
        self._track_id = track_id
        self._model = None
        self._scaler = None
        self._is_trained: bool = False
        self._max_delta: float = 1.0        # p95 do delta de treino — normaliza score
        self._feature_importance: dict[str, float] = {}
        self._n_training_sectors: int = 0

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True se o modelo foi treinado ou carregado com sucesso."""
        return self._is_trained

    @property
    def feature_importance(self) -> dict[str, float]:
        """
        Importância relativa de cada feature (soma = 1.0).

        Disponível apenas após treino. Retorna dict vazio se não treinado.
        """
        return dict(self._feature_importance)

    @property
    def n_training_sectors(self) -> int:
        """Número de mini-setores usados no treino."""
        return self._n_training_sectors

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def train(self, lap_data: list[dict]) -> bool:
        """
        Treina o modelo com histórico de voltas.

        Extrai todos os mini-setores de todas as voltas fornecidas,
        treina um GradientBoostingRegressor em (features → delta_vs_best),
        e calibra a normalização do score de anomalia.

        Args:
            lap_data: lista de dicts no formato retornado pelo LapRecorder.
                      Cada dict deve ter a chave 'mini_sectors': list[dict].

        Returns:
            True se o treino foi concluído com sucesso, False caso contrário.
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            import numpy as np
        except ImportError:
            logger.error(
                "scikit-learn não instalado — execute: pip install scikit-learn>=1.3",
                extra={"track_id": self._track_id},
            )
            return False

        # Achatar todas as voltas em uma lista plana de setores
        sectors: list[dict] = []
        for lap in lap_data:
            sectors.extend(lap.get("mini_sectors", []))

        if len(sectors) < _MIN_SECTORS_TO_TRAIN:
            logger.warning(
                "Setores insuficientes para treino",
                extra={
                    "track_id": self._track_id,
                    "sectors": len(sectors),
                    "min_required": _MIN_SECTORS_TO_TRAIN,
                },
            )
            return False

        # Extrair X (features) e y (target)
        X_rows, y_values = [], []
        for sector in sectors:
            if _TARGET_FIELD not in sector:
                continue
            row = [float(sector.get(f, 0.0)) for f in _FEATURE_FIELDS]
            X_rows.append(row)
            y_values.append(float(sector[_TARGET_FIELD]))

        if len(X_rows) < _MIN_SECTORS_TO_TRAIN:
            logger.warning(
                "Setores com target válido insuficientes",
                extra={"count": len(X_rows)},
            )
            return False

        X = np.array(X_rows, dtype=float)
        y = np.array(y_values, dtype=float)

        # Normalizar features (importante: track_position tem escala 0–1,
        # mas rpms pode ser 8000+ — StandardScaler equaliza as distribuições)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # GradientBoosting: captura relações não-lineares entre posição,
        # inputs e delta. Subsample=0.8 reduz overfitting com poucas voltas.
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler
        self._is_trained = True
        self._n_training_sectors = len(X_rows)

        # p95 dos deltas positivos → ancora a normalização do score
        positive_deltas = y[y > 0]
        self._max_delta = float(np.percentile(positive_deltas, 95)) if len(positive_deltas) > 0 else 1.0

        # Feature importance ordenada
        raw_importance = dict(zip(_FEATURE_FIELDS, model.feature_importances_))
        self._feature_importance = {
            k: round(float(v), 4)
            for k, v in sorted(raw_importance.items(), key=lambda x: -x[1])
        }

        top_features = list(self._feature_importance.items())[:3]
        logger.info(
            "SectorModel treinado com sucesso",
            extra={
                "track_id": self._track_id,
                "sectors": self._n_training_sectors,
                "laps": len(lap_data),
                "max_delta_p95": round(self._max_delta, 3),
                "top_features": str(top_features),
            },
        )
        return True

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, sector: dict) -> float:
        """
        Prediz o score de anomalia de performance do setor.

        Args:
            sector: mini-setor no formato schema §4.5

        Returns:
            Score de anomalia 0.0–1.0:
            - 0.0 → setor dentro do padrão aprendido
            - 1.0 → perda máxima prevista (calibrada no p95 do treino)
            Retorna 0.0 se o modelo não estiver treinado (fail-safe silencioso).
        """
        if not self._is_trained:
            return 0.0

        try:
            import numpy as np
            features = np.array(
                [[float(sector.get(f, 0.0)) for f in _FEATURE_FIELDS]],
                dtype=float,
            )
            features_scaled = self._scaler.transform(features)
            predicted_delta = float(self._model.predict(features_scaled)[0])
            score = max(0.0, min(1.0, predicted_delta / self._max_delta))
            return round(score, 4)
        except Exception as exc:
            logger.debug("Falha na predição do SectorModel", extra={"error": str(exc)})
            return 0.0

    def predict_batch(self, sectors: list[dict]) -> list[float]:
        """
        Prediz scores de anomalia para uma lista de mini-setores em lote.

        Mais eficiente que chamar predict() individualmente quando o modelo
        está treinado — usa uma única chamada ao numpy/sklearn.

        Args:
            sectors: lista de mini-setores no formato schema §4.5

        Returns:
            Lista de scores 0.0–1.0 na mesma ordem dos setores de entrada.
        """
        if not self._is_trained or not sectors:
            return [0.0] * len(sectors)

        try:
            import numpy as np
            X = np.array(
                [[float(s.get(f, 0.0)) for f in _FEATURE_FIELDS] for s in sectors],
                dtype=float,
            )
            X_scaled = self._scaler.transform(X)
            predictions = self._model.predict(X_scaled)
            return [
                round(max(0.0, min(1.0, float(p) / self._max_delta)), 4)
                for p in predictions
            ]
        except Exception as exc:
            logger.debug(
                "Falha na predição em lote do SectorModel",
                extra={"error": str(exc)},
            )
            return [0.0] * len(sectors)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: str) -> bool:
        """
        Persiste o modelo treinado em disco via joblib.

        Args:
            path: caminho completo do arquivo .pkl (diretório criado se necessário)

        Returns:
            True se salvo com sucesso, False caso contrário.
        """
        if not self._is_trained:
            logger.warning(
                "SectorModel não treinado — nada a salvar",
                extra={"track_id": self._track_id},
            )
            return False

        try:
            import joblib
            payload = {
                "track_id": self._track_id,
                "model": self._model,
                "scaler": self._scaler,
                "max_delta": self._max_delta,
                "feature_importance": self._feature_importance,
                "feature_fields": _FEATURE_FIELDS,
                "n_training_sectors": self._n_training_sectors,
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, path, compress=3)
            logger.info(
                "SectorModel salvo",
                extra={"path": path, "track_id": self._track_id},
            )
            return True
        except Exception as exc:
            logger.error(
                "Falha ao salvar SectorModel",
                extra={"error": str(exc), "path": path},
            )
            return False

    def load(self, path: str) -> bool:
        """
        Carrega modelo do disco.

        Valida que o track_id do arquivo corresponde ao desta instância.

        Args:
            path: caminho do arquivo .pkl gerado por save()

        Returns:
            True se carregado com sucesso, False caso contrário.
        """
        try:
            import joblib
            payload = joblib.load(path)

            loaded_track = payload.get("track_id")
            if loaded_track != self._track_id:
                logger.warning(
                    "track_id do modelo não corresponde",
                    extra={"expected": self._track_id, "loaded": loaded_track},
                )

            self._model = payload["model"]
            self._scaler = payload["scaler"]
            self._max_delta = payload.get("max_delta", 1.0)
            self._feature_importance = payload.get("feature_importance", {})
            self._n_training_sectors = payload.get("n_training_sectors", 0)
            self._is_trained = True

            logger.info(
                "SectorModel carregado",
                extra={
                    "path": path,
                    "track_id": self._track_id,
                    "n_training_sectors": self._n_training_sectors,
                },
            )
            return True
        except FileNotFoundError:
            logger.info(
                "Arquivo de modelo não encontrado",
                extra={"path": path},
            )
            return False
        except Exception as exc:
            logger.warning(
                "Falha ao carregar SectorModel",
                extra={"error": str(exc), "path": path},
            )
            return False
