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
#
# REMOVIDO (2026-04-08, relatório de validação v2):
#   brake_bias — constante de setup por sessão, não varia por mini-setor.
#                Alta importância (13.5%) era artefato de data leakage por sessão,
#                não correlação real com performance do piloto.
#
# REMOVIDO (2026-04-14, relatório de validação v3):
#   surface_grip — constante em todas as sessões analisadas (mean=1.0, std≈0).
#                  Feature importance = 0.0 em todos os modelos treinados.
#                  Ocupa posição no scaler sem qualquer ganho analítico.
#
# REMOVIDO (2026-06-15, validação MLflow v7):
#   clutch — em datasets cross-car, atuava como proxy de sessão/carro
#            (aidAutoClutch varia entre carros). Importância 4.5% com
#            sensibilidade dominante (+20ms) era leakage, não sinal de
#            pilotagem. Validação v7 (CV R² caiu para 0.20) confirmou.
# ---------------------------------------------------------------------------
_FEATURE_FIELDS: list[str] = [
    "track_position",
    # === Inputs do piloto: média + max + min + std (Proposal P1, 2026-04-25) ===
    # Adição da tripla (max, min, std) recupera a dinâmica intra-setor que a
    # média sozinha aniquilava. Esperado: importance dos inputs sobe de <1%
    # para ~10–25% após retreino com novos dados gravados.
    "throttle", "throttle_max", "throttle_min", "throttle_std",
    "brake", "brake_max", "brake_min", "brake_std",
    "steering", "steering_max", "steering_min", "steering_std",
    "gear", "rpms",
    "speed_kmh", "speed_min",
    "gforce_x", "gforce_y", "gforce_z",
    "local_ang_vel_x", "local_ang_vel_y", "local_ang_vel_z",
    "wheel_slip_fl", "wheel_slip_fl_max", "wheel_slip_fl_min", "wheel_slip_fl_std",
    "wheel_slip_fr", "wheel_slip_fr_max", "wheel_slip_fr_min", "wheel_slip_fr_std",
    "wheel_slip_rl", "wheel_slip_rl_max", "wheel_slip_rl_min", "wheel_slip_rl_std",
    "wheel_slip_rr", "wheel_slip_rr_max", "wheel_slip_rr_min", "wheel_slip_rr_std",
    "tc_active", "tc_active_max", "tc_active_min", "tc_active_std",
    "abs_active", "abs_active_max", "abs_active_min", "abs_active_std",
]

# ---------------------------------------------------------------------------
# Target do modelo (2026-04-14, relatório de validação v3)
#
# ALTERADO de "delta_vs_best" para "delta_per_sector".
#
# Por quê: delta_vs_best é um delta ACUMULADO desde o início da volta
# (= performanceMeter do AC). Usá-lo como target ensina o modelo "onde na
# pista o delta costuma ser alto" — não "o que o piloto fez de errado aqui".
# Isso causava:
#   1. Scores inflados no final da volta (0.69–1.00 mesmo sem erros)
#   2. Inversão de tc_active e wheel_slip (correlação negativa vs o esperado)
#
# delta_per_sector = delta_vs_best[último snapshot] − delta_vs_best[primeiro snapshot]
# dentro do mini-setor. Captura apenas a perda de tempo ocorrida NESTE setor.
# Calculado pelo SectorAggregator (novos dados) ou retroativamente aqui
# via diffs consecutivos entre mini-setores da mesma volta (dados históricos).
# ---------------------------------------------------------------------------
_TARGET_FIELD: str = "delta_per_sector"

# Mínimo de setores para garantir generalização mínima
_MIN_SECTORS_TO_TRAIN: int = 200   # ≈ 2 voltas completas

# ---------------------------------------------------------------------------
# Thresholds de validação de dados
# ---------------------------------------------------------------------------
# Clutch da Shared Memory deve ser 0.0–1.0. Valores acima indicam offset errado.
_CLUTCH_MAX_VALUE: float = 1.0

# Limiar de outlier para o TARGET.
#
# Com delta_vs_best (cumulativo): era 60.0 s — primeira volta sem referência
# podia chegar a centenas de segundos.
#
# Com delta_per_sector (por mini-setor): 5.0 s é conservador. Um mini-setor
# cobre ~1% da pista (≈1.1 s em Monza). Perder 5 s num único mini-setor
# implica velocidade zero ou crash — claramente um artefato de dados (ex:
# reset de performanceMeter ao cruzar a linha de chegada com sessão anterior,
# ou primeira volta sem referência de best lap).
_DELTA_OUTLIER_THRESHOLD_S: float = 5.0


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
        self._n_discarded_clutch_laps: int = 0
        self._n_discarded_outlier_sectors: int = 0
        # One-hot de car_model (2026-06-15): em dataset cross-car, sem essa
        # feature o GBR usa rpms/clutch/speed_min como proxy do carro, o que
        # destrói o sinal de pilotagem. Lista ordenada dos carros vistos no
        # treino; predict() faz lookup; carro desconhecido → vetor zerado
        # (modelo cai de volta para a média entre carros).
        self._car_models: list[str] = []

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

    @property
    def n_discarded_clutch_laps(self) -> int:
        """Número de voltas descartadas por clutch corrompido no último treino."""
        return self._n_discarded_clutch_laps

    @property
    def n_discarded_outlier_sectors(self) -> int:
        """Número de setores descartados por |delta_vs_best| excessivo no último treino."""
        return self._n_discarded_outlier_sectors

    @property
    def car_models(self) -> list[str]:
        """Lista ordenada de car_models conhecidos pelo modelo (one-hot columns)."""
        return list(self._car_models)

    # ------------------------------------------------------------------
    # Helpers de feature building
    # ------------------------------------------------------------------

    def _row_for_sector(self, sector: dict, car_model: str | None) -> list[float]:
        """
        Monta a linha de features para um mini-setor:
        [_FEATURE_FIELDS values...] + [one-hot car_model...]

        Carro desconhecido → todas as colunas one-hot ficam 0.0 (fallback
        para média entre carros, sem quebrar a predição).
        """
        row = [float(sector.get(f) or 0.0) for f in _FEATURE_FIELDS]
        for cm in self._car_models:
            row.append(1.0 if cm == car_model else 0.0)
        return row

    @property
    def _all_feature_names(self) -> list[str]:
        """Nome completo das colunas (base + one-hot) — usado em feature_importance."""
        return list(_FEATURE_FIELDS) + [f"car__{cm}" for cm in self._car_models]

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

        # ------------------------------------------------------------------
        # 1. Filtro de voltas: descartar voltas com clutch corrompido
        #
        # Clutch deve ser 0.0–1.0 (AC SDK). Valores acima de _CLUTCH_MAX_VALUE
        # indicam offset errado na leitura da Shared Memory — identificado
        # no relatório de validação 2026-04-08. A volta inteira é descartada
        # porque o scaler aprende com os dados de treino e um clutch fora do
        # range distorce a escala de toda a feature.
        # ------------------------------------------------------------------
        clean_laps: list[dict] = []
        discarded_clutch: int = 0
        for lap in lap_data:
            sectors_in_lap = lap.get("mini_sectors", [])
            if any(
                float(s.get("clutch") or 0.0) > _CLUTCH_MAX_VALUE
                for s in sectors_in_lap
            ):
                discarded_clutch += 1
                logger.warning(
                    "Volta descartada — clutch corrompido (>%.1f)",
                    _CLUTCH_MAX_VALUE,
                    extra={
                        "track_id": self._track_id,
                        "lap_number": lap.get("lap_number"),
                        "clutch_max_found": max(
                            (float(s.get("clutch") or 0.0) for s in sectors_in_lap),
                            default=0.0,
                        ),
                    },
                )
            else:
                clean_laps.append(lap)

        if discarded_clutch > 0:
            logger.info(
                "Filtragem de clutch: %d volta(s) descartada(s), %d mantida(s)",
                discarded_clutch,
                len(clean_laps),
                extra={"track_id": self._track_id},
            )

        # ------------------------------------------------------------------
        # 2. Extração do target: delta_per_sector
        #
        # Estratégia de extração (em ordem de prioridade):
        #   a) Se o setor já tem "delta_per_sector" (gravado pelo SectorAggregator
        #      a partir da versão 2026-04-14), usa diretamente.
        #   b) Se tem "delta_vs_best" mas não "delta_per_sector" (dados históricos
        #      gravados antes da atualização), computa retroativamente como a
        #      diferença entre o valor atual e o do setor anterior na mesma volta:
        #      Δ = delta_vs_best[i] - delta_vs_best[i-1]
        #      O primeiro setor de cada volta é descartado (sem referência prévia).
        #   c) Sem target disponível: setor descartado silenciosamente.
        #
        # Filtro de outliers: |delta_per_sector| > _DELTA_OUTLIER_THRESHOLD_S
        # indica reset do performanceMeter entre voltas ou dados corrompidos.
        # Com o target por-setor (≈1% da pista ≈ 1.1s em Monza), 5 s é
        # conservador o suficiente para filtrar artefatos sem perder dados reais.
        # ------------------------------------------------------------------
        # Descobrir car_models presentes — define as colunas one-hot.
        # Ordem alfabética determinística (necessária para o scaler ser
        # reprodutível). Filtra None/"unknown" sem descartar a volta — vira
        # vetor zerado e o GBR usa as features de base.
        unique_cars = {
            (lap.get("car_model") or "").strip()
            for lap in clean_laps
        }
        unique_cars.discard("")
        unique_cars.discard("unknown")
        self._car_models = sorted(unique_cars)

        X_rows, y_values = [], []
        discarded_outliers: int = 0

        for lap in clean_laps:
            lap_sectors = lap.get("mini_sectors", [])
            lap_car = (lap.get("car_model") or "").strip() or None
            prev_dvb: Optional[float] = None  # delta_vs_best do setor anterior nesta volta

            for sector in lap_sectors:
                # --- Determinar o target ---
                # Importante: usar `is not None` em vez de `in sector`. Dados vindos
                # do Supabase sempre carregam todas as colunas selecionadas, mesmo
                # quando o valor é NULL no banco — o que ocorre em mini-setores
                # gravados antes da migração que introduziu `delta_per_sector`
                # (commit a425202). Sem essa proteção, `float(None)` levanta TypeError.
                if sector.get("delta_per_sector") is not None:
                    # Formato novo: campo já computado pelo SectorAggregator
                    target = float(sector["delta_per_sector"])
                elif sector.get("delta_vs_best") is not None:
                    # Formato histórico: computar retroativamente
                    curr_dvb = float(sector["delta_vs_best"])
                    if prev_dvb is None:
                        # Primeiro setor da volta: sem referência anterior
                        prev_dvb = curr_dvb
                        continue
                    target = curr_dvb - prev_dvb
                    prev_dvb = curr_dvb
                else:
                    # Sem target disponível: descartado
                    prev_dvb = None
                    continue

                # --- Filtrar outlier ---
                if abs(target) > _DELTA_OUTLIER_THRESHOLD_S:
                    discarded_outliers += 1
                    prev_dvb = sector.get("delta_vs_best", None)
                    if prev_dvb is not None:
                        prev_dvb = float(prev_dvb)
                    continue

                # `or 0.0` (não apenas default) porque Supabase retorna chaves
                # COM valor None para colunas NULL no banco — o default do .get()
                # só dispara se a chave estiver ausente. Mesma armadilha já
                # corrigida em delta_per_sector linha ~261. Sem isso, dados
                # pré-migration P1 (multi-stats NULL) provocam TypeError.
                row = self._row_for_sector(sector, lap_car)
                X_rows.append(row)
                y_values.append(target)

        # Verificação antecipada de volume (antes de instanciar numpy/sklearn)
        total_sectors_seen = sum(len(l.get("mini_sectors", [])) for l in clean_laps)
        if total_sectors_seen < _MIN_SECTORS_TO_TRAIN:
            logger.warning(
                "Setores insuficientes para treino após filtragem",
                extra={
                    "track_id": self._track_id,
                    "sectors": total_sectors_seen,
                    "min_required": _MIN_SECTORS_TO_TRAIN,
                },
            )
            return False

        if discarded_outliers > 0:
            logger.info(
                "Filtragem de outliers: %d setor(es) descartado(s) (|delta_per_sector| > %.1fs)",
                discarded_outliers,
                _DELTA_OUTLIER_THRESHOLD_S,
                extra={"track_id": self._track_id},
            )

        if len(X_rows) < _MIN_SECTORS_TO_TRAIN:
            logger.warning(
                "Setores com target válido insuficientes após filtragem",
                extra={
                    "count": len(X_rows),
                    "min_required": _MIN_SECTORS_TO_TRAIN,
                    "discarded_outliers": discarded_outliers,
                    "discarded_clutch_laps": discarded_clutch,
                },
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
        #
        # loss='huber' (Proposal P2, 2026-04-25): mais robusto a outliers
        # residuais que passam pelo filtro de 5s. Squared loss penaliza
        # quadraticamente erros grandes — em telemetria de sim-racing,
        # picos isolados (saída de pista, contato) deslocam a otimização.
        # Huber é linear acima de delta=alpha, quadrática abaixo. alpha=0.9
        # significa: usa quantile 0.9 dos resíduos como ponto de transição.
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
            loss="huber",
            alpha=0.9,
        )
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler
        self._is_trained = True
        self._n_training_sectors = len(X_rows)
        self._n_discarded_clutch_laps = discarded_clutch
        self._n_discarded_outlier_sectors = discarded_outliers

        # p95 dos deltas positivos → ancora a normalização do score
        positive_deltas = y[y > 0]
        self._max_delta = float(np.percentile(positive_deltas, 95)) if len(positive_deltas) > 0 else 1.0

        # Feature importance ordenada (base + colunas one-hot do car_model)
        raw_importance = dict(zip(self._all_feature_names, model.feature_importances_))
        self._feature_importance = {
            k: round(float(v), 4)
            for k, v in sorted(raw_importance.items(), key=lambda x: -x[1])
        }

        top_features = list(self._feature_importance.items())[:3]
        logger.info(
            "SectorModel treinado com sucesso",
            extra={
                "track_id": self._track_id,
                "target_field": _TARGET_FIELD,
                "laps_input": len(lap_data),
                "laps_clean": len(clean_laps),
                "laps_discarded_clutch": discarded_clutch,
                "sectors_discarded_outliers": discarded_outliers,
                "sectors_trained": self._n_training_sectors,
                "max_delta_per_sector_p95": round(self._max_delta, 3),
                "top_features": str(top_features),
            },
        )
        return True

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, sector: dict, car_model: str | None = None) -> float:
        """
        Prediz o score de anomalia de performance do setor.

        Args:
            sector: mini-setor no formato schema §4.5
            car_model: identificador do carro (opcional). Necessário para
                       modelos treinados cross-car — sem ele, a one-hot fica
                       zerada e a predição cai para a média entre carros.

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
            features = np.array([self._row_for_sector(sector, car_model)], dtype=float)
            features_scaled = self._scaler.transform(features)
            predicted_delta = float(self._model.predict(features_scaled)[0])
            score = max(0.0, min(1.0, predicted_delta / self._max_delta))
            return round(score, 4)
        except Exception as exc:
            logger.debug("Falha na predição do SectorModel", extra={"error": str(exc)})
            return 0.0

    def predict_batch(
        self,
        sectors: list[dict],
        car_model: str | None = None,
    ) -> list[float]:
        """
        Prediz scores de anomalia para uma lista de mini-setores em lote.

        Mais eficiente que chamar predict() individualmente quando o modelo
        está treinado — usa uma única chamada ao numpy/sklearn.

        Args:
            sectors: lista de mini-setores no formato schema §4.5
            car_model: identificador do carro (opcional). Necessário para
                       modelos treinados cross-car (one-hot do car).

        Returns:
            Lista de scores 0.0–1.0 na mesma ordem dos setores de entrada.
        """
        if not self._is_trained or not sectors:
            return [0.0] * len(sectors)

        try:
            import numpy as np
            X = np.array(
                [self._row_for_sector(s, car_model) for s in sectors],
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
            import shutil
            import tempfile

            import joblib

            payload = {
                "track_id": self._track_id,
                "model": self._model,
                "scaler": self._scaler,
                "max_delta": self._max_delta,
                "feature_importance": self._feature_importance,
                "feature_fields": _FEATURE_FIELDS,
                "car_models": self._car_models,
                "n_training_sectors": self._n_training_sectors,
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Salva em arquivo temporário no filesystem local antes de copiar
            # para o destino final. Evita corrupção de arrays numpy quando o
            # destino é um filesystem montado (ex: mount Linux→Windows).
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                joblib.dump(payload, tmp_path, compress=3)
                shutil.copy2(tmp_path, path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

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

        Suporta dois formatos de arquivo:
        - Joblib nativo (compress=N): joblib.load() descomprime automaticamente.
        - Zlib puro + joblib: gerado por versões antigas do save() que chamavam
          zlib.compress() manualmente. Detectado pelo magic byte 0x78 no início.

        Valida que o track_id do arquivo corresponde ao desta instância.

        Args:
            path: caminho do arquivo .pkl gerado por save()

        Returns:
            True se carregado com sucesso, False caso contrário.
        """
        try:
            import io
            import zlib

            import joblib

            # Estratégia 1: joblib.load(path) — funciona para o formato moderno
            # (joblib.dump com compress=N) que usa multi-chunk de numpy.
            try:
                payload = joblib.load(path)
            except Exception:
                # Estratégia 2: formato legado — arquivo salvo manualmente como
                # zlib.compress(pickle.dumps(payload)). Magic byte 0x78.
                with open(path, "rb") as fh:
                    raw = fh.read()
                if raw[:1] != b"\x78":
                    raise
                decompressed = zlib.decompress(raw)
                payload = joblib.load(io.BytesIO(decompressed))

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
            # car_models: ausente em pkl legado (pré-2026-06-15) → lista vazia,
            # one-hot vira no-op, modelo se comporta como antes.
            self._car_models = list(payload.get("car_models", []))
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
