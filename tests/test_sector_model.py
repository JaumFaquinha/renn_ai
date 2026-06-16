"""
tests/test_sector_model.py — Testes de confiabilidade do SectorModel.

Cobre:
  - Modelo não treinado: predict() retorna 0.0 (fail-safe silencioso)
  - Treinamento com dados sintéticos determinísticos
  - Ordenação de scores: setores com delta alto pontuam mais que setores normais
  - predict_batch(): comprimento da saída igual ao da entrada
  - Coerência entre predict() e predict_batch() para o mesmo setor
  - save() / load() round-trip preserva comportamento de predição
  - feature_importance não vazia após treino
  - n_training_sectors correto após treino
  - Setores insuficientes retornam False do train()
  - Setor sem campo 'delta_per_sector' nem 'delta_vs_best' é ignorado sem exception
  - Compatibilidade retroativa: dados com 'delta_vs_best' (sem delta_per_sector)
    são aceitos via computação retroativa de diffs consecutivos
  - Métricas de evaluate_model() (MAE, R², Pearson, Precision@10%)
"""

import math
import sys
import tempfile
from pathlib import Path

import pytest

# Garante que o root do projeto está no path, independente de onde pytest é chamado
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sector_model import SectorModel, _FEATURE_FIELDS, _MIN_SECTORS_TO_TRAIN

# ---------------------------------------------------------------------------
# Fixtures e helpers
# ---------------------------------------------------------------------------

_TRACK_ID = "test_track"


def _make_sector(
    track_position: float = 0.5,
    delta_vs_best: float = 0.0,
    delta_per_sector: float = 0.0,
    speed_kmh: float = 150.0,
    brake: float = 0.0,
    throttle: float = 1.0,
) -> dict:
    """
    Cria um mini-setor sintético com valores padrão coerentes.

    O target do SectorModel é 'delta_per_sector' (perda por mini-setor).
    'delta_vs_best' é mantido para compatibilidade com fixtures históricas.
    """
    return {
        "track_position": track_position,
        "delta_vs_best": delta_vs_best,       # mantido para compatibilidade
        "delta_per_sector": delta_per_sector,  # target correto
        "throttle": throttle,
        "brake": brake,
        "steering": 0.1,
        "clutch": 0.0,
        "gear": 4,
        "rpms": 7000,
        "speed_kmh": speed_kmh,
        "speed_min": speed_kmh * 0.7,
        "gforce_x": 0.2,
        "gforce_y": 0.1,
        "gforce_z": 1.0,
        "local_ang_vel_x": 0.05,
        "local_ang_vel_y": 0.02,
        "local_ang_vel_z": 0.03,
        "wheel_slip_fl": 0.05,
        "wheel_slip_fr": 0.05,
        "wheel_slip_rl": 0.06,
        "wheel_slip_rr": 0.06,
        "tc_active": 0.0,
        "abs_active": 0.0,
        "brake_bias": 0.58,   # não é feature, ignorado pelo modelo
        "surface_grip": 0.97, # removido das features em 2026-04-14, ignorado
    }


def _make_lap(
    n_sectors: int = 100,
    delta_low: float = 0.02,
    delta_high: float | None = None,
) -> dict:
    """
    Cria uma volta sintética com n_sectors mini-setores.

    Se delta_high é fornecido, metade dos setores recebe delta_high como
    delta_per_sector e a outra metade recebe delta_low — útil para testar
    ordenação de scores. O campo 'delta_vs_best' é preenchido com a soma
    cumulativa apenas para compatibilidade de fixtures.
    """
    sectors = []
    cumulative_dvb = 0.0
    for i in range(n_sectors):
        pos = round(i / n_sectors, 4)
        if delta_high is not None and i % 2 == 0:
            dps = delta_high   # delta_per_sector alto → setor "ruim"
            brake = 0.9
            throttle = 0.1
        else:
            dps = delta_low    # delta_per_sector baixo → setor "bom"
            brake = 0.0
            throttle = 1.0
        cumulative_dvb += dps
        sectors.append(
            _make_sector(
                track_position=pos,
                delta_vs_best=cumulative_dvb,   # cumulativo (compatibilidade)
                delta_per_sector=dps,            # target real do modelo
                brake=brake,
                throttle=throttle,
            )
        )
    return {"lap_number": 1, "lap_time_ms": 82000, "mini_sectors": sectors}


def _make_sufficient_laps(
    n_laps: int = 4,
    sectors_per_lap: int = 100,
    delta_low: float = 0.02,
    delta_high: float = 0.5,
) -> list[dict]:
    """
    Gera voltas sintéticas suficientes para satisfazer _MIN_SECTORS_TO_TRAIN.

    Total de setores = n_laps * sectors_per_lap >= 200 (padrão).
    """
    return [
        _make_lap(n_sectors=sectors_per_lap, delta_low=delta_low, delta_high=delta_high)
        for _ in range(n_laps)
    ]


# ---------------------------------------------------------------------------
# 1 — Modelo não treinado: fail-safe silencioso
# ---------------------------------------------------------------------------


class TestUntrained:
    def test_is_trained_false_on_init(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        assert model.is_trained is False

    def test_predict_returns_zero_when_not_trained(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        sector = _make_sector(delta_vs_best=1.0)
        assert model.predict(sector) == 0.0

    def test_predict_batch_returns_zeros_when_not_trained(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        sectors = [_make_sector() for _ in range(5)]
        scores = model.predict_batch(sectors)
        assert scores == [0.0] * 5

    def test_predict_batch_empty_input_when_not_trained(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        assert model.predict_batch([]) == []

    def test_save_returns_false_when_not_trained(self, tmp_path: Path) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        result = model.save(str(tmp_path / "model.pkl"))
        assert result is False

    def test_feature_importance_empty_when_not_trained(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        assert model.feature_importance == {}

    def test_n_training_sectors_zero_when_not_trained(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        assert model.n_training_sectors == 0


# ---------------------------------------------------------------------------
# 2 — Treinamento com dados insuficientes
# ---------------------------------------------------------------------------


class TestTrainInsufficient:
    def test_train_fails_with_zero_laps(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train([])
        assert result is False
        assert model.is_trained is False

    def test_train_fails_below_min_sectors(self) -> None:
        # 1 lap com 50 setores < _MIN_SECTORS_TO_TRAIN (200)
        laps = [_make_lap(n_sectors=50)]
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train(laps)
        assert result is False
        assert model.is_trained is False

    def test_train_ignores_sectors_without_target(self) -> None:
        """
        Setores sem 'delta_per_sector' nem 'delta_vs_best' são descartados.

        O modelo precisa de ao menos um dos dois para calcular o target.
        Sem nenhum dos dois, todos os setores são ignorados → insuficientes.
        """
        sectors_no_target = [
            {
                k: v for k, v in _make_sector().items()
                if k not in ("delta_per_sector", "delta_vs_best")
            }
            for _ in range(300)
        ]
        laps = [{"lap_number": 1, "lap_time_ms": 82000, "mini_sectors": sectors_no_target}]
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train(laps)
        assert result is False  # todos descartados → insuficientes

    def test_train_handles_laps_without_mini_sectors_key(self) -> None:
        """Voltas sem chave 'mini_sectors' não devem lançar exceção."""
        laps = [{"lap_number": 1, "lap_time_ms": 82000}]  # sem 'mini_sectors'
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train(laps)
        assert result is False

    def test_train_retroactive_compat_with_delta_vs_best_only(self) -> None:
        """
        Dados históricos com apenas 'delta_vs_best' (sem 'delta_per_sector')
        devem ser aceitos pelo treino via computação retroativa de diffs.

        Cria 4 voltas onde delta_vs_best cresce monotonicamente — os diffs
        consecutivos serão todos positivos e dentro do threshold de 5s.
        """
        sectors_legacy = []
        dvb = 0.0
        for i in range(110):
            s = {k: v for k, v in _make_sector(track_position=i / 110).items()
                 if k != "delta_per_sector"}  # remove o campo novo
            dvb += 0.02  # incremento constante → diff = 0.02s por setor
            s["delta_vs_best"] = dvb
            sectors_legacy.append(s)

        laps = [
            {"lap_number": i, "lap_time_ms": 82000, "mini_sectors": sectors_legacy}
            for i in range(4)
        ]
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train(laps)
        # Deve treinar com sucesso usando diffs de delta_vs_best
        # (4 voltas × 110 setores − 4 primeiros = 436 setores, acima do mínimo 200)
        assert result is True
        assert model.is_trained is True

    def test_train_handles_delta_per_sector_none_from_supabase(self) -> None:
        """
        Dados vindos do Supabase carregam todas as colunas selecionadas, mesmo
        quando o valor é NULL no banco (caso de mini-setores gravados antes da
        migração que introduziu `delta_per_sector`).

        O treino deve cair no fallback retroativo via `delta_vs_best` em vez de
        levantar `TypeError: float() argument must be ... not 'NoneType'`.
        """
        sectors_mixed = []
        dvb = 0.0
        for i in range(110):
            s = _make_sector(track_position=i / 110)
            # Simula coluna NULL no banco: chave existe, valor é None
            s["delta_per_sector"] = None
            dvb += 0.02
            s["delta_vs_best"] = dvb
            sectors_mixed.append(s)

        laps = [
            {"lap_number": i, "lap_time_ms": 82000, "mini_sectors": sectors_mixed}
            for i in range(4)
        ]
        model = SectorModel(track_id=_TRACK_ID)
        # Antes da correção, isto levantava TypeError no float(None)
        result = model.train(laps)
        assert result is True
        assert model.is_trained is True

    def test_train_skips_sector_when_both_targets_are_none(self) -> None:
        """
        Setor com `delta_per_sector=None` E `delta_vs_best=None` deve ser
        descartado silenciosamente (sem TypeError), igual a quando ambas as
        chaves estão ausentes.
        """
        sectors_all_null = []
        for i in range(300):
            s = _make_sector(track_position=i / 300)
            s["delta_per_sector"] = None
            s["delta_vs_best"] = None
            sectors_all_null.append(s)

        laps = [{"lap_number": 1, "lap_time_ms": 82000, "mini_sectors": sectors_all_null}]
        model = SectorModel(track_id=_TRACK_ID)
        result = model.train(laps)
        # Todos descartados → insuficientes (sem exceção)
        assert result is False


# ---------------------------------------------------------------------------
# 3 — Treinamento bem-sucedido
# ---------------------------------------------------------------------------


class TestTrainSuccess:
    @pytest.fixture(scope="class")
    def trained_model(self) -> SectorModel:
        """Modelo treinado compartilhado por todos os testes desta classe."""
        laps = _make_sufficient_laps(n_laps=4, sectors_per_lap=100)
        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok, "Treino deve ter sucesso com dados suficientes"
        return model

    def test_is_trained_true_after_train(self, trained_model: SectorModel) -> None:
        assert trained_model.is_trained is True

    def test_n_training_sectors_correct(self, trained_model: SectorModel) -> None:
        # 4 laps × 100 setores = 400 setores
        assert trained_model.n_training_sectors == 400

    def test_feature_importance_not_empty(self, trained_model: SectorModel) -> None:
        fi = trained_model.feature_importance
        assert len(fi) == len(_FEATURE_FIELDS)

    def test_feature_importance_sums_to_one(self, trained_model: SectorModel) -> None:
        total = sum(trained_model.feature_importance.values())
        assert math.isclose(total, 1.0, abs_tol=0.01)

    def test_feature_importance_all_non_negative(self, trained_model: SectorModel) -> None:
        for feat, imp in trained_model.feature_importance.items():
            assert imp >= 0.0, f"Feature '{feat}' tem importância negativa: {imp}"

    def test_predict_returns_float_in_range(self, trained_model: SectorModel) -> None:
        sector = _make_sector(delta_vs_best=0.5)
        score = trained_model.predict(sector)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_batch_returns_correct_length(self, trained_model: SectorModel) -> None:
        sectors = [_make_sector() for _ in range(10)]
        scores = trained_model.predict_batch(sectors)
        assert len(scores) == 10

    def test_predict_batch_empty_input_returns_empty(self, trained_model: SectorModel) -> None:
        assert trained_model.predict_batch([]) == []

    def test_predict_batch_all_scores_in_range(self, trained_model: SectorModel) -> None:
        sectors = [_make_sector(delta_vs_best=d) for d in [0.0, 0.1, 0.5, 1.0, 2.0]]
        scores = trained_model.predict_batch(sectors)
        for score in scores:
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 4 — Monotonicidade: features de perda devem aumentar o score
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """
    Verifica que o modelo aprendeu relações monotônicas corretas.

    Motivação: o relatório de validação v3 (2026-04-14) identificou que o
    modelo treinado com delta_vs_best (cumulativo) produzia INVERSÃO de
    monotonicidade para tc_active e wheel_slip — score caía quando essas
    features aumentavam. Com delta_per_sector como target, a relação causal
    direta deve ser preservada.
    """

    @pytest.fixture(scope="class")
    def trained_model(self) -> SectorModel:
        """
        Treina com dados que têm sinal claro de causalidade:
        - Setores com delta_per_sector alto têm tc_active=1, wheel_slip alto, abs_active=1
        - Setores com delta_per_sector baixo têm todos esses campos em zero
        """
        sectors_good = [
            _make_sector(
                track_position=round(i / 200, 4),
                delta_per_sector=0.01,
                throttle=1.0, brake=0.0,
            )
            for i in range(100)
        ]
        sectors_bad_tc = [
            _make_sector(
                track_position=round((100 + i) / 200, 4),
                delta_per_sector=0.5,
                throttle=0.3, brake=0.0,
            )
            | {"tc_active": 1.0, "wheel_slip_rl": 0.4, "wheel_slip_rr": 0.4}
            for i in range(100)
        ]
        sectors_bad_abs = [
            _make_sector(
                track_position=round(i / 200, 4),
                delta_per_sector=0.4,
                throttle=0.0, brake=1.0,
            )
            | {"abs_active": 1.0, "wheel_slip_fl": 0.6, "wheel_slip_fr": 0.6}
            for i in range(100)
        ]

        all_sectors = sectors_good + sectors_bad_tc + sectors_bad_abs
        lap = {"lap_number": 1, "lap_time_ms": 82000, "mini_sectors": all_sectors}
        laps = [lap] * 4  # 4 repetições → 1200 setores, acima do mínimo

        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok, "Treino deve ter sucesso"
        return model

    def test_tc_active_increases_score(self, trained_model: SectorModel) -> None:
        """tc_active=1 deve resultar em score maior que tc_active=0, tudo mais igual."""
        base = _make_sector(track_position=0.25, delta_per_sector=0.0, throttle=0.8)
        s_no_tc = {**base, "tc_active": 0.0}
        s_tc = {**base, "tc_active": 1.0, "wheel_slip_rl": 0.4, "wheel_slip_rr": 0.4}
        assert trained_model.predict(s_tc) >= trained_model.predict(s_no_tc), (
            "Score com TC ativo deve ser >= score sem TC"
        )

    def test_abs_active_increases_score(self, trained_model: SectorModel) -> None:
        """abs_active=1 deve resultar em score maior que abs_active=0."""
        base = _make_sector(track_position=0.1, delta_per_sector=0.0, brake=0.8)
        s_no_abs = {**base, "abs_active": 0.0}
        s_abs = {**base, "abs_active": 1.0, "wheel_slip_fl": 0.6}
        assert trained_model.predict(s_abs) >= trained_model.predict(s_no_abs), (
            "Score com ABS ativo deve ser >= score sem ABS"
        )

    def test_neutral_sector_scores_low(self, trained_model: SectorModel) -> None:
        """Setor sem nenhum marcador de perda deve ter score baixo."""
        neutral = _make_sector(
            track_position=0.5,
            delta_per_sector=0.01,
            throttle=1.0, brake=0.0,
        )
        score = trained_model.predict(neutral)
        # Score abaixo de 0.5 (limiar conservador — o importante é não ser alto)
        assert score < 0.5, f"Setor neutro não deve ter score alto: {score:.4f}"


# ---------------------------------------------------------------------------
# 6 — Ordenação de scores: setores com alta perda pontuam mais
# ---------------------------------------------------------------------------


class TestScoreOrdering:
    """
    Verifica que o modelo aprendeu a diferenciar setores ruins de setores bons.

    Estratégia: treina com dados onde setor_ruim (alta frenagem + delta alto)
    é claramente diferente de setor_bom (throttle cheio + delta baixo).
    A média dos scores dos setores ruins deve ser maior que a dos bons.
    """

    @pytest.fixture(scope="class")
    def trained_model(self) -> SectorModel:
        laps = _make_sufficient_laps(
            n_laps=4,
            sectors_per_lap=100,
            delta_low=0.01,
            delta_high=0.8,
        )
        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok
        return model

    def test_high_delta_sector_scores_higher_on_average(
        self, trained_model: SectorModel
    ) -> None:
        """
        Um conjunto de setores com padrão de alta perda deve ter score médio
        maior que setores com padrão de baixa perda.
        """
        high_loss_sectors = [
            _make_sector(
                track_position=round(i / 10, 2),
                delta_per_sector=0.8,
                brake=0.9,
                throttle=0.1,
            )
            for i in range(10)
        ]
        low_loss_sectors = [
            _make_sector(
                track_position=round(i / 10, 2),
                delta_per_sector=0.01,
                brake=0.0,
                throttle=1.0,
            )
            for i in range(10)
        ]

        high_scores = trained_model.predict_batch(high_loss_sectors)
        low_scores = trained_model.predict_batch(low_loss_sectors)

        avg_high = sum(high_scores) / len(high_scores)
        avg_low = sum(low_scores) / len(low_scores)

        assert avg_high > avg_low, (
            f"Score médio de setores ruins ({avg_high:.4f}) deve ser maior "
            f"que setores bons ({avg_low:.4f})"
        )


# ---------------------------------------------------------------------------
# 7 — Coerência predict() vs predict_batch()
# ---------------------------------------------------------------------------


class TestPredictConsistency:
    @pytest.fixture(scope="class")
    def trained_model(self) -> SectorModel:
        laps = _make_sufficient_laps()
        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok
        return model

    def test_predict_and_predict_batch_agree(self, trained_model: SectorModel) -> None:
        """predict() e predict_batch() devem retornar o mesmo score para o mesmo setor."""
        sectors = [_make_sector(track_position=round(i * 0.1, 1)) for i in range(5)]
        batch_scores = trained_model.predict_batch(sectors)
        single_scores = [trained_model.predict(s) for s in sectors]

        for i, (batch, single) in enumerate(zip(batch_scores, single_scores)):
            assert math.isclose(batch, single, abs_tol=1e-6), (
                f"Setor {i}: predict_batch={batch}, predict={single}"
            )


# ---------------------------------------------------------------------------
# 8 — save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Modelo salvo e recarregado deve produzir os mesmos scores."""
        laps = _make_sufficient_laps()
        model_original = SectorModel(track_id=_TRACK_ID)
        ok = model_original.train(laps)
        assert ok

        model_path = str(tmp_path / "test_model.pkl")
        saved = model_original.save(model_path)
        assert saved is True
        assert Path(model_path).exists()

        model_loaded = SectorModel(track_id=_TRACK_ID)
        loaded = model_loaded.load(model_path)
        assert loaded is True
        assert model_loaded.is_trained is True

        # Scores devem ser idênticos após round-trip
        test_sectors = [
            _make_sector(delta_vs_best=0.3, brake=0.7),
            _make_sector(delta_vs_best=0.05, throttle=0.9),
        ]
        original_scores = model_original.predict_batch(test_sectors)
        loaded_scores = model_loaded.predict_batch(test_sectors)

        for orig, load in zip(original_scores, loaded_scores):
            assert math.isclose(orig, load, abs_tol=1e-6)

    def test_load_preserves_n_training_sectors(self, tmp_path: Path) -> None:
        laps = _make_sufficient_laps(n_laps=4, sectors_per_lap=100)
        model = SectorModel(track_id=_TRACK_ID)
        model.train(laps)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        model2 = SectorModel(track_id=_TRACK_ID)
        model2.load(path)
        assert model2.n_training_sectors == model.n_training_sectors

    def test_load_preserves_feature_importance(self, tmp_path: Path) -> None:
        laps = _make_sufficient_laps()
        model = SectorModel(track_id=_TRACK_ID)
        model.train(laps)

        path = str(tmp_path / "model.pkl")
        model.save(path)

        model2 = SectorModel(track_id=_TRACK_ID)
        model2.load(path)
        assert model2.feature_importance == model.feature_importance

    def test_load_nonexistent_file_returns_false(self) -> None:
        model = SectorModel(track_id=_TRACK_ID)
        result = model.load("/nonexistent/path/model.pkl")
        assert result is False
        assert model.is_trained is False

    def test_load_wrong_track_id_still_loads(self, tmp_path: Path) -> None:
        """
        Modelo de pista diferente emite warning mas ainda carrega
        (não bloqueia uso, a decisão de rejeitar é do chamador).
        """
        laps = _make_sufficient_laps()
        model_spa = SectorModel(track_id="spa")
        model_spa.train(laps)

        path = str(tmp_path / "spa.pkl")
        model_spa.save(path)

        model_monza = SectorModel(track_id="monza")
        result = model_monza.load(path)
        assert result is True  # carregou mesmo com track_id diferente


# ---------------------------------------------------------------------------
# 9 — Métricas de evaluate_model() (via scripts/train_model.py)
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """
    Valida as métricas de confiabilidade retornadas por evaluate_model().

    Em dados sintéticos com sinal claro (delta_high muito maior que delta_low),
    o modelo deve atingir correlação e precision mínimas aceitáveis.
    """

    @pytest.fixture(scope="class")
    def trained_model_and_laps(self):
        """Retorna (model, laps) para uso nos testes de métricas."""
        laps = _make_sufficient_laps(
            n_laps=6,
            sectors_per_lap=100,
            delta_low=0.01,
            delta_high=1.0,
        )
        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok
        return model, laps

    def test_evaluate_model_returns_all_metrics(
        self, trained_model_and_laps
    ) -> None:
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)

        assert "n_sectors" in metrics
        assert "mae_s" in metrics
        assert "r2" in metrics
        assert "pearson_correlation" in metrics
        assert "precision_at_10pct" in metrics

    def test_n_sectors_in_metrics_correct(
        self, trained_model_and_laps
    ) -> None:
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)
        expected = sum(len(lap["mini_sectors"]) for lap in laps)
        assert metrics["n_sectors"] == expected

    def test_pearson_correlation_above_threshold(
        self, trained_model_and_laps
    ) -> None:
        """
        Com sinal claro (delta_high=1.0 vs delta_low=0.01), correlação >= 0.5
        deve ser alcançada — limiar mínimo do scripts/train_model.py.
        """
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)
        corr = metrics.get("pearson_correlation", 0.0)
        assert corr >= 0.5, (
            f"Correlação de Pearson ({corr:.3f}) abaixo do limiar mínimo de 0.5"
        )

    def test_precision_at_10pct_above_threshold(
        self, trained_model_and_laps
    ) -> None:
        """
        Precision@10% >= 0.5 em dados com sinal claro indica que o modelo
        detecta pelo menos metade dos setores mais problemáticos.
        """
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)
        precision = metrics.get("precision_at_10pct", 0.0)
        assert precision >= 0.5, (
            f"Precision@10% ({precision:.3f}) abaixo do limiar mínimo de 0.5"
        )

    def test_mae_is_non_negative(self, trained_model_and_laps) -> None:
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)
        assert metrics.get("mae_s", -1.0) >= 0.0

    def test_r2_below_one(self, trained_model_and_laps) -> None:
        """R² in-sample pode ser alto, mas nunca deve ser > 1."""
        from scripts.train_model import evaluate_model

        model, laps = trained_model_and_laps
        metrics = evaluate_model(model, laps)
        assert metrics.get("r2", 2.0) <= 1.0


# ---------------------------------------------------------------------------
# 10 — Robustez: setor com campos ausentes não lança exceção
# ---------------------------------------------------------------------------


class TestRobustness:
    @pytest.fixture(scope="class")
    def trained_model(self) -> SectorModel:
        laps = _make_sufficient_laps()
        model = SectorModel(track_id=_TRACK_ID)
        ok = model.train(laps)
        assert ok
        return model

    def test_predict_empty_sector_returns_zero_to_one(
        self, trained_model: SectorModel
    ) -> None:
        """Setor completamente vazio usa 0.0 para todos os campos — não deve lançar."""
        score = trained_model.predict({})
        assert 0.0 <= score <= 1.0

    def test_predict_partial_sector_does_not_raise(
        self, trained_model: SectorModel
    ) -> None:
        """Setor com apenas alguns campos preenchidos não deve lançar exceção."""
        partial = {"track_position": 0.5, "brake": 0.8}
        score = trained_model.predict(partial)
        assert 0.0 <= score <= 1.0

    def test_predict_batch_with_mixed_sectors_does_not_raise(
        self, trained_model: SectorModel
    ) -> None:
        """Lista mista (setores completos e vazios) não deve lançar exceção."""
        sectors = [
            _make_sector(),
            {},
            {"track_position": 0.3},
            _make_sector(brake=0.9),
        ]
        scores = trained_model.predict_batch(sectors)
        assert len(scores) == 4
        for score in scores:
            assert 0.0 <= score <= 1.0
