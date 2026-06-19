"""
Testes do construtor de mensagem falada (src/output/voice_message.py).

Cobre os dois tons: mensagem positiva (volta boa / melhor da sessão / recorde)
e mensagem de perda (contexto + pior zona + até 2 causas).
"""

from src.analysis.pattern_detector import PatternMatch
from src.analysis.report_builder import LapReport, SectorReport
from src.output.voice_message import build_voice_message


def _sector(
    *,
    position: float = 0.50,
    delta: float = 0.30,
    speed: float = 90.0,
    causes: list[PatternMatch] | None = None,
    corner_name: str | None = None,
    corner_type: str | None = None,
    sector_name: str | None = None,
    model_score: float | None = None,
) -> SectorReport:
    return SectorReport(
        track_position=position,
        delta_per_sector_s=delta,
        speed_min_kmh=speed,
        causes=causes or [],
        corner_name=corner_name,
        corner_type=corner_type,
        sector_name=sector_name,
        model_score=model_score,
    )


def _lap(top_sectors: list[SectorReport], *, total_lost: float = 0.0, lap_number: int = 5) -> LapReport:
    return LapReport(
        lap_number=lap_number,
        lap_time_ms=82_340,
        track_id="monza",
        top_sectors=top_sectors,
        total_time_lost_s=total_lost,
    )


# ---------------------------------------------------------------------------
# Mensagens positivas
# ---------------------------------------------------------------------------


def test_alltime_best_returns_positive():
    # Mesmo com setores de perda, recorde tem prioridade na mensagem positiva.
    lap = _lap([_sector(delta=0.4)], total_lost=0.4)
    msg = build_voice_message(lap, is_session_best=True, is_alltime_best=True)
    assert msg is not None
    assert any(w in msg.lower() for w in ("recorde", "todos os tempos"))


def test_session_best_returns_positive():
    lap = _lap([_sector(delta=0.2)], total_lost=0.2)
    msg = build_voice_message(lap, is_session_best=True, is_alltime_best=False)
    assert "sessão" in msg.lower()


def test_low_model_score_is_good_lap():
    # Modelo prevê anomalia baixa → mensagem positiva (base elogiosa).
    lap = _lap([_sector(delta=0.25, model_score=0.05, corner_name="Lesmo")], total_lost=0.25)
    msg = build_voice_message(lap)
    assert msg is not None
    assert any(w in msg.lower() for w in ("limpa", "sólida"))


def test_good_lap_with_relevant_gain_gets_hybrid_clause():
    # Volta boa (modelo baixo) mas com ganho relevante → híbrida.
    causes = [PatternMatch(cause="Frenagem tardia", confidence=0.8)]
    lap = _lap([_sector(delta=0.15, model_score=0.05, corner_name="Parabólica", causes=causes)], total_lost=0.15)
    msg = build_voice_message(lap)
    assert any(w in msg.lower() for w in ("limpa", "sólida"))  # base positivo
    assert "ganhar" in msg.lower()                              # cláusula híbrida
    assert "0,15" in msg
    assert "Parabólica" in msg
    assert "frenagem tardia" in msg.lower()


def test_session_best_with_gain_is_hybrid():
    lap = _lap([_sector(delta=0.20, corner_name="Ascari")], total_lost=0.20)
    msg = build_voice_message(lap, is_session_best=True)
    assert "sessão" in msg.lower()
    assert "ganhar" in msg.lower()
    assert "Ascari" in msg


def test_positive_without_relevant_gain_stays_pure():
    # Ganho abaixo do limiar (0.08) → sem cláusula híbrida.
    lap = _lap([_sector(delta=0.03, corner_name="Parabólica")], total_lost=0.03)
    msg = build_voice_message(lap, is_session_best=True)
    assert "ganhar" not in msg.lower()


def test_tiny_total_loss_is_good_lap_without_model():
    lap = _lap([_sector(delta=0.05, model_score=None)], total_lost=0.05)
    msg = build_voice_message(lap)
    assert msg is not None
    assert any(w in msg.lower() for w in ("limpa", "sólida"))


def test_no_sectors_returns_positive():
    msg = build_voice_message(_lap([], total_lost=0.0))
    assert msg is not None


# ---------------------------------------------------------------------------
# Mensagens de perda
# ---------------------------------------------------------------------------


def test_loss_message_has_context_zone_and_cause():
    causes = [PatternMatch(cause="Frenagem tardia com bloqueio de rodas", confidence=0.9)]
    lap = _lap(
        [_sector(delta=0.30, model_score=0.8, corner_name="Parabólica", corner_type="curva rápida", causes=causes)],
        total_lost=0.30,
    )
    msg = build_voice_message(lap)
    assert "Parabólica" in msg
    assert "curva rápida" in msg
    assert "0,30" in msg              # pt-BR: vírgula decimal
    assert "Frenagem tardia" in msg
    assert "no total" in msg          # contexto de perda total


def test_loss_message_includes_two_causes_from_top_sector():
    causes = [
        PatternMatch(cause="Frenagem tardia com bloqueio de rodas", confidence=0.9),
        PatternMatch(cause="Trail-braking excessivo", confidence=0.6),
    ]
    lap = _lap([_sector(delta=0.30, model_score=0.8, corner_name="Lesmo", causes=causes)], total_lost=0.30)
    msg = build_voice_message(lap)
    assert "Frenagem tardia" in msg
    assert "Trail-braking" in msg


def test_loss_message_pulls_cause_from_second_sector():
    top = _sector(
        delta=0.30, model_score=0.8, corner_name="Ascari",
        causes=[PatternMatch(cause="Velocidade de entrada elevada", confidence=0.8)],
    )
    second = _sector(
        position=0.7, delta=0.15, model_score=0.7, corner_name="Variante",
        causes=[PatternMatch(cause="Aceleração precoce", confidence=0.7)],
    )
    lap = _lap([top, second], total_lost=0.45)
    msg = build_voice_message(lap)
    assert "Velocidade de entrada elevada" in msg
    assert "Variante" in msg
    assert "aceleração precoce" in msg.lower()


def test_loss_message_uses_sector_when_no_corner():
    # Sem nome de curva → fala o setor oficial, não o valor cru da spline.
    causes = [PatternMatch(cause="Hesitação no acelerador", confidence=0.7)]
    lap = _lap(
        [_sector(position=0.55, delta=0.20, model_score=0.7, corner_name=None, sector_name="Setor 2", causes=causes)],
        total_lost=0.20,
    )
    msg = build_voice_message(lap)
    assert "Setor 2" in msg
    assert "trecho" not in msg
    assert "spline" not in msg.lower()


def test_loss_message_falls_back_to_spline_without_sector():
    # Sem curva e sem setor (track map ausente) → mantém o trecho da spline.
    lap = _lap([_sector(position=0.55, delta=0.20, model_score=0.7)], total_lost=0.20)
    msg = build_voice_message(lap)
    assert "trecho" in msg


def test_loss_message_fits_char_budget():
    causes = [
        PatternMatch(cause="Frenagem tardia com bloqueio de rodas", confidence=0.9),
        PatternMatch(cause="Aceleração precoce ou agressiva — TC interveio", confidence=0.6),
    ]
    lap = _lap(
        [_sector(delta=0.30, model_score=0.8, corner_name="Parabólica", corner_type="curva rápida", causes=causes)],
        total_lost=0.82,
    )
    msg = build_voice_message(lap)
    assert len(msg) <= 220
