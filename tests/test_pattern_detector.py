"""
Testes da FASE 4 — PatternDetector.

Verifica cada um dos 5 padrões de causa de perda de tempo individualmente.
"""

import pytest

from src.analysis.pattern_detector import PatternDetector, PatternMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_sector(**overrides) -> dict:
    """Mini-setor base com valores neutros (sem padrão detectável)."""
    sector = {
        "track_position": 0.5,
        "delta_vs_best": 0.3,  # Perda suficiente para análise
        "throttle": 1.0,
        "brake": 0.0,
        "steering": 0.05,
        "gear": 5,
        "rpms": 7000,
        "clutch": 0.0,
        "speed_kmh": 220.0,
        "speed_min": 210.0,
        "gforce_x": 0.2,
        "gforce_y": -0.5,
        "gforce_z": 1.0,
        "local_ang_vel_x": 0.01,
        "local_ang_vel_y": 0.01,
        "local_ang_vel_z": 0.02,
        "wheel_slip_fl": 0.03,
        "wheel_slip_fr": 0.03,
        "wheel_slip_rl": 0.04,
        "wheel_slip_rr": 0.04,
        "tc_active": 0.0,
        "abs_active": 0.0,
        "drs_active": 0,
        "drs_available": 0,
        "brake_bias": 0.58,
        "surface_grip": 0.97,
        "air_temp": 24.0,
        "road_temp": 31.0,
    }
    sector.update(overrides)
    return sector


def late_braking_sector() -> dict:
    """Setor com padrão de frenagem tardia com bloqueio."""
    return base_sector(
        brake=0.95,
        abs_active=0.35,
        speed_min=87.0,
        throttle=0.0,
    )


def early_throttle_sector() -> dict:
    """Setor com padrão de aceleração precoce/agressiva."""
    return base_sector(
        throttle=0.6,
        tc_active=0.25,
        wheel_slip_rl=0.20,
        wheel_slip_rr=0.22,
    )


def fast_corner_entry_sector() -> dict:
    """Setor com entrada de curva rápida demais."""
    return base_sector(
        gforce_x=3.2,
        steering=0.75,
        speed_kmh=185.0,
        brake=0.3,
    )


def suboptimal_shift_late_sector(max_rpm: int = 8000) -> dict:
    """Setor com troca de marcha tardia."""
    return base_sector(
        rpms=int(max_rpm * 0.98),  # Bem acima do ponto ótimo
        gear=3,
    )


def compromised_exit_sector() -> dict:
    """Setor com saída de curva comprometida (reta com throttle parcial)."""
    return base_sector(
        throttle=0.65,
        steering=0.05,  # Reta
        speed_kmh=200.0,
        tc_active=0.0,
    )


# ---------------------------------------------------------------------------
# Testes de detecção individual
# ---------------------------------------------------------------------------

class TestPatternDetector:
    def setup_method(self):
        self.detector = PatternDetector(max_rpm=8000)

    # --- Padrão 1: Frenagem tardia ---

    def test_detects_late_braking(self):
        sector = late_braking_sector()
        matches = self.detector.detect(sector)
        causes = [m.cause for m in matches]
        assert any("frenagem" in c.lower() for c in causes), f"Frenagem tardia não detectada. Causas: {causes}"

    def test_late_braking_confidence_is_valid(self):
        sector = late_braking_sector()
        matches = self.detector.detect(sector)
        braking_match = next(m for m in matches if "frenagem" in m.cause.lower())
        assert 0.0 < braking_match.confidence <= 1.0

    def test_no_late_braking_without_abs(self):
        sector = base_sector(brake=0.95, abs_active=0.0)
        matches = self.detector.detect(sector)
        braking_matches = [m for m in matches if "frenagem" in m.cause.lower()]
        assert len(braking_matches) == 0

    def test_no_late_braking_with_low_brake(self):
        sector = base_sector(brake=0.3, abs_active=0.5)
        matches = self.detector.detect(sector)
        braking_matches = [m for m in matches if "frenagem" in m.cause.lower()]
        assert len(braking_matches) == 0

    # --- Padrão 2: Aceleração precoce ---

    def test_detects_early_throttle(self):
        sector = early_throttle_sector()
        matches = self.detector.detect(sector)
        causes = [m.cause for m in matches]
        assert any("acelera" in c.lower() for c in causes), f"Aceleração precoce não detectada. Causas: {causes}"

    def test_early_throttle_confidence_is_valid(self):
        sector = early_throttle_sector()
        matches = self.detector.detect(sector)
        tc_match = next(m for m in matches if "acelera" in m.cause.lower())
        assert 0.0 < tc_match.confidence <= 1.0

    def test_no_early_throttle_without_tc(self):
        sector = base_sector(tc_active=0.0, wheel_slip_rl=0.25)
        matches = self.detector.detect(sector)
        tc_matches = [m for m in matches if "acelera" in m.cause.lower()]
        assert len(tc_matches) == 0

    # --- Padrão 3: Entrada de curva rápida ---

    def test_detects_fast_corner_entry(self):
        sector = fast_corner_entry_sector()
        matches = self.detector.detect(sector)
        causes = [m.cause for m in matches]
        assert any("curva" in c.lower() for c in causes), f"Entrada rápida não detectada. Causas: {causes}"

    def test_no_fast_corner_entry_with_low_gforce(self):
        sector = base_sector(gforce_x=1.5, steering=0.8)
        matches = self.detector.detect(sector)
        corner_matches = [m for m in matches if "curva" in m.cause.lower()]
        assert len(corner_matches) == 0

    # --- Padrão 4: Troca subótima ---

    def test_detects_suboptimal_shift_late(self):
        sector = suboptimal_shift_late_sector(max_rpm=8000)
        matches = self.detector.detect(sector)
        causes = [m.cause for m in matches]
        assert any("troca" in c.lower() for c in causes), f"Troca subótima não detectada. Causas: {causes}"

    def test_no_suboptimal_shift_at_optimal_rpm(self):
        # RPM no meio da faixa de potência — sem detecção esperada
        sector = base_sector(rpms=7000, gear=4)  # 7000/8000 = 87.5% do max
        matches = self.detector.detect(sector)
        shift_matches = [m for m in matches if "troca" in m.cause.lower()]
        assert len(shift_matches) == 0

    # --- Padrão 5: Saída comprometida ---

    def test_detects_compromised_exit(self):
        sector = compromised_exit_sector()
        matches = self.detector.detect(sector)
        causes = [m.cause for m in matches]
        assert any("saída" in c.lower() for c in causes), f"Saída comprometida não detectada. Causas: {causes}"

    def test_no_compromised_exit_at_full_throttle(self):
        sector = base_sector(throttle=1.0, steering=0.05, speed_kmh=250.0)
        matches = self.detector.detect(sector)
        exit_matches = [m for m in matches if "saída" in m.cause.lower()]
        assert len(exit_matches) == 0

    def test_no_compromised_exit_with_high_steering(self):
        # Curvando — não é uma reta
        sector = base_sector(throttle=0.5, steering=0.5, speed_kmh=200.0)
        matches = self.detector.detect(sector)
        exit_matches = [m for m in matches if "saída" in m.cause.lower()]
        assert len(exit_matches) == 0

    # --- Testes gerais ---

    def test_no_detection_below_delta_threshold(self):
        """Não deve detectar padrões se a perda for insignificante."""
        sector = late_braking_sector()
        sector["delta_vs_best"] = 0.01  # Abaixo do threshold
        matches = self.detector.detect(sector)
        assert len(matches) == 0

    def test_multiple_patterns_ordered_by_confidence(self):
        """Com múltiplos padrões, resultado deve ser ordenado por confiança."""
        sector = base_sector(
            brake=0.95, abs_active=0.5,  # Frenagem tardia
            gforce_x=3.5, steering=0.8,  # Entrada rápida
        )
        matches = self.detector.detect(sector)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].confidence >= matches[i + 1].confidence

    def test_pattern_match_has_evidence(self):
        """Cada PatternMatch deve ter evidence não vazio."""
        sector = late_braking_sector()
        matches = self.detector.detect(sector)
        for match in matches:
            assert isinstance(match.evidence, dict)
            assert len(match.evidence) > 0
