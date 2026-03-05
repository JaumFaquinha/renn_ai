"""
Configurações globais do Engenheiro de Corrida IA.

Todas as constantes de configuração estão aqui. Valores sensíveis
(API keys, etc.) devem ser carregados via .env com python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# === Diretórios ===
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
LAPS_DIR: Path = Path(os.getenv("LAPS_OUTPUT_DIR", str(DATA_DIR / "laps")))
MODELS_DIR: Path = DATA_DIR / "models"
TRACK_MAPS_DIR: Path = ROOT_DIR / "config" / "track_maps"

# === Sampling ===
SAMPLING_RATE_HZ: int = 20          # Frequência de leitura da Shared Memory
MINI_SECTOR_SIZE: float = 0.01      # Tamanho de mini-setor em spline normalizada (~1% da pista)

# === Campos da Shared Memory ===
SPLINE_POSITION_FIELD: str = "normalizedCarPosition"
PERFORMANCE_METER_FIELD: str = "performanceMeter"

# === Validação de Volta ===
# Descarta volta se qualquer componente de dano ultrapassar este valor (0.0–1.0)
CAR_DAMAGE_THRESHOLD: float = float(os.getenv("CAR_DAMAGE_THRESHOLD", "0.1"))

# Número mínimo de mini-setores para uma volta ser considerada válida
MIN_SECTORS_PER_LAP: int = 80       # ~80% da pista mínimo

# === Análise ===
# Número de top setores com maior perda a destacar no relatório
TOP_SECTORS_TO_REPORT: int = 5

# Margem aceitável entre performanceMeter do AC e delta calculado (ms)
DELTA_TOLERANCE_MS: int = 50

# === Detecção de Padrões ===
# Thresholds para os padrões definidos em CLAUDE.md §5

# Frenagem tardia com bloqueio
ABS_ACTIVE_THRESHOLD: float = 0.1          # abs > X → ABS ativado
LATE_BRAKE_SPEED_LOSS_THRESHOLD: float = 0.8  # speed_min / speed_expected < X

# Aceleração precoce/agressiva
TC_ACTIVE_THRESHOLD: float = 0.1           # tc > X → TC ativado
WHEEL_SLIP_THRESHOLD: float = 0.15         # wheel_slip médio > X

# Entrada de curva rápida demais
GFORCE_LATERAL_THRESHOLD: float = 2.5     # gforce_x > X (G)
STEERING_THRESHOLD: float = 0.6           # steering normalizado > X

# Ponto de troca subótimo
RPM_SHIFT_MARGIN: float = 0.05            # % do maxRpm fora da faixa ótima

# === TTS — Fase 6 ===
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "none")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")

# === Logging ===
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
