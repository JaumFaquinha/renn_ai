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
SAMPLING_RATE_HZ: int = int(os.getenv("SAMPLING_RATE_HZ", "20"))  # Frequência de leitura da Shared Memory
MINI_SECTOR_SIZE: float = 0.01      # Tamanho de mini-setor em spline normalizada (~1% da pista)

# === Campos da Shared Memory ===
SPLINE_POSITION_FIELD: str = "normalizedCarPosition"
PERFORMANCE_METER_FIELD: str = "performanceMeter"

# === Validação de Volta ===
# Descarta volta se qualquer componente de dano ultrapassar este valor (0.0–1.0)
CAR_DAMAGE_THRESHOLD: float = float(os.getenv("CAR_DAMAGE_THRESHOLD", "0.1"))

# Número mínimo de mini-setores para uma volta ser considerada válida
MIN_SECTORS_PER_LAP: int = int(os.getenv("MIN_SECTORS_PER_LAP", "80"))  # ~80% da pista mínimo

# === Validação de Telemetria ===
# Clutch do AC é 0.0–1.0; valores acima indicam campo corrompido (bug de offset)
CLUTCH_MAX_VALUE: float = float(os.getenv("CLUTCH_MAX_VALUE", "1.0"))

# delta_vs_best acima deste limiar (segundos) indica bug de sessão ou volta inválida
DELTA_OUTLIER_THRESHOLD_S: float = float(os.getenv("DELTA_OUTLIER_THRESHOLD_S", "60.0"))

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

# === Supabase — Fase 7 ===
SUPABASE_ENABLED: bool = os.getenv("SUPABASE_ENABLED", "false").lower() == "true"
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")       # service_role key
SUPABASE_USER_ID: str = os.getenv("SUPABASE_USER_ID", "")  # UUID fixo do piloto

# === TTS — Fase 6 ===
# Provider: "none" | "pyttsx3" | "edge_tts" | "elevenlabs" | "azure"
# Recomendação: "edge_tts" (free, neural, online) com fallback "pyttsx3" (offline)
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "none")
TTS_FALLBACK: str = os.getenv("TTS_FALLBACK", "pyttsx3")
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "pt-BR")
TTS_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "")
# Truncamento de mensagens longas — reduz TTFB de síntese
TTS_MAX_MESSAGE_CHARS: int = int(os.getenv("TTS_MAX_MESSAGE_CHARS", "140"))
# Cooldown entre alertas (segundos) — evita overlap auditivo
TTS_MIN_INTERVAL_S: float = float(os.getenv("TTS_MIN_INTERVAL_S", "3.0"))

ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")

# === Logging ===
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
