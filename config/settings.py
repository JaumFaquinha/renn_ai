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

# Pneus fora da pista: a volta só é invalidada com MAIS de N pneus fora.
# 2 (padrão) → invalida a partir de 3 pneus fora (2 pneus na zebra/grama é tolerado).
MAX_TYRES_OUT_ALLOWED: int = int(os.getenv("MAX_TYRES_OUT_ALLOWED", "2"))

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

# === Novos detectores (2026-06-16) ===
# Trail-braking excessivo: piloto continua freando dentro da curva
TRAIL_BRAKE_MIN_FLOOR: float = 0.10       # brake_min > X → ainda freando no fim do setor
TRAIL_BRAKE_STEERING_MIN: float = 0.40    # steering_max > X → realmente curvando
TRAIL_BRAKE_BRAKE_MAX: float = 0.30       # brake_max > X → frenagem não-desprezível

# Coasting no apex: nem freia nem acelera no meio da curva
COAST_THROTTLE_MAX: float = 0.20          # throttle (médio) < X
COAST_BRAKE_MAX: float = 0.10             # brake (médio) < X
COAST_STEERING_MIN: float = 0.30          # steering_max > X
COAST_SPEED_MAX: float = 130.0            # speed_min < X km/h (curva lenta/média)

# Understeer: steering alto + slip dianteiro alto sem TC
UNDERSTEER_STEERING_MIN: float = 0.50     # steering_max > X
UNDERSTEER_FRONT_SLIP_MIN: float = 0.15   # max(wheel_slip_fl/fr_max) > X

# Oversteer / correção: alta variabilidade de steering + slip traseiro
OVERSTEER_STEERING_STD_MIN: float = 0.10  # steering_std > X
OVERSTEER_REAR_SLIP_MIN: float = 0.20     # max(wheel_slip_rl/rr_max) > X
OVERSTEER_YAW_RATE_MIN: float = 0.30      # |local_ang_vel_z| > X rad/s

# Hesitação no throttle: oscila o pé na reta ou saída
THROTTLE_HESITATION_STD_MIN: float = 0.15 # throttle_std > X
THROTTLE_HESITATION_STEERING_MAX: float = 0.20  # steering < X (reta)
THROTTLE_HESITATION_SPEED_MIN: float = 100.0    # speed_kmh > X

# Over-braking sem ABS: agressivo demais, mata velocidade sem bloquear
OVER_BRAKING_BRAKE_MIN: float = 0.70      # brake_max > X
OVER_BRAKING_SPEED_MAX: float = 80.0      # speed_min < X km/h

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
# Cap de segurança para o texto enviado ao TTS. Acima disto, o corte é feito
# no fim da última frase completa (nunca no meio) — ver TTSIntegration._truncate.
# 350: acomoda a mensagem mais rica (contexto + pior zona + 2 causas ≈ 280 chars)
# com folga, sem truncar. O edge_tts sintetiza mensagens longas sem problema
# (fatia internamente em blocos de 4096 bytes).
TTS_MAX_MESSAGE_CHARS: int = int(os.getenv("TTS_MAX_MESSAGE_CHARS", "350"))
# Cooldown entre alertas (segundos) — evita overlap auditivo
TTS_MIN_INTERVAL_S: float = float(os.getenv("TTS_MIN_INTERVAL_S", "3.0"))

# Ganho de volume aplicado ao áudio antes do playback (edge_tts e elevenlabs,
# que compartilham o pipeline soundfile→sounddevice). 1.0 = original; 2.0 ≈ 2x
# de amplitude. O áudio é clampeado em [-1, 1] para evitar wrap-around; ganhos
# muito altos (>3–4x) tendem a distorcer/clipar sinais já próximos do full scale.
TTS_VOLUME_GAIN: float = float(os.getenv("TTS_VOLUME_GAIN", "1.0"))

ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")

# === Feedback de voz (mensagem do engenheiro) — Fase 6 ===
# Volta é "boa" se a anomalia máxima prevista pelo modelo nos setores
# reportados ficar abaixo deste valor (score 0.0 normal → 1.0 perda severa).
GOOD_LAP_ANOMALY_MAX: float = float(os.getenv("GOOD_LAP_ANOMALY_MAX", "0.30"))
# Sem modelo treinado: volta é "boa" se a perda total ficar abaixo disto (s).
GOOD_LAP_TOTAL_LOSS_MAX_S: float = float(os.getenv("GOOD_LAP_TOTAL_LOSS_MAX_S", "0.15"))
# Só menciona um segundo setor na mensagem se a perda dele for ≥ isto (s).
VOICE_SECONDARY_SECTOR_MIN_LOSS_S: float = float(
    os.getenv("VOICE_SECONDARY_SECTOR_MIN_LOSS_S", "0.10")
)
# Mensagem híbrida: numa volta boa/recorde, menciona a maior oportunidade
# restante só se o ganho potencial no pior setor for ≥ isto (s). Abaixo,
# a mensagem fica puramente elogiosa.
VOICE_HYBRID_MIN_GAIN_S: float = float(os.getenv("VOICE_HYBRID_MIN_GAIN_S", "0.08"))

# === Logging ===
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
