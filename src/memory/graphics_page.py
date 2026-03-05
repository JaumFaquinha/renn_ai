"""
SPageFileGraphic — Struct ctypes mapeada à Shared Memory do Assetto Corsa.

Referência: docs/ACSharedMemoryDocumentation.pdf
Shared Memory name: Local\\acpmf_graphics
"""

import ctypes
from ctypes import c_float, c_int, c_wchar


# Constantes de status da sessão
AC_OFF = 0
AC_REPLAY = 1
AC_LIVE = 2
AC_PAUSE = 3

# Constantes de tipo de sessão
AC_UNKNOWN = -1
AC_PRACTICE = 0
AC_QUALIFY = 1
AC_RACE = 2
AC_HOTLAP = 3
AC_TIME_ATTACK = 4
AC_DRIFT = 5
AC_DRAG = 6

# Constantes de bandeiras
AC_NO_FLAG = 0
AC_BLUE_FLAG = 1
AC_YELLOW_FLAG = 2
AC_BLACK_FLAG = 3
AC_WHITE_FLAG = 4
AC_CHECKERED_FLAG = 5
AC_PENALTY_FLAG = 6


class SPageFileGraphic(ctypes.Structure):
    """
    Espelho exato da struct SPageFileGraphic do AC SDK.

    Atualizada a ~25Hz pelo jogo. Contém dados de UI, posicionamento
    na pista e estado da sessão.
    """

    _fields_ = [
        # Identificador de pacote
        ("packetId", c_int),

        # === Estado da Sessão ===
        ("status", c_int),      # AC_OFF | AC_REPLAY | AC_LIVE | AC_PAUSE
        ("session", c_int),     # Tipo de sessão (AC_PRACTICE, AC_QUALIFY, etc.)

        # === Tempos Formatados (strings) ===
        ("currentTime", c_wchar * 15),  # Tempo atual formatado
        ("lastTime", c_wchar * 15),     # Último tempo formatado
        ("bestTime", c_wchar * 15),     # Melhor tempo formatado
        ("split", c_wchar * 15),        # Tempo parcial do setor atual

        # === Contadores e Posição ===
        ("completedLaps", c_int),   # Voltas completadas na sessão
        ("position", c_int),        # Posição na corrida

        # === Tempos em Milissegundos ===
        ("iCurrentTime", c_int),    # Tempo atual da volta (ms)
        ("iLastTime", c_int),       # Tempo da última volta (ms)
        ("iBestTime", c_int),       # Melhor tempo da sessão (ms)

        # === Sessão ===
        ("sessionTimeLeft", c_float),   # Tempo restante na sessão (s)
        ("distanceTraveled", c_float),  # Distância percorrida (m)

        # === Status de Pit ===
        ("isInPit", c_int),         # Na caixa: 0 ou 1
        ("currentSectorIndex", c_int), # Índice do setor atual (0-based)
        ("lastSectorTime", c_int),  # Tempo do último setor completo (ms)
        ("numberOfLaps", c_int),    # Número total de voltas da sessão

        # === Pneus ===
        ("tyreCompound", c_wchar * 33), # Nome do composto de pneu atual

        # === Replay ===
        ("replayTimeMultiplier", c_float), # Multiplicador de tempo no replay

        # === Posicionamento na Pista ===
        ("normalizedCarPosition", c_float),     # Posição normalizada 0.0–1.0 (spline)
        ("carCoordinates", c_float * 3),        # Coordenadas 3D do carro no mundo

        # === Penalidades e Bandeiras ===
        ("penaltyTime", c_float),   # Tempo de penalidade acumulado (s)
        ("flag", c_int),            # Bandeira atual (AC_NO_FLAG, AC_YELLOW_FLAG, etc.)
        ("idealLineOn", c_int),     # Linha ideal ativa: 0 ou 1
        ("isInPitLane", c_int),     # No corredor do pit: 0 ou 1

        # === Grip e Condições ===
        ("surfaceGrip", c_float),   # Grip da superfície 0.0–1.0 (evolui durante a sessão)

        # === Pit Obrigatório ===
        ("mandatoryPitDone", c_int), # Pit obrigatório realizado: 0 ou 1

        # === Condições Ambientais (constantes na sessão) ===
        ("windSpeed", c_float),
        ("windDirection", c_float),

        # === Setup e Interface ===
        ("isSetupMenuVisible", c_int),
        ("mainDisplayIndex", c_int),
        ("secondaryDisplayIndex", c_int),
        ("TC", c_int),
        ("TCLEVEL", c_int),
        ("trackGripStatus", c_int),
        ("rainLights", c_int),
        ("flashingLights", c_int),
        ("lightsStage", c_int),
        ("exhaustTemperature", c_float),
        ("wiperLV", c_int),

        # === Stint de Piloto (endurance) ===
        ("driverStintTotalTimeLeft", c_int),
        ("driverStintTimeLeft", c_int),
        ("rainTyres", c_int),
    ]
