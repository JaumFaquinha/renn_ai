"""
SPageFileStatic — Struct ctypes mapeada à Shared Memory do Assetto Corsa.

Referência: docs/ACSharedMemoryDocumentation.pdf
Shared Memory name: Local\\acpmf_static
"""

import ctypes
from ctypes import c_float, c_int, c_wchar


class SPageFileStatic(ctypes.Structure):
    """
    Espelho exato da struct SPageFileStatic do AC SDK.

    Atualizada uma única vez ao carregar a sessão. Contém metadados
    estáticos do carro, pista e configurações de sessão.
    """

    _fields_ = [
        # === Versões ===
        ("smVersion", c_wchar * 15),    # Versão da Shared Memory
        ("acVersion", c_wchar * 15),    # Versão do Assetto Corsa

        # === Configuração da Sessão ===
        ("numberOfSessions", c_int),    # Número de sessões
        ("numCars", c_int),             # Número de carros na sessão

        # === Identificação do Carro e Pista ===
        ("carModel", c_wchar * 33),         # Modelo do carro (ex: "ferrari_458_gt2")
        ("track", c_wchar * 33),            # Nome da pista (ex: "monza")
        ("playerName", c_wchar * 33),       # Nome do piloto — descartado
        ("playerSurname", c_wchar * 33),    # Sobrenome do piloto — descartado
        ("playerNick", c_wchar * 33),       # Nick do piloto — descartado

        # === Configuração da Pista ===
        ("sectorCount", c_int),         # Número de setores oficiais da pista

        # === Perfil de Performance do Carro ===
        ("maxTorque", c_float),         # Torque máximo (Nm)
        ("maxPower", c_float),          # Potência máxima (W)
        ("maxRpm", c_int),              # RPM máximo
        ("maxFuel", c_float),           # Capacidade máxima de combustível (L)

        # === Suspensão e Pneus (setup) ===
        ("suspensionMaxTravel", c_float * 4),   # Curso máximo de suspensão por roda (m)
        ("tyreRadius", c_float * 4),            # Raio do pneu por roda (m)

        # === Turbo e Setup ===
        ("maxTurboBoost", c_float),

        # === Campos Obsoletos ===
        ("deprecated_1", c_float),
        ("deprecated_2", c_float),

        # === Regras da Sessão ===
        ("penaltiesEnabled", c_int),    # Penalidades habilitadas: 0 ou 1
        ("aidFuelRate", c_float),       # Taxa de consumo de combustível (aid)
        ("aidTireRate", c_float),       # Taxa de desgaste de pneu (aid)
        ("aidMechanicalDamage", c_float), # Taxa de dano mecânico (aid)
        ("allowTyreBlankets", c_int),   # Cobertores de pneu permitidos
        ("aidStability", c_float),      # Aid de estabilidade
        ("aidAutoClutch", c_int),       # Embreagem automática ativa
        ("aidAutoBlip", c_int),         # Auto-blip ativo

        # === Sistemas do Carro ===
        ("hasDRS", c_int),              # Carro tem DRS: 0 ou 1
        ("hasERS", c_int),              # Carro tem ERS: 0 ou 1
        ("hasKERS", c_int),             # Carro tem KERS: 0 ou 1
        ("kersMaxJ", c_float),          # Capacidade máxima do KERS (kJ)
        ("engineBrakeSettingsCount", c_int),    # Número de configurações de engine brake
        ("ersPowerControllerCount", c_int),     # Número de configurações de potência ERS

        # === Geometria da Pista ===
        ("trackSPlineLength", c_float),         # Comprimento da spline da pista (m)
        ("trackConfiguration", c_wchar * 33),   # Configuração do layout (ex: "GP", "Junior")

        # === ERS Avançado ===
        ("ersMaxJ", c_float),           # Capacidade máxima do ERS (kJ)

        # === Formato da Corrida ===
        ("isTimedRace", c_int),         # Corrida por tempo: 0 ou 1
        ("hasExtraLap", c_int),         # Volta extra ao acabar o tempo

        # === Skin e Configuração Online ===
        ("carSkin", c_wchar * 33),          # Skin do carro
        ("reversedGridPositions", c_int),   # Posições de grid invertidas
        ("PitWindowStart", c_int),          # Abertura da janela de pit (lap/min)
        ("PitWindowEnd", c_int),            # Fechamento da janela de pit (lap/min)
        ("isOnline", c_int),                # Sessão online: 0 ou 1
    ]
