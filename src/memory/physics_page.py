"""
SPageFilePhysics — Struct ctypes mapeada à Shared Memory do Assetto Corsa.

Referência: docs/ACSharedMemoryDocumentation.pdf
Shared Memory name: Local\\acpmf_physics
"""

import ctypes
from ctypes import c_float, c_int


class SPageFilePhysics(ctypes.Structure):
    """
    Espelho exato da struct SPageFilePhysics do AC SDK.

    Atualizada a ~50Hz pelo jogo. Contém dados de física e inputs do piloto.
    Todos os campos estão declarados na ordem exata do SDK para garantir
    alinhamento correto da memória.
    """

    _fields_ = [
        # Identificador de pacote — incrementado a cada atualização
        ("packetId", c_int),

        # === Inputs do Piloto ===
        ("gas", c_float),               # Acelerador 0.0–1.0
        ("brake", c_float),             # Freio 0.0–1.0
        ("fuel", c_float),              # Combustível restante (litros)
        ("gear", c_int),                # Marcha atual (0=R, 1=N, 2=1ª, ...)
        ("rpms", c_int),                # RPM atual do motor
        ("steerAngle", c_float),        # Ângulo de esterçamento (rad, normalizado)
        ("speedKmh", c_float),          # Velocidade em km/h

        # === Dinâmica do Veículo ===
        ("velocity", c_float * 3),          # Vetor velocidade no referencial global (m/s)
        ("accG", c_float * 3),              # Aceleração em G [lateral, longitudinal, vertical]
        ("wheelSlip", c_float * 4),         # Slip por roda [FL, FR, RL, RR]
        ("wheelLoad", c_float * 4),         # Carga por roda [FL, FR, RL, RR] (N) — descartado
        ("wheelsPressure", c_float * 4),    # Pressão de pneu [FL, FR, RL, RR] (PSI)
        ("wheelAngularSpeed", c_float * 4), # Velocidade angular por roda [FL, FR, RL, RR] (rad/s)

        # === Estado dos Pneus (descartados no modelo atual) ===
        ("tyreWear", c_float * 4),              # Desgaste por pneu
        ("tyreDirtyLevel", c_float * 4),        # Sujeira por pneu
        ("tyreCoreTemperature", c_float * 4),   # Temperatura core por pneu (°C)
        ("camberRAD", c_float * 4),             # Câmber por roda (rad)
        ("suspensionTravel", c_float * 4),      # Curso de suspensão por roda (m)

        # === Sistemas Ativos ===
        ("drs", c_float),           # DRS ativo: 0.0 ou 1.0
        ("tc", c_float),            # Slip limit do Traction Control
        ("heading", c_float),       # Ângulo de heading do carro (rad)
        ("pitch", c_float),         # Pitch do chassi (rad) — descartado
        ("roll", c_float),          # Roll do chassi (rad) — descartado
        ("cgHeight", c_float),      # Altura do centro de gravidade (m) — descartado
        ("carDamage", c_float * 5), # Dano por zona [frente, trás, esq, dir, geral] 0.0–1.0
        ("numberOfTyresOut", c_int), # Número de pneus fora da pista
        ("pitLimiterOn", c_int),     # Pit limiter ativo: 0 ou 1
        ("abs", c_float),           # Slip limit do ABS

        # === KERS/ERS (descartados no modelo atual) ===
        ("kersCharge", c_float),    # Carga KERS (kJ)
        ("kersInput", c_float),     # Input KERS
        ("autoShifterOn", c_int),   # Câmbio automático ativo
        ("rideHeight", c_float * 2), # Altura de solo [frente, trás] (m)
        ("turboBoost", c_float),    # Pressão do turbo
        ("ballast", c_float),       # Lastro (kg)
        ("airDensity", c_float),    # Densidade do ar

        # === Condições Ambientais ===
        ("airTemp", c_float),       # Temperatura do ar (°C)
        ("roadTemp", c_float),      # Temperatura da pista (°C)

        # === Dinâmica no Referencial Local ===
        ("localVelocity", c_float * 3),     # Velocidade no referencial do carro (m/s)
        ("localAngularVel", c_float * 3),   # Velocidade angular local (rad/s) — detecta oversteer

        # === Performance e Sistemas Avançados ===
        ("finalFF", c_float),           # Force Feedback final — descartado
        ("performanceMeter", c_float),  # Delta vs melhor volta (s) — LABEL PRINCIPAL

        # === Motor e ERS (descartados) ===
        ("engineBrake", c_int),         # Nível de engine brake
        ("ersRecoveryLevel", c_int),    # Nível de recuperação ERS
        ("ersPowerLevel", c_int),       # Nível de potência ERS
        ("ersHeatCharging", c_int),     # Aquecimento ERS
        ("ersIsCharging", c_int),       # ERS carregando
        ("kersCurrentKJ", c_float),     # KERS atual (kJ)

        # === DRS ===
        ("drsAvailable", c_int),    # DRS disponível na zona atual: 0 ou 1
        ("drsEnabled", c_int),      # DRS habilitado para uso

        # === Temperatura de Freios (descartado) ===
        ("brakeTemp", c_float * 4), # Temperatura de freio por roda (°C)

        # === Embreagem ===
        ("clutch", c_float),        # Input de embreagem 0.0–1.0

        # === Temperatura de Pneus por Camada (descartado) ===
        ("tyreTempI", c_float * 4), # Temperatura interna
        ("tyreTempM", c_float * 4), # Temperatura média
        ("tyreTempO", c_float * 4), # Temperatura externa

        # === Controle IA ===
        ("isAIControlled", c_int),  # Carro controlado por IA: 0 ou 1

        # === Contato de Pneus (descartado — física interna) ===
        ("tyreContactPoint", (c_float * 3) * 4),   # Ponto de contato por roda
        ("tyreContactNormal", (c_float * 3) * 4),  # Normal de contato por roda
        ("tyreContactHeading", (c_float * 3) * 4), # Heading de contato por roda

        # === Distribuição de Frenagem ===
        ("brakeBias", c_float),     # Brake bias frente/trás 0.0–1.0

        # === Velocidade Local (duplicata para compatibilidade) ===
        ("localVelocity2", c_float * 3),

        # === Multiplayer e Setup ===
        ("playerCarID", c_int),
        ("pitLimiterSpeed", c_float),
        ("mandatoryPitDone", c_float),
        ("windSpeed", c_float),
        ("windDirection", c_float),
    ]
