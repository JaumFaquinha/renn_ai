# CLAUDE.md — Engenheiro de Corrida IA

> Contrato técnico do projeto para uso via Claude Code (Cowork).
> Leia este arquivo integralmente antes de qualquer ação no repositório.

---

## 1. Identidade do Projeto

**Nome:** Engenheiro de Corrida IA  
**Objetivo:** Desenvolver um analisador de telemetria em tempo real que atue como engenheiro de corrida limitado, identificando onde o piloto humano está perdendo tempo durante voltas no Assetto Corsa.  
**Linguagem:** Python 3.10+  
**Plataforma alvo:** Windows (obrigatório — Shared Memory do AC é exclusiva do Windows)  
**Status atual:** Módulo B ativo — Analisador de telemetria do piloto humano.

### O que este projeto NÃO é (por enquanto)
- Não é um agente RL que pilota o carro autonomamente (Módulo A — backlog futuro)
- Não usa modelos de linguagem (LLM) em nenhuma camada
- Não é um sistema de setup de carro (suspensão, aerodinâmica, etc.)

---

## 2. Arquitetura Técnica

### Stack definida e aprovada

| Camada | Ferramenta | Justificativa |
|---|---|---|
| Leitura de dados | `mmap` (Python stdlib) | Acesso direto à Shared Memory do AC |
| Ambiente RL (futuro Módulo A) | `Gymnasium` | Padrão da indústria para envs customizados |
| Framework RL (futuro Módulo A) | `Stable-Baselines3` | Maduro, documentado, PPO/SAC prontos |
| Monitoramento | `TensorBoard` | Integrado ao SB3, visualização de métricas |
| Persistência | `JSON` por volta | Schema definido na seção 6 |
| Saída futura | `ElevenLabs` ou `Azure TTS` | Integração de voz — Fase 6 |

### Fluxo do Pipeline (Módulo B)

```
Shared Memory (AC) 
    → [20-50Hz] SPageFilePhysics + SPageFileGraphic
    → SharedMemoryReader
    → LapRecorder (agrupamento por mini-setor ~0.01 spline)
    → LapAnalyzer (delta_vs_best por setor)
    → PatternDetector (correlação telemetria → causa)
    → ReportBuilder (output estruturado)
    → [Futuro] TTSIntegration (voz do engenheiro)
```

### Decisões arquiteturais registradas

| Decisão | Escolha | Motivo |
|---|---|---|
| Tipo de saída | `float` (delta em segundos) | Comunica magnitude da perda, não apenas binário |
| Granularidade | Mini-setores de ~0.01 na spline | ~1% da pista por ponto, precisão suficiente |
| Label principal | `performanceMeter` | Delta em tempo real já calculado pelo AC |
| Treino | Por pista, auto-supervisionado | Próprias voltas geram exemplos sem rotulação manual |

---

## 3. Estrutura de Diretórios

```
racing_engineer_ia/
│
├── CLAUDE.md                        # Este arquivo
├── README.md                        # Documentação pública do projeto
├── requirements.txt                 # Dependências fixadas com versão
├── .env.example                     # Variáveis de ambiente (sem secrets)
│
├── config/
│   ├── settings.py                  # Configurações globais (sampling rate, etc.)
│   └── track_maps/                  # Mapeamentos de curvas por pista (.json)
│       └── example_track.json
│
├── src/
│   ├── __init__.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── shared_memory_reader.py  # Leitura via mmap das structs do AC
│   │   ├── physics_page.py          # SPageFilePhysics — ctypes struct
│   │   ├── graphics_page.py         # SPageFileGraphic — ctypes struct
│   │   └── static_page.py           # SPageFileStatic — ctypes struct
│   │
│   ├── recording/
│   │   ├── __init__.py
│   │   ├── lap_recorder.py          # Gravação de volta em mini-setores
│   │   └── sector_aggregator.py     # Agrupamento e sumarização por setor
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── lap_analyzer.py          # Comparação vs melhor volta (delta)
│   │   ├── pattern_detector.py      # Correlação telemetria → causa de perda
│   │   └── report_builder.py        # Construção do relatório de saída
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── sector_model.py          # Modelo de análise por setor (ML leve)
│   │
│   └── output/
│       ├── __init__.py
│       ├── console_reporter.py      # Saída em terminal
│       └── tts_integration.py       # Placeholder — Fase 6 (ElevenLabs/Azure)
│
├── data/
│   ├── laps/                        # Voltas gravadas em JSON por sessão
│   └── models/                      # Modelos treinados por pista
│
├── tests/
│   ├── __init__.py
│   ├── test_memory_reader.py
│   ├── test_lap_recorder.py
│   ├── test_pattern_detector.py
│   └── fixtures/                    # JSONs de volta para testes offline
│
└── scripts/
    ├── run_session.py               # Entrypoint principal
    └── map_track.py                 # Script de mapeamento de curvas (manual)
```

---

## 4. Mapeamento Completo da Shared Memory

> Mapeamento integral de todos os campos disponíveis no AC.
> Organizado por categoria de uso para facilitar expansão futura de escopo.

---

### 4.1 Campos Ativos — Modelo Atual (Módulo B)

#### Posicionamento e Tempo

| Campo | Struct | Tipo | Uso no Modelo |
|---|---|---|---|
| `normalizedCarPosition` | SPageFileGraphic | float 0–1 | Índice primário do mini-setor |
| `trackSPlineLength` | SPageFileStatic | float | Conversão posição → metros reais |
| `performanceMeter` | SPageFilePhysics | float | **Label principal** — delta vs melhor volta |
| `iCurrentTime` | SPageFileGraphic | int (ms) | Tempo atual da volta |
| `iBestTime` | SPageFileGraphic | int (ms) | Referência da melhor volta |
| `iLastTime` | SPageFileGraphic | int (ms) | Referência da volta anterior |
| `lastSectorTime` | SPageFileGraphic | int (ms) | Tempo do último setor completo |
| `currentSectorIndex` | SPageFileGraphic | int | Setor atual da volta |

#### Inputs do Piloto

| Campo | Struct | Tipo | Uso no Modelo |
|---|---|---|---|
| `gas` | SPageFilePhysics | float 0–1 | Input de aceleração |
| `brake` | SPageFilePhysics | float 0–1 | Input de frenagem |
| `steerAngle` | SPageFilePhysics | float | Trajetória executada |
| `gear` | SPageFilePhysics | int | Identificar trocas incorretas |
| `clutch` | SPageFilePhysics | float 0–1 | Uso de embreagem |
| `rpms` | SPageFilePhysics | int | ⚡ **ADICIONADO** — Faixa de potência por marcha; essencial para detectar trocas subótimas |

#### Velocidade e Dinâmica

| Campo | Struct | Tipo | Uso no Modelo |
|---|---|---|---|
| `speedKmh` | SPageFilePhysics | float | Velocidade instantânea |
| `accG[3]` | SPageFilePhysics | float[3] | Força G por eixo (x, y, z) — qualidade de curva |
| `velocity[3]` | SPageFilePhysics | float[3] | Vetor de velocidade (mundo) |
| `localVelocity[3]` | SPageFilePhysics | float[3] | Vetor de velocidade local |
| `localAngularVel[3]` | SPageFilePhysics | float[3] | ⚡ **ADICIONADO** — Velocidade angular local; detecta oversteer/understeer com precisão |

#### Tração e Sistemas Ativos

| Campo | Struct | Tipo | Uso no Modelo |
|---|---|---|---|
| `wheelSlip[4]` | SPageFilePhysics | float[4] | Escorregamento por pneu [FL,FR,RL,RR] |
| `tc` | SPageFilePhysics | float | Slip limit do traction control |
| `abs` | SPageFilePhysics | float | Slip limit do ABS |
| `drs` | SPageFilePhysics | float | DRS ativo (0 ou 1) |
| `drsAvailable` | SPageFilePhysics | int | DRS disponível na zona atual |
| `brakeBias` | SPageFilePhysics | float 0–1 | ⚡ **ADICIONADO** — Distribuição de frenagem; contexto essencial para interpretar padrões de brake |

#### Contexto de Sessão (normalização)

| Campo | Struct | Tipo | Uso no Modelo |
|---|---|---|---|
| `surfaceGrip` | SPageFileGraphic | float | ⚡ **ADICIONADO** — Grip evoluindo na sessão; normaliza wheelSlip entre voltas |
| `airTemp` | SPageFilePhysics | float | Temperatura ambiente da sessão |
| `roadTemp` | SPageFilePhysics | float | Temperatura do asfalto |
| `tyreCompound[33]` | SPageFileGraphic | wchar_t | ⚡ **ADICIONADO** — Composto ativo (SM/MS/SS/H etc.); metadado de volta (`laps.tyre_compound`), não de mini-setor. Propagado via `_tyre_compound` no snapshot. Essencial para que PatternDetector ajuste limiares por composto futuramente. |

---

### 4.2 Campos de Filtro e Validação de Volta

Campos não gravados no schema de mini-setor mas usados para validar se a volta/setor é utilizável.

| Campo | Struct | Regra de Descarte |
|---|---|---|
| `status` | SPageFileGraphic | Descartar se ≠ `AC_LIVE (2)` |
| `flag` | SPageFileGraphic | Descartar setores com `AC_YELLOW_FLAG` ou superior |
| `numberOfTyresOut` | SPageFilePhysics | Descartar setor se > 0 |
| `pitLimiterOn` | SPageFilePhysics | Descartar setor se = 1 |
| `isInPit` | SPageFileGraphic | Descartar volta se = 1 em qualquer ponto |
| `isInPitLane` | SPageFileGraphic | Descartar volta se = 1 |
| `penaltyTime` | SPageFileGraphic | Descartar volta se > 0 |
| `carDamage[5]` | SPageFilePhysics | Descartar volta se qualquer valor > threshold configurável |
| `isAIControlled` | SPageFilePhysics | Descartar volta se = 1 |

---

### 4.3 Campos de Contexto Estático (Leitura Única por Sessão)

Lidos uma vez ao iniciar a sessão. Não variam por mini-setor.

| Campo | Struct | Uso |
|---|---|---|
| `trackSPlineLength` | SPageFileStatic | Comprimento da pista em metros |
| `trackConfiguration` | SPageFileStatic | Identificar layout em pistas multi-config |
| `sectorCount` | SPageFileStatic | Número de setores oficiais da pista |
| `maxRpm` | SPageFileStatic | Referência para análise de faixa de potência por marcha |
| `maxTorque` | SPageFileStatic | Perfil de torque do carro |
| `maxPower` | SPageFileStatic | Perfil de potência do carro |
| `hasDRS` | SPageFileStatic | Habilitar/desabilitar análise de DRS |
| `hasERS` | SPageFileStatic | Habilitar análise de ERS futuramente |
| `hasKERS` | SPageFileStatic | Habilitar análise de KERS futuramente |
| `carModel` | SPageFileStatic | Identificação do carro na sessão |
| `track` | SPageFileStatic | Chave para carregar track_map correto |
| `penaltiesEnabled` | SPageFileStatic | Contexto de validade de cortes |
| `completedLaps` | SPageFileGraphic | Número de voltas completadas na sessão |
| `sessionTimeLeft` | SPageFileGraphic | Contexto de fim de sessão |
| `session` | SPageFileGraphic | Tipo: treino, qualificação, corrida, hotlap — lido via `_session_int_to_str()` e gravado em `sessions.session_type` |
| `aidAutoClutch` | SPageFileStatic | int | Se ativo, o campo `clutch` nos mini-setores reflete o jogo, não o piloto. Logar warning no início da sessão. |
| `aidStability` | SPageFileStatic | float | Contextualiza comportamentos de oversteer/understeer interpretados pelo PatternDetector. |
| `drsEnabled` | SPageFilePhysics | int | Indica se DRS está habilitado pelo race director (distinto de `drs` = flap aberto e `drsAvailable` = em zona). |
| `tyreRadius[4]` | SPageFileStatic | float[4] | Raio do pneu por roda (m) — necessário para calcular slip ratio real via `wheelAngularSpeed[4]` (escopo futuro 4.4). |

---

### 4.4 Campos Mapeados para Escopo Futuro

Lidos e armazenados no JSON raw mas não usados no modelo atual.
Disponíveis para expansão sem necessidade de regravar sessões antigas.

| Campo | Struct | Expansão Prevista |
|---|---|---|
| `carCoordinates[3]` | SPageFileGraphic | Análise de trajetória — linha real vs ótima |
| `heading` | SPageFilePhysics | Ângulo de entrada/saída de curva |
| `wheelAngularSpeed[4]` | SPageFilePhysics | Cálculo preciso de slip ratio real por roda |
| `wheelsPressure[4]` | SPageFilePhysics | Correlação pressão de pneu com comportamento |
| `distanceTraveled` | SPageFileGraphic | Métrica de consistência entre voltas |
| `position` | SPageFileGraphic | Contexto de corrida — pressão do tráfego |
| `split` | SPageFileGraphic | Tempo parcial por setor oficial |

---

### 4.5 Schema de Gravação por Mini-Setor (Atualizado)

> **Nota:** `speed_min` é um **campo computado** (mínimo de `speedKmh` dentro do mini-setor), não um campo da Shared Memory. Não deve ser confundido com um campo lido via mmap.
>
> **Nota:** `tyre_compound` é um **metadado de volta** gravado em `laps.tyre_compound`, não nos mini-setores. Um composto não muda dentro de uma mesma volta.
>
> **Nota (2026-04-25, Proposal P1):** Inputs do piloto e sistemas ativos têm 4 estatísticas: `mean` (campo base), `_max` (peak), `_min` (valley), `_std` (variabilidade). A média sozinha aniquila a dinâmica intra-setor (~22 snapshots em ~1.1s @20Hz). Campos com tripla extra: `throttle`, `brake`, `steering`, `wheel_slip_fl/fr/rl/rr`, `tc_active`, `abs_active` (9 inputs × 3 stats = 27 colunas adicionais). Validação empírica em `model_validation_playbook.md` Observação C.

```json
{
  "track_position": 0.23,
  "delta_vs_best": 0.18,
  "delta_per_sector": 0.045,

  "throttle": 0.71, "throttle_max": 1.0,  "throttle_min": 0.42, "throttle_std": 0.18,
  "brake": 0.18,    "brake_max": 0.92,    "brake_min": 0.0,     "brake_std": 0.31,
  "steering": 0.34, "steering_max": 0.51, "steering_min": 0.20, "steering_std": 0.09,

  "gear": 3,
  "rpms": 8400,
  "clutch": 0.0,
  "speed_kmh": 142.3,
  "speed_min": 87.1,
  "gforce_x": 2.1,
  "gforce_y": 0.3,
  "gforce_z": 1.8,
  "local_ang_vel_x": 0.12,
  "local_ang_vel_y": 0.04,
  "local_ang_vel_z": 0.09,

  "wheel_slip_fl": 0.08, "wheel_slip_fl_max": 0.18, "wheel_slip_fl_min": 0.02, "wheel_slip_fl_std": 0.04,
  "wheel_slip_fr": 0.09, "wheel_slip_fr_max": 0.20, "wheel_slip_fr_min": 0.03, "wheel_slip_fr_std": 0.05,
  "wheel_slip_rl": 0.12, "wheel_slip_rl_max": 0.25, "wheel_slip_rl_min": 0.04, "wheel_slip_rl_std": 0.06,
  "wheel_slip_rr": 0.11, "wheel_slip_rr_max": 0.24, "wheel_slip_rr_min": 0.04, "wheel_slip_rr_std": 0.06,

  "tc_active": 0.05,  "tc_active_max": 1.0,  "tc_active_min": 0.0, "tc_active_std": 0.22,
  "abs_active": 0.10, "abs_active_max": 1.0, "abs_active_min": 0.0, "abs_active_std": 0.30,

  "drs_active": 0,
  "drs_available": 0,
  "brake_bias": 0.58,
  "surface_grip": 0.97,
  "air_temp": 24.0,
  "road_temp": 31.5
}
```

---

### 4.6 Campos Explicitamente Descartados

| Campo | Motivo |
|---|---|
| `tyreWear[4]` | Performance do carro, não input do piloto |
| `tyreDirtyLevel[4]` | Condição do carro |
| `suspensionTravel[4]` | Dado de setup |
| `rideHeight[2]` | Dado de setup |
| `brakeTemp[4]` | Estado térmico do carro |
| `tyreCoreTemperature[4]`, `tyreTempI/M/O[4]` | Estado térmico do carro |
| `wheelLoad[4]` | Física do chassi |
| `kersCharge`, `kersInput`, `kersCurrentKJ` | Sistema do carro |
| `turboBoost` | Sistema do carro |
| `cgHeight`, `pitch`, `roll` | Geometria do chassi |
| `windSpeed`, `windDirection` | Constante na sessão |
| `camberRAD[4]` | Setup do carro |
| `ballast` | Contexto de multiplayer |
| `finalFF` | Force Feedback — irrelevante para telemetria |
| `tyreContactPoint/Normal/Heading[4][3]` | Física interna — complexidade sem ganho analítico |
| `autoShifterOn`, `engineBrake` | Setup do carro |
| `ersRecoveryLevel`, `ersPowerLevel`, `ersHeatCharging`, `ersIsCharging` | Sistema do carro |
| `airDensity` | Constante durante a sessão, sem variação por setor |
| `replayTimeMultiplier` | Controle de replay |
| `playerName`, `playerSurname`, `playerNick` | Metadados do jogador |
| `deprecated_1`, `deprecated_2` | Obsoletos — documentados para não uso |
| `smVersion[15]`, `acVersion[15]` | SPageFileStatic | Versões de compatibilidade — lidas pelo struct mas sem uso analítico |
| `aidFuelRate`, `aidTireRate`, `aidMechanicalDamage` | SPageFileStatic | Parâmetros de simulação do AC — afetam o carro, não o piloto |
| `allowTyreBlankets` | SPageFileStatic | Configuração de aquecimento de pneus — contexto de setup |
| `aidAutoBlip` | SPageFileStatic | Auto-blip de trocas — confunde análise de clutch, mas sem modelo de clutch ativo por enquanto |
| `kersMaxJ`, `ersMaxJ`, `engineBrakeSettingsCount`, `ersPowerControllerCount` | SPageFileStatic | Capacidades de sistemas avançados do carro — sem uso no modelo atual |
| `maxFuel` | SPageFileStatic | Capacidade máxima de combustível — relevante apenas se `fuel` for adicionado ao pipeline |
| `suspensionMaxTravel[4]` | SPageFileStatic | Dado de geometria de setup |
| `isTimedRace`, `hasExtraLap`, `reversedGridPositions` | SPageFileStatic | Variantes de formato de corrida — sem impacto na análise de piloto |
| `carSkin[33]`, `isOnline`, `numCars`, `numberOfSessions` | SPageFileStatic | Metadados de sessão/visual |
| `PitWindowStart`, `PitWindowEnd` | SPageFileStatic | Estratégia de corrida — fora do escopo do Módulo B |
| `numberOfLaps` | SPageFileGraphic | Total de voltas da sessão — contexto de formato, não de piloto |
| `idealLineOn` | SPageFileGraphic | Estado da UI — irrelevante para telemetria |
| `mandatoryPitDone` | SPageFileGraphic | Estratégia de pit — fora do escopo do Módulo B |
| `isSetupMenuVisible`, `mainDisplayIndex`, `secondaryDisplayIndex` | SPageFileGraphic | Estado da UI |
| `TC`, `TCLEVEL`, `trackGripStatus` | SPageFileGraphic | Estado da UI do TC — redundante com `tc` da physics |
| `rainLights`, `flashingLights`, `lightsStage`, `wiperLV` | SPageFileGraphic | Estado de luzes e limpadores |
| `exhaustTemperature` | SPageFileGraphic | Estado térmico do carro |
| `driverStintTotalTimeLeft`, `driverStintTimeLeft` | SPageFileGraphic | Tempo de stint (endurance) — fora do escopo atual |
| `rainTyres` | SPageFileGraphic | Tipo de pneu para chuva — coberto por `tyreCompound` |
| `playerCarID`, `pitLimiterSpeed` | SPageFilePhysics | Identificador interno e velocidade de pit limiter — sem uso analítico |

---

## 5. Padrões de Causa de Perda de Tempo

Tabela de correlação usada pelo `PatternDetector`:

| Padrão de Telemetria | Causa Identificada |
|---|---|
| `brake` alto + `abs` ativo + `speed_min` baixo | Frenagem tardia com bloqueio |
| `throttle` baixo na saída + `tc` ativo + `wheel_slip` alto | Aceleração precoce/agressiva |
| `gforce_x` alto + `steering` alto + `speed` alto | Entrada de curva rápida demais |
| `gear` incorreta + RPM fora do range | Ponto de troca subótimo |
| `throttle` parcial em reta longa | Saída de curva anterior comprometida |

---

## 6. Plano de Fases com Critérios de Aceite

### FASE 1 — Shared Memory Reader
**Objetivo:** Ler em tempo real os dados das três structs do AC via `mmap`.

**Entregáveis:**
- `src/memory/shared_memory_reader.py`
- `src/memory/physics_page.py` (ctypes struct)
- `src/memory/graphics_page.py` (ctypes struct)
- `src/memory/static_page.py` (ctypes struct)

**Critérios de aceite:**
- [ ] Leitura estável a 20Hz sem memory leaks
- [ ] Todos os campos da seção 4 acessíveis e tipados corretamente
- [ ] Graceful handling se AC não estiver rodando
- [ ] Teste offline com fixture JSON (sem necessidade do jogo)
- [ ] Log estruturado com timestamp por leitura

---

### FASE 2 — Lap Recorder
**Objetivo:** Gravar cada volta em mini-setores e persistir em JSON ao cruzar a linha de chegada.

**Entregáveis:**
- `src/recording/lap_recorder.py`
- `src/recording/sector_aggregator.py`
- `data/laps/` com voltas de exemplo

**Critérios de aceite:**
- [ ] Detecção correta do início e fim de volta via `normalizedCarPosition`
- [ ] Agrupamento em mini-setores de ~0.01 na spline
- [ ] JSON gravado automaticamente ao cruzar a linha de chegada
- [ ] Schema idêntico ao definido na seção 4
- [ ] Descartar voltas com `carDamage` > threshold configurável
- [ ] Descartar voltas com `isInPit = 1` em qualquer ponto

---

### FASE 3 — Lap Analyzer (Delta vs Best)
**Objetivo:** Comparar a volta atual contra a melhor volta gravada, setor a setor.

**Entregáveis:**
- `src/analysis/lap_analyzer.py`
- Output: lista de mini-setores com `delta_vs_best` calculado

**Critérios de aceite:**
- [ ] Alinhamento correto de mini-setores entre voltas diferentes
- [ ] `delta_vs_best` consistente com `performanceMeter` do AC (margem < 50ms)
- [ ] Identificação correta dos top 5 setores com maior perda
- [ ] Funciona com mínimo de 2 voltas gravadas
- [ ] Teste unitário com fixtures de volta reais

---

### FASE 4 — Pattern Detector
**Objetivo:** Correlacionar padrões de telemetria com causas específicas de perda de tempo.

**Entregáveis:**
- `src/analysis/pattern_detector.py`
- Tabela de padrões implementada (seção 5 deste arquivo)

**Critérios de aceite:**
- [ ] Todos os 5 padrões da tabela implementados
- [ ] Cada setor com perda retorna ao menos uma causa identificada
- [ ] Confiança da causa expressada como float (0.0 a 1.0)
- [ ] Casos ambíguos (múltiplas causas) retornam lista ordenada
- [ ] Teste unitário para cada padrão individualmente

---

### FASE 5 — Report Builder + Console Output
**Objetivo:** Consolidar análise em relatório legível e exibir no terminal em tempo real.

**Entregáveis:**
- `src/analysis/report_builder.py`
- `src/output/console_reporter.py`

**Critérios de aceite:**
- [ ] Relatório ordenado por magnitude de perda (maior primeiro)
- [ ] Exibe: zona da pista, posição na spline, delta em segundos, causa
- [ ] Mapeamento posição → nome da curva (via `config/track_maps/`)
- [ ] Output legível em terminal sem bibliotecas de UI
- [ ] Latência do relatório pós-volta < 2 segundos

---

### FASE 6 — Integração TTS (Voz do Engenheiro)
**Objetivo:** Transformar o relatório em feedback de voz via ElevenLabs ou Azure TTS.

**Entregáveis:**
- `src/output/tts_integration.py`
- Suporte a ambos os providers com fallback configurável

**Critérios de aceite:**
- [ ] Provider configurável via `.env` sem alteração de código
- [ ] Síntese de voz em < 3 segundos após fim da volta
- [ ] Fila de mensagens para não bloquear o loop principal
- [ ] Funciona offline com Azure TTS local (opcional)
- [ ] Placeholder funcional sem API key (modo texto apenas)

---

## 7. Regras para o Claude Code

### O que você PODE fazer autonomamente
- Criar arquivos dentro da estrutura definida na seção 3
- Escrever testes unitários para código já implementado
- Refatorar código dentro de um módulo sem alterar interfaces públicas
- Adicionar docstrings e type hints
- Instalar dependências já listadas no `requirements.txt`

### O que você DEVE propor antes de implementar
- Qualquer nova dependência não listada no `requirements.txt`
- Mudanças na estrutura de diretórios
- Alterações no schema JSON da seção 4
- Mudanças em interfaces públicas entre módulos
- Qualquer decisão que afete mais de um módulo simultaneamente
- Implementação de algoritmos de ML/modelo — apresentar abordagem antes

### Como propor uma decisão técnica
```
## Proposta Técnica: [título curto]

**Contexto:** Por que essa decisão surgiu?
**Opções consideradas:**
  - Opção A: [descrição + trade-offs]
  - Opção B: [descrição + trade-offs]
**Recomendação:** [opção + justificativa técnica]
**Impacto:** Quais arquivos/fases seriam afetados?
```

### Onde registrar decisões aprovadas
Após aprovação, documentar em `docs/decisions/ADR-XXX-titulo.md` seguindo o formato Architecture Decision Record (ADR).

---

## 8. Convenções de Código

```python
# Tipagem obrigatória em todas as funções públicas
def calculate_delta(
    current_lap: list[dict],
    best_lap: list[dict]
) -> list[dict]:
    ...

# Logging estruturado — sem print() no código de produção
import logging
logger = logging.getLogger(__name__)
logger.info("Volta gravada", extra={"lap_time_ms": 82340, "sectors": 98})

# Constantes em UPPER_SNAKE_CASE em settings.py
SAMPLING_RATE_HZ: int = 20
MINI_SECTOR_SIZE: float = 0.01
SPLINE_POSITION_FIELD: str = "normalizedCarPosition"
```

**Regras gerais:**
- PEP 8 obrigatório
- Type hints em todas as funções públicas
- Docstring em todas as classes e funções públicas
- Sem `print()` fora de scripts — usar `logging`
- Testes para toda lógica de análise e detecção de padrões

---

## 9. Referências Técnicas do Projeto

| Documento | Localização | Conteúdo |
|---|---|---|
| Arquitetura de Análise de Volta | `docs/engenheiro_corrida_ia.docx` | Especificação original do Módulo B |
| AC Shared Memory Documentation | `docs/ACSharedMemoryDocumentation.pdf` | Structs e campos da Shared Memory |

> Sempre consultar esses documentos antes de implementar qualquer interação com a Shared Memory.

---

*Última atualização: 2026-03-05*  
*Próxima fase ativa: FASE 1 — Shared Memory Reader*
