# Renn.ai

> Analisador de telemetria em tempo real para **Assetto Corsa** que atua como um
> engenheiro de corrida virtual: identifica **onde** e **por que** o piloto humano
> está perdendo tempo em cada volta — e comunica o diagnóstico por voz, ao vivo.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue">
  <img alt="Plataforma" src="https://img.shields.io/badge/Plataforma-Windows-0078D6">
  <img alt="Status" src="https://img.shields.io/badge/Módulo%20B-Funcional-success">
  <img alt="ML" src="https://img.shields.io/badge/ML-GradientBoosting-orange">
</p>

---

## 🎥 Demonstração

[![Assista à demonstração no YouTube](https://img.youtube.com/vi/S78818iTQAo/maxresdefault.jpg)](https://youtu.be/S78818iTQAo)

> Clique na imagem acima para assistir ao vídeo no YouTube: <https://youtu.be/S78818iTQAo>

---

## Contexto e Problema

No automobilismo profissional, cada piloto tem ao seu lado um **engenheiro de
corrida** que analisa a telemetria e aponta, volta a volta, onde há tempo a
recuperar: uma frenagem tardia aqui, uma aceleração precoce ali, uma marcha errada
na saída de curva. O piloto amador de simuladores (*sim racing*) **não tem esse
suporte** — ele vê apenas o cronômetro, que diz *quanto* perdeu, mas nunca *onde*
nem *por quê*.

Ferramentas de telemetria existentes (MoTeC, Moza, Fanatec) exigem que o
piloto **pare, analise gráficos e interprete os dados sozinho** — uma barreira
técnica alta e um processo lento que quebra o fluxo de treino.

**Como dar a um piloto amador o feedback acionável de um engenheiro de corrida
real, em tempo real, sem que ele precise interpretar telemetria bruta?**

---

## Solução Proposta

Um pipeline que lê a **Shared Memory** do Assetto Corsa a 20 Hz, grava cada volta
em ~100 *mini-setores* (~1% da pista cada), compara contra a melhor volta de
referência e, ao cruzar a linha de chegada:

1. **Localiza** os setores onde mais se perdeu tempo (*delta vs. melhor volta*).
2. **Diagnostica a causa** combinando duas camadas:
   - **11 detectores heurísticos** baseados em física (frenagem tardia, understeer,
     trail-braking, coasting, troca de marcha subótima, etc.).
   - **Um modelo de Machine Learning** (`GradientBoostingRegressor`) treinado com as
     próprias voltas do piloto, que aprende o padrão "normal" da pista e pontua
     anomalias que as heurísticas fixas não capturam.
3. **Comunica por voz** (TTS em pt-BR), como um engenheiro pelo rádio: *"Volta boa.
   Você perdeu dois décimos na Variante della Roggia — frenagem tardia com
   bloqueio."*
4. **Persiste o histórico** (opcional, via Supabase) para acompanhar a evolução e
   carregar o *personal best* de sessões anteriores como referência.

Tudo isso roda **localmente e 100% offline por padrão** — a nuvem é opcional.

---

## Principais Funcionalidades

| Recurso | Descrição |
|---|---|
| **Leitura em tempo real** | `mmap` direto na Shared Memory do AC a 20 Hz (configurável), sem polling do jogo. |
| **Gravação por mini-setor** | ~100 mini-setores por volta com estatísticas `mean/max/min/std` de cada input. |
| **Análise de delta** | Alinhamento setor-a-setor contra a melhor volta; top-5 zonas de maior perda. |
| **11 detectores de padrão** | Correlaciona telemetria → causa específica de perda de tempo. |
| **Modelo de ML por pista** | `GradientBoosting` auto-supervisionado; *score* de anomalia 0–1 por setor. |
| **Feedback de voz** | TTS com 4 providers + fallback offline; mensagens ricas, positivas e híbridas. |
| **Persistência opcional** | Sessões, voltas e mini-setores no Supabase; *personal best* histórico. |
| **Validação de volta** | Descarta voltas inválidas (pit, dano, pneus fora, penalidades, sem foco). |

---

##  Arquitetura

### Pipeline (Módulo B)

```
┌─────────────────────┐
│  Assetto Corsa      │  Shared Memory (Windows)
│  SPageFilePhysics   │  ── 20–50 Hz ──┐
│  SPageFileGraphic   │                │
│  SPageFileStatic    │                ▼
└─────────────────────┘    ┌────────────────────────┐
                           │  SharedMemoryReader    │  leitura via mmap + ctypes
                           └───────────┬────────────┘
                                       ▼
                           ┌────────────────────────┐
                           │  LapRecorder           │  agrupa por mini-setor (~0.01 spline)
                           │  + SectorAggregator    │  + validação de volta
                           └───────────┬────────────┘
                                       ▼ (ao cruzar a linha)
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                                ▼
┌──────────────┐          ┌───────────────────────┐          ┌──────────────────┐
│ LapAnalyzer  │          │  PatternDetector      │          │  SectorModel     │
│ delta vs best│          │ 11 heurísticas físicas│          │  ML (anomalia)   │
└──────┬───────┘          └───────────┬───────────┘          └────────┬─────────┘
       └──────────────────────────────┼─────────────────────────────┘
                                       ▼
                           ┌────────────────────────┐
                           │  ReportBuilder         │  consolida + nomeia curvas
                           └───────────┬────────────┘
                                       ▼
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼
     ┌─────────────────┐    ┌──────────────────┐      ┌──────────────────┐
     │ ConsoleReporter │    │ TTSIntegration   │      │ LapUploader      │
     │ (terminal)      │    │ (voz pt-BR)      │      │ (Supabase async) │
     └─────────────────┘    └──────────────────┘      └──────────────────┘
```

### Stack técnica

| Camada | Ferramenta | Papel |
|---|---|---|
| Leitura de dados | `mmap` + `ctypes` (stdlib) | Acesso direto à Shared Memory do AC |
| Análise / ML | `scikit-learn` (GradientBoosting), `numpy` | Modelo de anomalia por setor |
| Tracking de experimentos | `MLflow` (opcional) | Métricas de treino e validação |
| Voz | `edge-tts` / `pyttsx3` / `ElevenLabs` / `Azure` | Feedback falado em tempo real |
| Persistência | `Supabase` (PostgreSQL) | Histórico de sessões e *personal bests* |
| Configuração | `python-dotenv` | Variáveis de ambiente / segredos |
| Qualidade | `pytest`, `pytest-cov`, `ruff` | Testes e lint |

---

## Estrutura do Projeto

```
renn_ai/
├── config/
│   ├── settings.py              # Constantes globais e thresholds (carregadas do .env)
│   └── track_maps/              # Mapeamento posição→curva por pista (monza.json)
├── src/
│   ├── memory/                  # Leitura da Shared Memory (ctypes structs + mmap)
│   ├── recording/               # LapRecorder + SectorAggregator (mini-setores)
│   ├── analysis/                # LapAnalyzer, PatternDetector, ReportBuilder
│   ├── models/                  # SectorModel (ML por pista)
│   ├── output/                  # ConsoleReporter, TTSIntegration, voice_message
│   └── persistence/             # SupabaseClient, LapUploader, QueryService
├── scripts/
│   ├── run_session.py           # ▶ Entrypoint principal (sessão ao vivo)
│   ├── train_model.py           # Treino offline do SectorModel
│   ├── query_history.py         # Consulta de histórico no Supabase
│   ├── map_track.py             # Mapeamento manual de curvas
│   └── benchmark_tts.py         # Benchmark de latência dos providers de TTS
├── database/                    # Schema SQL e migrations do Supabase
├── data/
│   ├── laps/                    # Voltas gravadas em JSON
│   └── models/                  # Modelos treinados (monza.pkl)
├── docs/                        # Documentação técnica e relatórios de validação
├── tests/                       # Testes unitários + fixtures
├── CLAUDE.md                    # Contrato técnico completo do projeto
└── requirements.txt
```

---

## Instalação

**Pré-requisitos:** Python 3.11+, Windows (a Shared Memory do AC é exclusiva do
Windows) e o Assetto Corsa instalado.

```bash
# 1. Clonar e entrar no diretório
git clone <repo-url> && cd renn_ai

# 2. (recomendado) criar venv
python -m venv .venv && .venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
copy .env.example .env        # depois edite o .env conforme necessário
```

> O projeto roda **100% offline** sem nenhuma chave: TTS em modo texto e Supabase
> desabilitado são os padrões.

---

## Uso

### Sessão ao vivo (principal)

Com o Assetto Corsa rodando e a Shared Memory habilitada:

```bash
python scripts/run_session.py --track monza
```

O programa aguarda o AC, conecta, e a cada volta completada exibe o relatório no
terminal e (se habilitado) fala o diagnóstico. `Ctrl+C` encerra com segurança.

| Argumento | Default | Descrição |
|---|---|---|
| `--track` | `monza` | ID da pista (carrega o `track_map` e o modelo correspondente). |
| `--rate` | `20` | Frequência de amostragem em Hz. |

### Treinar o modelo de uma pista

Depois de gravar algumas voltas (mínimo recomendado ~2 voltas / 200 mini-setores):

```bash
python scripts/train_model.py --track monza            # de data/laps/ e/ou Supabase
python scripts/train_model.py --track monza --verbose  # + feature importance
```

O modelo é salvo em `data/models/{track}.pkl` e carregado automaticamente na
próxima sessão.

### Consultar histórico (requer Supabase habilitado)

```bash
python scripts/query_history.py
```

---

## Configuração (`.env`)

Todos os comportamentos são ajustáveis por variável de ambiente — sem alterar
código. Destaques (lista completa comentada em [`.env.example`](.env.example)):

| Variável | Default | Função |
|---|---|---|
| `SAMPLING_RATE_HZ` | `20` | Frequência de leitura da Shared Memory. |
| `CAR_DAMAGE_THRESHOLD` | `0.1` | Descarta volta acima deste dano (0–1). |
| `MIN_SECTORS_PER_LAP` | `80` | Mínimo de mini-setores para a volta ser válida. |
| `TTS_PROVIDER` | `none` | `none` / `pyttsx3` / `edge_tts` / `elevenlabs` / `azure`. |
| `TTS_FALLBACK` | `pyttsx3` | Provider offline de reserva. |
| `SUPABASE_ENABLED` | `false` | Liga a persistência na nuvem. |

---

## Como o diagnóstico é construído

### 1. Schema de mini-setor

Cada volta é fatiada em ~100 mini-setores (~0.01 na spline normalizada). Para
preservar a **dinâmica intra-setor** (~22 amostras em ~1,1 s a 20 Hz), os inputs do
piloto e sistemas ativos são gravados com **4 estatísticas** — `mean`, `_max`,
`_min`, `_std`:

```jsonc
{
  "track_position": 0.23, "delta_vs_best": 0.18, "delta_per_sector": 0.045,
  "throttle": 0.71, "throttle_max": 1.0, "throttle_min": 0.42, "throttle_std": 0.18,
  "brake": 0.18, "brake_max": 0.92, ... ,
  "gear": 3, "rpms": 8400, "speed_kmh": 142.3, "speed_min": 87.1,
  "gforce_x": 2.1, "local_ang_vel_z": 0.09,
  "wheel_slip_fl": 0.08, ... , "tc_active": 0.05, "abs_active": 0.10
}
```

### 2. Detectores de padrão (heurísticos)

Onze regras baseadas em física, cada uma retornando uma causa e uma confiança
(0–1):

| Detector | Sintoma na telemetria |
|---|---|
| Frenagem tardia c/ bloqueio | `brake` alto + `abs` ativo + `speed_min` baixo |
| Aceleração precoce/agressiva | `throttle` na saída + `tc` ativo + `wheel_slip` alto |
| Entrada de curva rápida demais | `gforce_x` alto + `steering` alto + `speed` alto |
| Troca de marcha subótima | `gear` + RPM fora da faixa de potência |
| Saída de curva comprometida | `throttle` parcial em reta longa |
| Trail-braking excessivo | freio ainda ativo dentro do ápice |
| Coasting no ápice | nem freia nem acelera no meio da curva |
| Understeer | `steering` alto + slip dianteiro alto sem TC |
| Oversteer / correção | `steering_std` alto + slip traseiro + yaw rate |
| Hesitação no acelerador | `throttle_std` alto em reta |
| Frenagem excessiva | freio muito forte, mata velocidade demais |

### 3. Modelo de ML (`SectorModel`)

`GradientBoostingRegressor` (loss *Huber*) treinado **por pista**, de forma
**auto-supervisionada**: as próprias voltas geram os exemplos, com alvo
`delta_per_sector` (perda de tempo ocorrida *naquele* setor). O modelo aprende o
comportamento esperado por zona da pista e por carro (one-hot de `car_model`) e
emite um **score de anomalia 0–1** que complementa as heurísticas fixas.

A escolha de features, alvo e filtros é fruto de validação empírica iterativa —
documentada em [`docs/`](docs/) (relatórios v1→v3 e validações MLflow) — incluindo
remoção de *data leakage* (`brake_bias`, `surface_grip`, `clutch`), filtros
posicionais e de outliers, e calibração do *score* no p95 do treino.

---

## Persistência (Supabase — opcional)

Quando `SUPABASE_ENABLED=true`, cada sessão é persistida em PostgreSQL de forma
**assíncrona** (não bloqueia o loop de telemetria). Tabelas principais:

`sessions` · `laps` · `mini_sectors` · `lap_patterns` · `personal_bests`

Benefícios: o *personal best* histórico vira a referência de delta logo no início
da sessão; o histórico por posição alimenta tendências no relatório. O schema e as
migrations estão em [`database/`](database/). Uma trava de instância única evita
sessões duplicadas, e as queries são paginadas para contornar limites do servidor.

---

## Qualidade e Testes

```bash
pytest                 # suíte de testes unitários
pytest --cov=src       # com cobertura
ruff check .           # lint (PEP 8)
```

Cobertura de testes para memória, gravação, detecção de padrões, modelo, voz e
persistência (ver pasta [`tests/`](tests/)).

---

## Roadmap

| Fase | Status | Descrição |
|---|---|---|
| 1 — Shared Memory Reader | ✅ | Leitura via `mmap` das 3 structs do AC. |
| 2 — Lap Recorder | ✅ | Gravação por mini-setor + validação de volta. |
| 3 — Lap Analyzer | ✅ | Delta vs. melhor volta, setor a setor. |
| 4 — Pattern Detector | ✅ | 11 detectores de causa de perda. |
| 5 — Report Builder + Console | ✅ | Relatório legível com nome das curvas. |
| 6 — Integração TTS | ✅ | Voz do engenheiro (4 providers + fallback). |
| 7 — Persistência (Supabase) | ✅ | Histórico e *personal best* na nuvem. |
| 8 — Modelo de ML por pista | ✅ | `SectorModel` com score de anomalia. |
| Futuro — Modelos especializados | 🔭 Planejado | Novos modelos por aspecto (pneus, combustível, estratégia). **Sem pilotagem autônoma — o foco é o assistente.** |

---

## Referências

- **AC Shared Memory Documentation** — structs e campos
  ([`docs/ACSharedMemoryDocumentation.pdf`](docs/ACSharedMemoryDocumentation.pdf)).
- **Contrato técnico do projeto** — especificação completa de campos, schema e
  decisões arquiteturais ([`CLAUDE.md`](CLAUDE.md)).
- **Relatórios de validação do modelo** — metodologia e iterações
  ([`docs/validation_monza_model*.md`](docs/)).
- scikit-learn — *Gradient Boosting Regression* (base do `SectorModel` e dos futuros modelos especializados).

---

## Autor

**João Pedro Fachini Moreira Silva** — joaofachini01@gmail.com / jo00.silva@catolicasc.edu.br
Projeto de Portfólio · Católica SC

---

<sub>Plataforma alvo: Windows · Python 3.11+ · Não utiliza LLMs em nenhuma camada</sub>
