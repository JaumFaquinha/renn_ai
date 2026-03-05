# Engenheiro de Corrida IA

Analisador de telemetria em tempo real para Assetto Corsa que atua como engenheiro de corrida, identificando onde o piloto humano está perdendo tempo durante voltas.

## Requisitos

- Python 3.10+
- Windows (Shared Memory do AC é exclusiva do Windows)
- Assetto Corsa instalado e rodando

## Instalação

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Uso

```bash
python scripts/run_session.py
```

## Arquitetura

Veja `CLAUDE.md` para especificação técnica completa.

### Pipeline

```
Shared Memory (AC) → SharedMemoryReader → LapRecorder → LapAnalyzer → PatternDetector → ReportBuilder
```

### Fases

| Fase | Status | Descrição |
|------|--------|-----------|
| 1 | ✅ Implementada | Shared Memory Reader |
| 2 | 🔧 Scaffold | Lap Recorder |
| 3 | 🔧 Scaffold | Lap Analyzer (Delta vs Best) |
| 4 | 🔧 Scaffold | Pattern Detector |
| 5 | 🔧 Scaffold | Report Builder + Console Output |
| 6 | ⏳ Backlog | TTS Integration |

## Estrutura

```
renn_ai/
├── config/          # Configurações globais e mapas de pista
├── src/
│   ├── memory/      # Leitura da Shared Memory via mmap
│   ├── recording/   # Gravação de voltas por mini-setor
│   ├── analysis/    # Análise de delta e detecção de padrões
│   ├── models/      # Modelos ML por pista
│   └── output/      # Console e TTS
├── data/laps/       # Voltas gravadas em JSON
├── tests/           # Testes unitários + fixtures
└── scripts/         # Entrypoints
```
