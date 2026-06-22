# MEMORY.md — Engenheiro de Corrida IA

> Arquivo de memória de sessão. Atualizado automaticamente ao fim de cada sessão de trabalho.
> Leia este arquivo junto com `.claude/CLAUDE.md` para contexto completo.

---

## Estado Atual do Projeto

**Data da última atualização:** 2026-03-18
**Fase ativa:** Todas as fases (1–7) implementadas. Projeto funcional.
**Próximo passo:** Rodar sessão real no AC para coletar voltas e treinar o SectorModel.

---

## Status das Fases

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Shared Memory Reader (ctypes + mmap) | ✅ Completo |
| 2 | Lap Recorder (mini-setores + JSON) | ✅ Completo |
| 3 | Lap Analyzer (delta vs best) | ✅ Completo |
| 4 | Pattern Detector (5 padrões heurísticos) | ✅ Completo |
| 5 | Report Builder + Console Output | ✅ Completo |
| 6 | TTS Integration | ✅ Placeholder funcional (`src/output/tts_integration.py`) |
| 7 | Supabase Persistence (LapUploader + QueryService) | ✅ Completo |
| ML | SectorModel (GradientBoostingRegressor por pista) | ✅ Completo |

---

## Módulos Implementados

### `src/memory/`
- `shared_memory_reader.py` — leitura via mmap, 20-50Hz, `snapshot_to_dict()`
- `physics_page.py` — ctypes struct SPageFilePhysics (completo conforme PDF)
- `graphics_page.py` — ctypes struct SPageFileGraphic (completo conforme PDF)
- `static_page.py` — ctypes struct SPageFileStatic (completo conforme PDF)

### `src/recording/`
- `lap_recorder.py` — mini-setores de 0.01 na spline, `_current_tyre_compound`, descarte de voltas inválidas
- `sector_aggregator.py` — sumarização e agrupamento por setor

### `src/analysis/`
- `lap_analyzer.py` — delta vs best por mini-setor
- `pattern_detector.py` — 5 padrões (frenagem tardia, aceleração precoce, entrada rápida, troca subótima, saída comprometida), confiança como float
- `report_builder.py` — `SectorReport` dataclass com `model_score: Optional[float]`, método `build(..., model_scores=...)`

### `src/models/`
- `sector_model.py` — GradientBoostingRegressor (scikit-learn), 23 features do schema §4.5, normalização StandardScaler, anomaly score 0.0–1.0, save/load via joblib

### `src/output/`
- `console_reporter.py` — exibe relatório ordenado por delta; linha de score do modelo IA com barra visual
- `tts_integration.py` — placeholder (Fase 6)

### `src/persistence/`
- `lap_uploader.py` — fila async + retry, grava `tyre_compound` na tabela `laps`
- `query_service.py` — personal bests, histórico de setores (batch)
- `supabase_client.py` — cliente Supabase configurável via `.env`

### `scripts/`
- `run_session.py` — entrypoint principal; carrega SectorModel por pista, predição em lote por volta
- `train_model.py` — treino offline via JSONs e/ou Supabase; métricas MAE/R²/Pearson/Precision@10%
- `map_track.py` — mapeamento manual de curvas por pista

### `tests/`
- `test_memory_reader.py` — offline via fixtures JSON
- `test_lap_recorder.py`
- `test_pattern_detector.py`
- `test_query_service.py`
- `test_lap_uploader.py`
- `test_supabase_client.py`
- `test_sector_model.py` — **36 testes em 8 classes** (fail-safe, treino, score ordering, predict_batch, save/load, métricas evaluate_model, robustez)

---

## Mudanças Recentes (Sessão 2026-03-10)

### 1. Pipeline de `tyreCompound` (correção de gap)
- `shared_memory_reader.py` → adicionado `"_tyre_compound": g.tyreCompound` em `snapshot_to_dict()`
- `lap_recorder.py` → campo `_current_tyre_compound`, atualizado por snapshot, gravado em `lap_data["tyre_compound"]`
- `lap_uploader.py` → `"tyre_compound": lap_data.get("tyre_compound", "unknown")` no `lap_row`
- `database/schema.sql` → coluna `tyre_compound TEXT` na tabela `laps` + migration idempotente

### 2. `session_type` dinâmico (correção de bug)
- `run_session.py` → `_SESSION_TYPE_MAP` + `_session_int_to_str()`, leitura via `graphics.session`
- `first_snap` movido para antes da criação do recorder para capturar `session_type` e `airTemp`
- **Bug corrigido:** `airTemp` era lido de `first_snap.airTemp` (errado) → corrigido para `first_snap.physics.airTemp`

### 3. CLAUDE.md atualizado
- Seção 4.1: `tyreCompound` adicionado ao grupo "Contexto de Sessão"
- Seção 4.3: `aidAutoClutch`, `aidStability`, `drsEnabled`, `tyreRadius[4]` documentados; `session` com referência a `_session_int_to_str()`
- Seção 4.5: notas sobre `speed_min` (campo computado) e `tyre_compound` (metadado de volta, não de mini-setor)
- Seção 4.6: ~25 campos anteriormente não classificados adicionados com justificativas

### 4. SectorModel implementado (antigo placeholder)
- `src/models/sector_model.py` → implementação completa substituiu scaffold vazio
- `requirements.txt` → `scikit-learn>=1.3.0` adicionado
- `scripts/train_model.py` → novo script de treino offline
- `src/analysis/report_builder.py` → `model_score` em `SectorReport`, parâmetro `model_scores` em `build()`
- `src/output/console_reporter.py` → linha de score IA com barra visual `[████████░░]`
- `scripts/run_session.py` → carga do modelo por pista + predição em lote por volta

### 5. Testes do SectorModel
- `tests/test_sector_model.py` → 36 testes cobrindo todas as garantias do modelo

---

## Mudanças Recentes (Sessão 2026-03-18)

### 1. Migração Supabase aplicada (erro 400 resolvido)
- `database/migration_fix_400.sql` foi executado com sucesso no Supabase Studio
- Colunas `drs_active` e `drs_available` corrigidas de SMALLINT para INT
- Coluna `tyre_compound TEXT` adicionada à tabela `laps`
- Upload de voltas ao Supabase está operacional

### 2. Validação completa do modelo de IA
- **37 testes** executados e aprovados: 12 SectorModel (fail-safe) + 17 PatternDetector + 8 pipeline
- **Sem target leakage:** `delta_vs_best` não está em `_FEATURE_FIELDS` ✓
- **Sem campos fora do schema:** todas as 23 features existem no schema §4.5 ✓
- **Supabase SELECT alinhado:** `load_laps_from_supabase()` busca exatamente os 23 campos necessários ✓
- **PatternDetector:** todos os 5 padrões do CLAUDE.md §5 validados com casos positivos e negativos ✓
- **Pipeline end-to-end:** PatternDetector → ReportBuilder → LapReport com model_score propagado ✓

### 3. Pontos técnicos identificados (não críticos, monitorar)
- `evaluate_model()` acessa `model._max_delta` diretamente (atributo privado) — sugestão futura: expor como `@property max_delta`
- Avaliação de métricas é in-sample (R² e MAE otimistas); Correlação de Pearson é a métrica mais confiável
- Deduplicação de voltas por `(lap_number, lap_time_ms)` — improvável mas poderia colidir; hash de setor seria mais robusto
- Padrão 5 (saída comprometida) exige `speed > 150 km/h` — pode não disparar em pistas muito técnicas

### 4. Thresholds do PatternDetector validados como tecnicamente corretos
- `WHEEL_SLIP_THRESHOLD = 0.15` (15% slip traseiro) — limiar realista para GT3/Sport
- `GFORCE_LATERAL_THRESHOLD = 2.5G` — correto para GT3, conservador para F1
- `RPM_SHIFT_MARGIN = 5%` → shift point em 95% do maxRpm — dentro da janela de peak power

### 5. Track map de Monza criado
- `config/track_maps/monza.json` — 7 curvas, 3 setores, layout GP
- Posições calculadas a partir de distâncias reais (5793m total, 1194m reta principal)
- Integração com ReportBuilder validada: `corner_name` retornou `Variante del Rettifilo` corretamente

### 6. Limpeza do banco de dados
- `database/cleanup_all_data.sql` criado — `TRUNCATE ... CASCADE` nas 5 tabelas
- Executado pelo usuário no Supabase Studio
- Banco zerado, schema e RLS intactos

### 7. Saúde geral do projeto (validação 2026-03-18)
- 36 arquivos Python, 0 erros de sintaxe
- 12 diretórios obrigatórios presentes
- 14 variáveis no `.env`
- 7 arquivos de teste, 37 casos aprovados

---

## Decisões Técnicas Registradas

| Decisão | Escolha | Sessão |
|---------|---------|--------|
| Algoritmo ML para SectorModel | GradientBoostingRegressor (scikit-learn) | 2026-03-10 |
| Placement de `tyre_compound` | Metadado de volta em `laps` (não mini-setor) | 2026-03-10 |
| Integração do score ML | Parâmetro opcional `model_scores` no `ReportBuilder` | 2026-03-10 |
| Eficiência de predição | `predict_batch()` — 1 chamada numpy por volta (~100 setores) | 2026-03-10 |
| Normalização do anomaly score | p95 dos deltas positivos como `_max_delta` (robusto a outliers) | 2026-03-18 |
| Avaliação de qualidade do modelo | Correlação de Pearson como métrica principal (in-sample menos enviesada que R²) | 2026-03-18 |
| Threshold de shift point | 95% do maxRpm como ponto ideal de troca — configurável via `RPM_SHIFT_MARGIN` | 2026-03-18 |
| Limpeza do banco | `TRUNCATE ... CASCADE` preserva schema + RLS, reseta apenas dados | 2026-03-18 |

> ADRs completos em `docs/decisions/` (a serem criados conforme CLAUDE.md §7).

---

## Backlog / Próximos Passos

1. **Coletar voltas reais no AC** — banco zerado e pronto; rodar `scripts/run_session.py` para gravar voltas limpas
2. **Treinar o SectorModel** — após 5+ voltas: `python scripts/train_model.py --track monza --verbose`
3. **Validar spline de Monza em sessão real** — rodar `map_track.py --track monza --resume` para afinar posições
4. **Refinar Padrão 5** — avaliar threshold de velocidade (`> 150 km/h`) após dados reais de pistas técnicas
5. **Expor `max_delta` como `@property`** no SectorModel — desacoplar `evaluate_model()` de atributo privado
6. **Criar ADRs** em `docs/decisions/` para as decisões das sessões 2026-03-10 e 2026-03-18
7. **Fase 6 real** — integrar ElevenLabs ou Azure TTS em `src/output/tts_integration.py`
8. **Modelos especializados (futuro)** — novos modelos por aspecto do engenheiro (desgaste de pneus, combustível, estratégia). NÃO é piloto autônomo: o projeto é um assistente que lê os dados do jogo continuamente. (Visão revista 2026-06-21; RL/Gymnasium/SB3 descartados)

---

## Notas de Ambiente

- **Plataforma alvo:** Windows (Shared Memory do AC é exclusiva do Windows)
- **Python:** 3.10+ (venv Windows)
- **scikit-learn** deve ser instalado manualmente na máquina Windows: `pip install scikit-learn>=1.3.0`
- **Linux VM (este ambiente):** sem acesso a pip externo (proxy 403); usar para edição de código e testes offline apenas
- **Testes sem sklearn:** 13 testes de `TestUntrained`, `TestTrainInsufficient` e `TestSaveLoad` (arquivo não existente) passam no Linux VM via `python3`
- **Testes completos:** executar `py -3.11 -m pytest tests/ -v` na máquina Windows
- [supabase-query-limits](memory/supabase-query-limits.md) — voltas em Supabase + JSON local; limites do PostgREST (1000 linhas, URL longa) truncavam consultas; clone novo tem `data/laps/` vazio
