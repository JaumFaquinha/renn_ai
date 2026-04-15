# Relatório de Validação — `data/models/monza.pkl` (v3)

**Data:** 2026-04-14  
**Modelo:** `GradientBoostingRegressor` (sklearn.ensemble)  
**Pista:** Monza  
**Relatório anterior:** `validation_monza_model_v2.md` (2026-04-08)  
**Avaliação geral:** 🔴 **NÃO APTO PARA PRODUÇÃO — 2 problemas críticos de design identificados**

---

## ⚠️ Alerta Principal: Modelo Retrained Sem Validação

O modelo atualmente em disco (`data/models/monza.pkl`) **não corresponde ao modelo descrito no relatório v2**. Todos os indicadores estruturais divergem significativamente:

| Métrica | Relatório v2 | Modelo atual | Status |
|---|---|---|---|
| `n_training_sectors` | 2.217 | **6.626** | ❌ DIVERGE (+199%) |
| `max_delta` (p95) | 22.54 s | **18.08 s** | ❌ DIVERGE |
| `tc_active` importance | 21.5% | **14.5%** | ❌ DIVERGE |
| `abs_active` importance | 6.4% | **22.1%** | ❌ DIVERGE |
| `train_score_[0]` | 153.3 | **34.6** | ❌ DIVERGE |
| `train_score_[99]` | 79.4 | **20.3** | ❌ DIVERGE |
| Plateau std (últimas 20 iter) | 2.95 | **0.49** | ❌ DIVERGE |

**Conclusão:** O modelo foi retreinado após a publicação do v2, provavelmente com mais voltas (6.626 vs 2.217 setores). O relatório v2 é obsoleto e não descreve o modelo em produção. **Esta validação descreve o modelo real.**

---

## 1. Estrutura e Consistência Interna

| Campo | Valor | Avaliação |
|---|---|---|
| `load()` sem erro | ✅ True | — |
| `n_features_in_` | 22 | ✅ Consistente com código (`_FEATURE_FIELDS`) |
| `feature_fields` pkl == código | ✅ Idênticos | Sem drift de interface |
| `n_training_sectors` | 6.626 | ✅ Melhora vs v2 (+199%) |
| `max_delta` (p95) | 18.08 s | ✅ Plausível para Monza |
| `brake_bias` nas features | ✅ Ausente | Leakage corrigido |
| `feature_importance` soma | 1.0000 | ✅ |
| NaN em `scaler.mean_` | 0 | ✅ |
| NaN em `scaler.scale_` | 0 | ✅ |
| Zeros em `scaler.scale_` | 0 | ✅ |
| `track_id` | "monza" | ✅ |
| Versão sklearn (treino) | 1.8.0 | ⚠️ Ver ressalva §8 |

---

## 2. Feature Importance (Modelo Atual)

| Feature | Importância | Avaliação |
|---|---|---|
| `abs_active` | **22.1%** | ⚠️ Principal feature — ver análise §4 |
| `track_position` | **20.9%** | 🔴 Alta demais — ver §3 (leakage posicional) |
| `tc_active` | **14.5%** | ⚠️ Comportamento invertido — ver §4 |
| `wheel_slip_rl` | 6.5% | ⚠️ Comportamento invertido — ver §4 |
| `rpms` | 5.6% | ✅ |
| `gforce_z` | 4.4% | ✅ |
| `wheel_slip_rr` | 4.3% | ⚠️ Comportamento invertido — ver §4 |
| `local_ang_vel_z` | 4.0% | ✅ |
| `surface_grip` | **0.0%** | ⚠️ Feature morta — constante nos dados |
| `clutch` | 0.02% | ✅ Irrelevante (esperado) |

---

## 3. 🔴 Problema Crítico 1: Target Leakage por Acumulação do Delta

### Descrição do Problema

O campo `delta_vs_best` (derivado de `performanceMeter` do AC) é uma métrica **acumulativa** ao longo da volta. Ele representa o delta total acumulado desde o início do lap, não a perda de tempo ocorrida especificamente no mini-setor atual.

**Consequência:** O modelo aprende onde na pista o delta *tipicamente é alto* ao final de uma volta, não o que o piloto fez de errado no setor específico. Um piloto que perde 1.0s na Prima Variante carregará esse delta acumulado por toda a volta — e todos os setores subsequentes terão `delta_vs_best` alto, mesmo que a execução neles seja perfeita.

### Evidência Empírica

Teste com setor "neutro" (throttle=0.9, sem ABS, sem TC, sem wheel slip, inputs perfeitos) variando apenas `track_position`:

| Posição | Zona | Score (modelo) | Interpretação correta |
|---|---|---|---|
| 0.05 | Reta principal | **0.69** | Deveria ser ≈ 0.0 |
| 0.11 | Prima Variante | **0.69** | Deveria ser ≈ 0.0 |
| 0.63 | Variante Ascari | **0.77** | Deveria ser ≈ 0.0 |
| 0.80 | Entrada Parabolica | **1.00** ❌ | Deveria ser ≈ 0.0 |
| 0.95 | Reta final | **0.97** ❌ | Deveria ser ≈ 0.0 |

**Um piloto executando uma saída perfeita da Parabolica recebe score máximo de anomalia.** O modelo é inutilizável como detector de falhas de pilotagem.

### Confirmação Estatística

Correlação de Spearman entre `track_position` e score de setores neutros:

```
ρ = 0.673  (p = 0.0008)
```

Correlação positiva e estatisticamente significativa entre posição na pista e score. Isso confirma que o modelo está capturando a progressão do delta acumulado, não a qualidade da pilotagem local.

### Causa Raiz

O `sector_aggregator.py` grava o `delta_vs_best` instantâneo (snapshot do `performanceMeter`) em cada mini-setor. O target correto para um modelo **por setor** seria a **derivada** do delta entre setores consecutivos — a variação `Δ(delta)` ocorrida dentro do mini-setor.

**Target errado (atual):** `delta_vs_best[setor_N]` = delta acumulado desde o início  
**Target correto:** `delta_vs_best[setor_N] - delta_vs_best[setor_N-1]` = perda no setor específico

### Correção Necessária

1. No `sector_aggregator.py`: calcular e gravar `delta_per_sector` (variação do delta dentro do mini-setor)
2. Em `sector_model.py`: substituir `_TARGET_FIELD = "delta_vs_best"` por `_TARGET_FIELD = "delta_per_sector"`
3. Retreinar o modelo com o target corrigido
4. Não requer nova coleta de dados — os JSONs existentes têm `track_position` e `delta_vs_best` para recalcular

---

## 4. 🔴 Problema Crítico 2: Inversão de Monotonicidade em Features de Tração

### Descrição do Problema

O modelo aprendeu relações **inversas** às fisicamente corretas para `tc_active` e `wheel_slip`:

| Feature | Comportamento esperado | Comportamento do modelo | Status |
|---|---|---|---|
| `tc_active` 0→1 | Score aumenta (TC corta potência = perda) | Score **diminui** −0.59 | 🔴 INVERTIDO |
| `wheel_slip_rl` 0→0.5 | Score aumenta (roda girando = perda) | Score **diminui** −0.16 | 🔴 INVERTIDO |
| `abs_active` 0→1 | Score aumenta (ABS = freio bloqueando) | Score **aumenta** +0.25 | ✅ Correto |

### Causa Raiz

Mesma causa do Problema 1: como o target é cumulativo, os setores onde o TC foi ativado (saídas de curva onde o piloto tentou acelerar) frequentemente ocorrem **antes** do ponto de máxima perda acumulada na volta. Os setores com TC ativo no meio da volta têm `delta_vs_best` **menor** que os setores neutros no final (onde o delta acumulado já atingiu seu máximo). O modelo aprende que "TC ativo → delta menor" — o inverso da relação causal real.

**Evidência direta:**

```
Saída da Roggia (pos=0.27) com:
  tc_active=0.0  → score = 0.694  (sem TC, piloto escorregando silenciosamente)
  tc_active=1.0  → score = 0.410  (TC ativo, piloto com tração cortada)
```

O modelo penaliza mais quem não ativa o TC do que quem ativa — indicação clara de que o sinal causal foi invertido pelo target acumulado.

---

## 5. Positivos do Modelo Atual

Apesar dos problemas críticos de design, algumas propriedades estruturais melhoraram em relação ao v2:

| Aspecto | Avaliação |
|---|---|
| Volume de dados | ✅ 6.626 setores (vs 2.217 no v2) — melhora de 199% |
| Convergência do treino | ✅ Plateau std = 0.49 (vs 2.95 no v2) — muito mais estável |
| `abs_active` monotonicamente correto | ✅ Comportamento físico preservado |
| Separação extremos (setor catastrófico ABS) | ✅ Score = 1.0 para frenagem com bloqueio total |
| Feature_fields sincronizados com o código | ✅ Sem drift de interface |
| `brake_bias` removido | ✅ Leakage por sessão corrigido |
| Clutch corrompido filtrado | ✅ Filtro automático funcionando |
| Arquivo íntegro (não corrompido) | ✅ |
| `surface_grip` importance = 0.0 | ✅ Feature constante corretamente ignorada |

---

## 6. Ressalvas Operacionais (Herdadas do v2)

1. **Delta mínimo de −50.3 s** — permanece nos dados. O filtro `|delta| > 60s` não captura esse outlier. Recomendado: `|delta| < 30s`.

2. **`surface_grip` como feature viva** — importance=0.0 mas ocupa posição no vetor de features. Não quebra nada, mas gasta capacidade do scaler. Candidata a remoção na próxima versão (requer retreino).

3. **Versão sklearn** — modelo treinado em sklearn 1.8.0; ambiente de validação usa 1.7.2. `InconsistentVersionWarning` no carregamento. Funcionalmente compatível até agora, mas monitorar incompatibilidades ao atualizar o ambiente de produção.

---

## 7. Plano de Correção Recomendado

### Prioridade 1 — Corrigir o target (impacto alto, esforço médio)

**Problema:** Target cumulativo contamina todo o modelo.

**Ação:**
1. Adicionar campo `delta_per_sector` no `SectorAggregator` — computar como `delta_vs_best[i] - delta_vs_best[i-1]` entre snapshots consecutivos no mini-setor
2. Atualizar `_TARGET_FIELD` em `sector_model.py` de `"delta_vs_best"` para `"delta_per_sector"`
3. Atualizar schema §4.5 no `CLAUDE.md`
4. Recalcular `delta_per_sector` retroativamente nos JSONs existentes (os dados têm `delta_vs_best` por setor, basta diferenciar)
5. Retreinar e re-validar

**Nota:** Esta mudança tornará o modelo fisicamente correto. O target por setor será geralmente pequeno (0.01–0.3s de perda em um mini-setor de 1% da pista), o que pode demandar ajuste nos hiperparâmetros do GBR (learning_rate menor ou mais estimadores).

### Prioridade 2 — Remover `surface_grip` (impacto baixo, esforço baixo)

**Ação:** Remover `"surface_grip"` de `_FEATURE_FIELDS` em `sector_model.py`. Requer retreino, mas sem coleta de novos dados.

### Prioridade 3 — Tightening do filtro de outliers (impacto médio, esforço baixo)

**Ação:** Alterar `_DELTA_OUTLIER_THRESHOLD_S` de 60.0 para 30.0 em `sector_model.py`. Isso elimina o delta de −50.3s que permanece nos dados.

---

## 8. Checklist de Pré-Deploy (Estado Atual)

- [x] `load()` funciona sem erro
- [x] `predict()` retorna valores em [0.0, 1.0]
- [x] `predict_batch()` funciona em lote
- [x] `feature_fields` no pkl == `_FEATURE_FIELDS` no código
- [x] `brake_bias` ausente das features
- [x] Clutch corrompido filtrado no treino
- [x] Arquivo não corrompido (sem truncamento cross-platform)
- [ ] **Score de setores neutros próximo de 0.0** ← FALHOU (score 0.69–1.00)
- [ ] **Monotonicidade tc_active** ← FALHOU (invertida)
- [ ] **Monotonicidade wheel_slip** ← FALHOU (invertida)
- [ ] Relatório de validação atualizado antes do deploy ← ESTE DOCUMENTO
- [ ] Coletar mais voltas limpas (recomendado: ≥ 50)
- [ ] Tightening filtro de delta para `|delta| < 30s`
- [ ] Verificar versão sklearn no ambiente Windows de produção

---

## 9. Conclusão

O modelo `monza.pkl` atual é o mais robusto estruturalmente desde o início do projeto: mais dados, melhor convergência, sem leakage de `brake_bias`, sem clutch corrompido. **Porém, contém um defeito de design fundamental no target** que compromete toda a capacidade de detecção por setor.

A ironia é que o `abs_active` funciona corretamente (22.1% de importância, monotonicamente correto) porque o ABS tende a ser ativado em zonas de frenagem onde o delta acumulado é alto — o sinal cumulativo e o sinal causal coincidem acidentalmente. Para `tc_active` e `wheel_slip`, os sinais divergem, revelando o defeito.

O modelo pode ser usado como **indicador de zona da pista com delta histórico alto** (o que detecta, mas não o que foi projetado para fazer). Para uso como **anomaly detector de inputs do piloto**, o target precisa ser corrigido para `delta_per_sector`.

*Validação conduzida em: 2026-04-14*  
*Ambiente: Python 3.10, sklearn 1.7.2, Ubuntu Linux (sandbox)*  
*Modelo treinado em: sklearn 1.8.0, Windows*
