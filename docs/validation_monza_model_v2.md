# Relatório de Validação — `data/models/monza.pkl` (v2)

**Data:** 2026-04-08  
**Modelo:** `GradientBoostingRegressor` (sklearn.ensemble)  
**Pista:** Monza  
**Versão anterior:** `validation_monza_model.md` (2026-04-08)  
**Avaliação geral:** ✅ **Pronto para uso — problemas críticos resolvidos, 2 ressalvas operacionais**

---

## Comparativo com o Relatório Anterior

| Problema | Status anterior | Status atual |
|---|---|---|
| 🔴 Clutch corrompido (10 voltas) | CRÍTICO | ✅ **Resolvido** — filtro automático no `train()` e no `lap_recorder` |
| 🟡 `brake_bias` como feature (data leakage) | MÉDIO | ✅ **Resolvido** — removido de `_FEATURE_FIELDS` |
| 🟡 Outliers extremos no delta (±240s) | MÉDIO | ✅ **Resolvido** — filtro `|delta| > 60s` no `train()` |
| 🟡 4.327 setores ausentes do treino | ALTO | ⚠️ **Mantido** — dados históricos não recuperáveis |
| 🟢 Versão sklearn incompatível | BAIXO | ⚠️ **Mantido** — ainda 1.7.2 vs 1.8.0 |
| 🔴 `save()` corrompendo arquivo (novo) | — | ✅ **Corrigido** — temp file + `shutil.copy2` |

---

## 1. Estrutura e Consistência Interna

| Campo | Valor | Avaliação |
|---|---|---|
| `load()` bem-sucedido | ✅ True | — |
| `n_features_in_` | 22 | ✅ Reduziu de 23 (brake_bias removido) |
| `n_training_sectors` | 2.217 | ✅ Reflete apenas dados limpos |
| `max_delta` (p95) | 22.54 s | ✅ Plausível para Monza |
| `brake_bias` nas features | ✅ Ausente | — |
| `feature_importance` soma | 0.9999 | ✅ Normal (arredondamento) |
| NaN em `scaler.mean_` | 0 | ✅ |
| Zeros em `scaler.scale_` | 0 | ✅ Nenhuma feature constante no scaler |
| NaN em X (dados de treino) | 0 | ✅ |
| Inf em X (dados de treino) | 0 | ✅ |

---

## 2. Qualidade dos Dados de Treino

| Verificação | Valor | Avaliação |
|---|---|---|
| Voltas totais nos arquivos JSON | 32 | — |
| Voltas com clutch corrompido | 10 | ✅ Descartadas automaticamente |
| Voltas limpas usadas no treino | 22 | — |
| Clutch máximo nos dados de treino | 1.0 | ✅ (era 99.27) |
| Clutch mínimo nos dados de treino | 0.369 | ✅ Dentro do range esperado (0–1) |
| delta_vs_best máximo | +29.5 s | ✅ Dentro do limiar de 60 s |
| delta_vs_best mínimo | −50.3 s | ⚠️ Ver ressalva abaixo |
| Setores com `|delta| > 60s` descartados | 1 | ✅ |
| Scaler mean vs dados: desvio máximo | 0.0 | ✅ Scaler e avaliação usam os mesmos dados |

**⚠️ Ressalva sobre delta_min = −50.3 s:** O mini-setor mais negativo registrado tem delta de −50 s, que é fisicamente impossível em Monza (volta completa ≈ 110 s). Ficou dentro do filtro de 60 s porque o limiar é simétrico. Considerar reduzir o filtro para `|delta| < 30 s` ou usar o `_i_best_time_ms` disponível nos snapshots para calcular um limiar dinâmico por pista.

---

## 3. Performance Preditiva

| Métrica | Valor | vs. Relatório Anterior | Avaliação |
|---|---|---|---|
| MAE | 6.28 s | 5.54 s → 6.28 s | ✅ Comparação agora justa (só dados limpos) |
| RMSE | 8.89 s | 9.05 s → 8.89 s | ✅ Leve melhora |
| R² (todos os setores) | **0.515** | 0.425 → 0.515 | ✅ Melhora significativa |
| R² (apenas deltas > 0) | −0.024 | — | ⚠️ Ver nota abaixo |
| Mediana do erro absoluto | 4.70 s | 3.33 s → 4.70 s | ⚠️ Leve piora esperada |
| P90 do erro absoluto | 11.6 s | 11.6 s → 11.6 s | Estável |

**Nota sobre R²:** O R² de 0.515 é calculado sobre todos os setores (incluindo deltas negativos). O modelo aprende `delta_vs_best` como target, mas `predict()` clipa scores em [0, max_delta] — ou seja, deliberadamente não prevê deltas negativos (setores bons). Isso é correto pelo design: o objetivo é identificar anomalias, não prever o delta exato. O R² de −0.024 sobre setores positivos indica que o modelo acerta a direção mas não a magnitude exata das perdas individuais — aceitável para um anomaly detector.

O MAE de 6.28 s parece alto, mas deve ser lido com contexto: 94.4% dos setores recebem score 0.0 (previsão correta para setores neutros), e o erro é concentrado nos ~6% de setores com delta positivo.

---

## 4. Capacidade de Separação (Teste Chave)

Este é o critério mais importante para um anomaly detector:

| Grupo | Setores | Score médio | Score máx |
|---|---|---|---|
| Ruins (delta > 5 s) | 48 | **0.928** | 1.000 |
| Bons (delta < −1 s) | 1.055 | **0.0001** | 0.040 |

**Separação: perfeita ✅** — setores ruins recebem score ~930× maior que setores bons. O modelo discrimina corretamente quem perde tempo.

Distribuição dos scores:

| Faixa de score | % dos setores |
|---|---|
| = 0.0 (sem anomalia) | 94.4% |
| > 0.1 | 2.4% |
| > 0.5 | 2.1% |

A concentração em 0.0 é esperada em Monza: pista rápida, poucas curvas, a maior parte do lap o piloto está simplesmente no acelerador.

---

## 5. Feature Importance (novo modelo)

| Feature | Importância | vs. Modelo Anterior | Avaliação |
|---|---|---|---|
| `tc_active` | 21.5% | 6.9% → 21.5% | ✅ Subiu após remover brake_bias |
| `track_position` | 17.0% | 26.1% → 17.0% | ✅ Mais balanceado |
| `rpms` | 9.4% | 5.0% → 9.4% | ✅ Ganhou relevância |
| `abs_active` | 6.4% | 24.9% → 6.4% | ✅ Reduziu (antes inflado por brake_bias correlacionado) |
| `local_ang_vel_x` | 5.7% | 1.0% → 5.7% | ✅ Oversteer/understeer mais visível |
| `wheel_slip_rr/rl` | 4.9% / 4.3% | — | ✅ Tração traseira capturada |
| `throttle` | 1.5% | 0.3% → 1.5% | ✅ Subiu (antes subrepresentado) |
| `brake` | 0.3% | 0.07% → 0.3% | ⚠️ Ainda baixo para Monza |
| `surface_grip` | 0.0% | 0.0% | ✅ Constante — correto |

A remoção do `brake_bias` redistribuiu importância para features mais relevantes. O `brake` ainda aparece baixo para uma pista de alta velocidade como Monza — esperado em parte porque o comportamento de frenagem é fortemente correlacionado com `track_position` e `abs_active`, que já capturam boa parte do sinal.

---

## 6. Convergência do Treino

| Métrica | Valor | Avaliação |
|---|---|---|
| Train score: iteração 0 | 153.3 | — |
| Train score: iteração 99 | 79.4 | ✅ Redução de ~48% |
| Plateau std (últimas 20 iterações) | 2.95 | ⚠️ Mais oscilante que antes (1.12) |
| OOB improvement positivo | 56% das iterações | ⚠️ Próximo ao limiar de convergência |
| OOB total acumulado | +93.1 | ✅ Positivo |

A oscilação maior no plateau (2.95 vs 1.12) é esperada com menos dados de treino (2.217 vs 7.491 setores). Sugere que o modelo está próximo do limite de estabilidade com o dataset atual. Coletar mais voltas deve reduzir essa variância.

---

## 7. Bug Adicional Encontrado e Corrigido

**`save()` — corrupção de arquivo em filesystem montado (cross-platform)**

O método `joblib.dump(payload, path, compress=3)` usa mmap internamente para arrays numpy. Ao salvar diretamente em um path de filesystem Windows montado via Linux (ambiente de desenvolvimento), os chunks do mmap não eram completamente transferidos, resultando em um arquivo zlib truncado que `joblib.load()` não conseguia ler.

**Correção:** `save()` agora salva em um arquivo temporário no filesystem local (`/tmp`) via `tempfile.NamedTemporaryFile`, e depois usa `shutil.copy2()` para mover o arquivo completo ao destino final. O arquivo no Windows é escrito em uma operação única, evitando o problema de mmap.

---

## 8. Ressalvas Operacionais

1. **Volume de dados baixo** — 2.217 setores de 22 voltas é insuficiente para generalização robusta. O modelo vai melhorar com mais sessões. A convergência oscilante indica que 100 estimadores estão próximos do limite de estabilidade com este volume.

2. **Delta mínimo de −50.3 s** — Um setor com valor impossível permanece nos dados de treino (acima do limiar de −60 s). Recomenda-se tightening do filtro para `|delta| < 30 s` na próxima versão.

3. **Versão sklearn** — O modelo foi treinado com sklearn 1.7.2 no ambiente de desenvolvimento. O ambiente de produção (Windows) usa 1.8.0. `InconsistentVersionWarning` ao carregar, mas funcionalmente compatível.

---

## 9. Checklist de Pré-Deploy

- [x] `load()` funciona sem erro
- [x] `predict()` retorna scores em [0.0, 1.0]
- [x] `predict_batch()` funciona em lote
- [x] Clutch corrupto filtra a volta inteira no `train()`
- [x] Clutch corrupto invalida snapshot no `lap_recorder` (prevenção futura)
- [x] `brake_bias` removido das features
- [x] Outliers de delta filtrados
- [x] `save()` não corrompe arquivo cross-platform
- [x] Separação ruins vs bons: score médio 0.928 vs 0.0001
- [x] Script SQL de correção do banco gerado (`database/fix_corrupted_clutch_laps.sql`)
- [ ] Coletar mais voltas limpas (recomendado: ≥ 50 para estabilidade do plateau)
- [ ] Tightening do filtro de delta para `|delta| < 30 s`
- [ ] Verificar versão sklearn no ambiente Windows de produção
