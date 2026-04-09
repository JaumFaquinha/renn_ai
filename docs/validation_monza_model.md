# Relatório de Validação — `data/models/monza.pkl`

**Data:** 2026-04-08  
**Modelo:** `GradientBoostingRegressor` (sklearn.ensemble)  
**Pista:** Monza  
**Avaliação geral:** ⚠️ **Compartilhar com ressalvas — 3 problemas identificados, sendo 1 crítico**

---

## 1. Estrutura do Arquivo

| Campo | Valor |
|---|---|
| Formato | zlib + joblib (compress=3) |
| Tamanho comprimido | 64 KB |
| Tamanho descomprimido | ~185 KB |
| Protocolo pickle | 4 |
| sklearn salvo com | 1.8.0 |
| sklearn no ambiente de leitura | 1.7.2 |

> **Nota de carregamento:** O arquivo **não pode ser aberto diretamente com `pickle.load()`**. É necessário descomprimir primeiro com `zlib.decompress()` e depois usar `joblib.load(io.BytesIO(data))`. O método `SectorModel.load()` usa `joblib.load(path)` diretamente, o que não funciona porque o arquivo foi salvo com `compress=3` (zlib nativo do joblib, não o padrão). Verificar se o ambiente Windows tem a mesma versão de sklearn/joblib usada para salvar.

---

## 2. Parâmetros do Modelo

| Parâmetro | Valor | Avaliação |
|---|---|---|
| Algoritmo | GradientBoostingRegressor | Adequado para o problema |
| n_estimators | 100 | Conservador — pode ser insuficiente para generalizar em todas as zonas |
| max_depth | 4 | Razoável |
| learning_rate | 0.1 | Padrão |
| subsample | 0.8 | Correto — reduz overfitting |
| min_samples_leaf | 5 | Correto |
| random_state | 42 | Reprodutível |
| n_features_in_ | 23 | ✅ Consistente com feature_fields e scaler |
| n_training_sectors | 7.491 | ✅ Acima do mínimo (200) |

---

## 3. Checklist de Qualidade

### Consistência interna

- [x] `feature_fields` (23) == `scaler.n_features_in_` (23) == `model.n_features_in_` (23) == `feature_importance` (23 chaves)
- [x] `feature_importance` soma ≈ 1.0 (1.0002 — diferença de arredondamento insignificante)
- [x] `max_delta` = 23.683 s = p95 dos deltas positivos do treino ✅
- [x] `track_id` = "monza" ✅
- [x] `surface_grip` constante (mean=1.0, std=0.0, importance=0.0) — esperado para Monza indoor

### Convergência do treino

- [x] Train score decresce de 96.0 → 50.9 — convergência confirmada
- [x] OOB improvement total positivo (+29.38)
- ⚠️ Train score **não é monotônico** nos últimos estágios (std=1.12 nas últimas 20 iterações) — oscilação leve, indica que 100 estimadores chegam perto mas não no mínimo. Não bloqueante.
- ⚠️ OOB improvement positivo em apenas 55/100 iterações — o modelo oscila na convergência

### Qualidade das predições

| Métrica | Valor | Interpretação |
|---|---|---|
| MAE | 5.54 s | Alto — ~5 segundos de erro médio por mini-setor |
| RMSE | 9.05 s | Alto — dominado por outliers extremos |
| R² | 0.43 | Moderado — modelo explica 43% da variância do delta |
| Mediana do erro absoluto | 3.33 s | Mais representativa que MAE — metade dos setores tem erro < 3.3 s |
| P90 do erro | 11.6 s | 10% dos setores têm erro > 11.6 s |

> O score de anomalia (0.0–1.0) tem distribuição fortemente assimétrica: 97.7% dos setores recebem score ≤ 0.10. Apenas 2.3% dos setores são flagrados como problemáticos. Em Monza (pista rápida com poucas curvas), isso pode ser comportamento esperado — mas se o modelo raramente dispara, vale questionar se os limiares estão bem calibrados.

---

## 4. Problemas Identificados

### 🔴 CRÍTICO — Clutch corrompido em 10/32 voltas (30% dos dados de treinamento)

**O que acontece:** Os primeiros 10 arquivos de volta (`monza_1772752940` a `monza_1773095492`) têm o campo `clutch` com valores entre 57–99, quando o esperado pelo schema do CLAUDE.md é 0.0–1.0.

**Evidência:**
- Voltas antigas: `clutch` ∈ [57.95, 99.27] — 946 setores (30% dos 3.164 disponíveis)
- Voltas recentes: `clutch` ∈ [0.0, 1.0]
- O valor parece ser temperatura de pneu ou outro campo de escala percentual gravado erroneamente no campo `clutch`

**Impacto no modelo:** O scaler foi treinado com `clutch` tendo média 10.31 e desvio padrão 24.63 — uma escala completamente diferente do esperado (média ~0.0 para carro automático, ou ~1.0 para clutch sempre solto). Isso contamina o StandardScaler e reduz a capacidade preditiva do modelo para qualquer sessão futura onde `clutch` esteja correto (0–1).

**Ação requerida:** Identificar a causa raiz (bug no `shared_memory_reader.py` ou `sector_aggregator.py`) e retreinar o modelo excluindo as 10 voltas corrompidas ou corrigindo o campo.

---

### 🟡 ALTO — Dataset de treinamento incompleto (4.327 setores ausentes)

**O que acontece:** O modelo registra `n_training_sectors = 7.491`, mas os arquivos em `data/laps/` somam apenas 3.164 setores. Faltam 4.327 setores (57% do dataset de treino) nos arquivos locais.

**Impacto:** Não é possível auditar ou reproduzir o treinamento completo com os dados disponíveis. Se precisar retreinar após corrigir o bug do clutch, haverá perda de dados históricos.

**Ação requerida:** Verificar se as sessões ausentes estão em outra máquina/backup, ou se foram gravadas e depois deletadas. Consolidar todos os arquivos de volta antes do próximo retreinamento.

---

### 🟡 MÉDIO — Outliers extremos no target (`delta_vs_best`)

**O que acontece:** Dois arquivos de volta têm setores com `delta_vs_best` fora do range razoável para Monza:
- `monza_1773100439_lap002.json`: 9 setores com delta ≈ –50.3 s (improvável — o carro não ganha 50s em um mini-setor)
- `monza_1773180622_lap002.json`: 1 setor com delta = +240.5 s (fisicamente impossível em uma volta de ~110 s)

**Impacto:** Esses outliers provavelmente indicam voltas que cruzaram a linha de chegada com o jogo ainda computando o `performanceMeter` de uma sessão anterior, ou bugs na detecção de volta. Com `max_delta` baseado no p95 dos deltas positivos (23.68 s), o modelo já tem alguma proteção, mas o RMSE alto (9.05 s) é parcialmente causado por esses pontos.

**Ação requerida:** Adicionar filtro em `lap_recorder.py` para descartar setores com `|delta_vs_best| > max_lap_time_estimate` (estimativa: 130 s para Monza). Ou usar o `_i_best_time_ms` disponível nos snapshots para validar.

---

### 🟢 BAIXO — Versão de sklearn incompatível (1.8.0 → 1.7.2)

O modelo foi salvo com sklearn 1.8.0 mas a leitura usa 1.7.2. O carregamento funciona com `InconsistentVersionWarning`, mas não é garantido em produção.

**Ação requerida:** Fixar `scikit-learn==1.8.0` no `requirements.txt` ou retreinar com a versão instalada no ambiente de produção Windows.

---

## 5. Feature Importance — Análise

| Feature | Importância | Interpretação |
|---|---|---|
| `track_position` | 26.1% | ✅ Correto — o comportamento esperado varia completamente por zona (frenagem, curva, aceleração) |
| `abs_active` | 25.0% | ⚠️ Muito alta — pode indicar que o modelo aprendeu que "ABS ativo = perda" sem discriminar causa |
| `brake_bias` | 13.5% | ⚠️ Suspeito — `brake_bias` é uma constante de setup por sessão, não varia por mini-setor. Alta importância pode ser artefato de correlação com sessão |
| `tc_active` | 6.9% | Razoável |
| `local_ang_vel_z` | 5.9% | ✅ Correto — detecta oversteer/understeer |
| `rpms` | 5.0% | ✅ Correto — ponto de troca de marcha |
| `throttle` | 0.3% | Surpreendentemente baixo para um modelo de perda de tempo |
| `brake` | 0.07% | ⚠️ Muito baixo — contradiz a hipótese de que frenagem tardia é a principal causa de perda em Monza |
| `surface_grip` | 0.0% | ✅ Esperado — constante na sessão |

> A alta importância do `brake_bias` (constante de setup) sugere que o modelo pode estar capturando variação entre **sessões** em vez de variação por **mini-setor**. Isso é um risco de data leakage por correlação sessão–condição.

---

## 6. Verificação de Reprodutibilidade

O modelo pode ser carregado com:

```python
import joblib, zlib, io, warnings

with open('data/models/monza.pkl', 'rb') as f:
    raw = f.read()

data = zlib.decompress(raw)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pkg = joblib.load(io.BytesIO(data))

model = pkg['model']        # GradientBoostingRegressor
scaler = pkg['scaler']      # StandardScaler
feature_fields = pkg['feature_fields']  # lista de 23 features
max_delta = pkg['max_delta']            # 23.683 s
```

> **Atenção:** O método `SectorModel.load(path)` chama `joblib.load(path)` sem descomprimir. Verificar se joblib ≥ 3.x lida com essa compressão automaticamente — se não, `load()` vai falhar silenciosamente (retorna False por exceção interna) sem registrar o erro real.

---

## 7. Ações Prioritárias

1. **[CRÍTICO]** Investigar e corrigir o bug do campo `clutch` no pipeline de gravação. Retreinar após exclusão das 10 voltas corrompidas.
2. **[ALTO]** Localizar os 4.327 setores ausentes do dataset de treino para auditoria completa.
3. **[MÉDIO]** Adicionar filtro de outliers extremos em `delta_vs_best` antes do treino (sugestão: `|delta| < 130 s`).
4. **[MÉDIO]** Investigar a alta importância do `brake_bias` — considerar remover da feature list se confirmado como constante de sessão.
5. **[BAIXO]** Fixar `scikit-learn>=1.8.0` no `requirements.txt` e validar o método `SectorModel.load()` com arquivo comprimido.
