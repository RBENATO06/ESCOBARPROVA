# TESTE - Q2 Wine Quality (Aluno 3)

Ambiente: Windows, Python 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4,
seaborn 0.13.2. Executado pelo auditor a partir da raiz `aluno_3/Exercicio 2_wine_quality`.

## Comandos

```
python src/main.py        # EDA + treino 3 modelos + experimento de faixas + salvar
python src/inferencia.py  # inferencia em vinhos novos (modelo principal = faixas)
```
(equivalente a `run.ps1`; o `pip install` do run.ps1 foi pulado pois o ambiente ja
tem as dependencias)

## EXIT CODES

| Etapa | Comando | EXIT CODE |
|-------|---------|-----------|
| Treino + EDA + faixas | `python src/main.py` | **0** |
| Inferencia | `python src/inferencia.py` | **0** |

## Metricas reproduzidas (conjunto de teste, 1300 amostras)

### 7 classes (selecao por F1-macro)

| Modelo | Acuracia | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| **ExtraTrees (vencedor)** | **0.6946** | **0.3987** | **0.6830** |
| MLP | 0.5623 | 0.2687 | 0.5466 |
| AdaBoost | 0.5185 | 0.2107 | 0.4921 |

Recall por classe do vencedor: 3=0.00, 4=0.14, 5=0.70, 6=0.81, 7=0.56, 8=0.36,
9=0.00 (classes 3 e 9 = limitacao dos dados, documentada).

### Experimento 7 classes vs 3 faixas (mesmo modelo, mesmo split estratificado)

| Metrica | 7 classes | 3 faixas | Ganho |
|---------|-----------|----------|-------|
| Acuracia | 0.6946 | 0.7323 | +0.0377 |
| **F1-macro** | **0.3987** | **0.7241** | **+0.3254** |
| F1-weighted | 0.6830 | 0.7320 | +0.0490 |

Impacto do agrupamento em faixas no F1-macro **quantificado: +0.3254** (quase
dobra). Modelo principal de inferencia escolhido = **3 faixas**. Numeros identicos
ao README, debate e `metricas.json`.

## Artefatos confirmados (regenerados nesta execucao)

- `matriz_confusao_7classes.png` (contagens) +
  `matriz_confusao_7classes_normalizada.png` (recall por classe) +
  `matriz_confusao_3faixas.png`
- 5 graficos EDA: `eda_distribuicao_quality.png`, `eda_quality_por_tipo.png`,
  `eda_correlacao.png`, `eda_alcohol_vs_quality.png`, `eda_correlacao_com_alvo.png`
- `comparacao_modelos.json` + `comparacao_modelos.txt`
- `metricas.json` (consolidado 7 classes + faixas, com `suporte_por_classe_teste`)
- `modelo_campeao_7classes.joblib` (~55 MB) + `modelo_campeao_faixas.joblib`
  (~44 MB), ambos com `compress=3`

## Screenshot textual da inferencia

```
======================================================================
INFERENCIA - vinhos NOVOS | modelo: faixas (ExtraTrees)
======================================================================

Vinho 1: previsao = baixa  (prob = 0.5275)
  Top-3 ranking de faixas:
    - baixa     52.75%  <== atribuido
    - media     42.00%
    - alta       5.25%

Vinho 2: previsao = media  (prob = 0.5575)
  Top-3 ranking de faixas:
    - media     55.75%  <== atribuido
    - baixa     33.00%
    - alta      11.25%

Vinho 3: previsao = alta  (prob = 0.5150)
  Top-3 ranking de faixas:
    - alta      51.50%  <== atribuido
    - media     45.25%
    - baixa      3.25%

Tabela completa (previsao + probabilidade):
  faixa_prevista  probabilidade
0          baixa         0.5275
1          media         0.5575
2           alta         0.5150
```

Inferencia funcional: usa automaticamente o modelo principal (faixas), produz
baixa/media/alta e mostra o ranking Top-N com `%` marcando a vencedora
(`<== atribuido`).

> Nota do auditor (ver AUDITORIA.md): os 3 vinhos sao FABRICADOS a mao (0
> correspondencias exatas nos `winequality-*.csv`). A "probabilidade" e a fracao
> das 400 arvores que votaram na faixa vencedora (confianca do comite, nao
> calibrada); como sao vinhos de FRONTEIRA, a certeza fica em ~0,52-0,56 —
> documentado na saida e no README. As metricas de treino permanecem identicas.

## Conclusao do teste

Reprodutibilidade **CONFIRMADA**: exit 0 nas duas etapas, todas as metricas
(7 classes e faixas) identicas ao relatado, EDA e duas matrizes de confusao
geradas, inferencia funcional com o experimento de faixas.
