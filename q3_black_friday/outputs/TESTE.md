# TESTE — Q3 Black Friday (Projeto C / aluno_3)

Ambiente: Windows 10, Python 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, matplotlib 3.11.0, seaborn 0.13.2. Sem venv.

## Comandos executados

```
python src/main.py          # dados + EDA + treino multi-saída + avaliação + artefatos
python src/inferencia.py    # inferência separada (3 vendas de exemplo)
```

(equivale a `run.ps1` / `run.sh`, inspecionados: rodam `main.py` depois `inferencia.py`).

## EXIT CODES

| Etapa | Comando | EXIT CODE |
|-------|---------|-----------|
| Treino/EDA | `python src/main.py` | **0** |
| Inferência | `python src/inferencia.py` | **0** |

## Reprodutibilidade

Execução do zero (dataset sintético regerado, seed 42). Métricas idênticas às do `metricas.json` versionado.

## Métricas por alvo (split estratificado por age_group: 3000 treino / 1000 teste)

| Alvo | Acurácia global | Acaso (uniforme) | F1 macro | F1 ponderado | Certeza média |
|------|-----------------|------------------|----------|--------------|---------------|
| product_category | 0,373 | 0,143 | 0,2025 | 0,2867 | 0,3453 |
| payment_method   | 0,388 | 0,200 | 0,2968 | 0,3320 | 0,3648 |
| age_group        | 0,405 | 0,143 | 0,3385 | 0,3672 | 0,3350 |

Acima do acaso. Sensibilidade/especificidade por classe (one-vs-rest, com TP/FN/FP/TN explícitos) em `metricas.json` — recalculadas de forma independente: **0 divergências**.

Exemplos (one-vs-rest):
- product `Eletronicos`: sens=0,766 / espec=0,607; `Beleza`: sens=0,629 / espec=0,743; `Esportes`/`Roupas`: sens=0,00 / espec=1,00 (colapsadas)
- payment `PIX`: sens=0,676 / espec=0,601; `Cartao_Debito`: sens=0,00 / espec=1,00 (colapsada)
- age_group `26-35`: sens=0,705 / espec=0,745; `51-55`: sens=0,718 / espec=0,818

## "Screenshot textual" da inferência (1º exemplo)

```
======================================================================
VENDA NOVA (apenas caracteristicas - SEM vazamento de alvos):
======================================================================
  gender            : F
  occupation        : 3
  city_category     : A
  stay_years        : 2
  marital_status    : 0
  purchase_amount   : 89.9
  quantity          : 1

--> 3 PREVISOES com GRAU DE CERTEZA:

  [Categoria de produto]
    Previsao : Beleza  (certeza 42.34%)
    Top-3 candidatos:
      - Beleza           42.34%
      - Alimentos        20.46%
      - Eletronicos      13.26%

  [Metodo de pagamento]
    Previsao : PIX  (certeza 47.85%)
    Top-3 candidatos:
      - PIX              47.85%
      - Dinheiro         21.00%
      - Cartao_Debito    18.88%

  [Faixa etaria]
    Previsao : 18-25  (certeza 41.89%)
    Top-3 candidatos:
      - 18-25            41.89%
      - 0-17             25.43%
      - 26-35            24.41%
======================================================================
```

(Mais 2 vendas de exemplo são impressas; todas retornam categoria + pagamento + faixa etária com % de certeza e top-3.) **Funciona.**

## Artefatos confirmados (regenerados na execução)

- `outputs/matriz_confusao_{product_category,payment_method,age_group}.png` (3 PNGs)
- `outputs/eda_*.png` (4 figuras de EDA)
- `outputs/metricas.json` (espec + sens + TP/FN/FP/TN por classe nos 3 alvos)
- `outputs/modelo_multisaida.joblib` (1 modelo multi-saída + metadados; ~8 KB)
- `data/black_friday_sintetico.csv` (4000 linhas)

## Verificação de vazamento

`feature_names_in_` do pipeline = `[gender, city_category, occupation, stay_years, marital_status, purchase_amount, quantity]`; metadados `colunas_alvo` mantidos separados. Venda nova montada só com features. **Nenhum alvo** entra como feature. OK.

## Verificação extra

Reproduzi o split (stratify=age_group): **todas** as classes dos 3 alvos aparecem em treino e teste — logo o F1 macro/ponderado (calculado sem `labels=` explícito) está correto neste dataset.
