# Q3 - Black Friday | Multi-classificacao MULTI-SAIDA (3 alvos)

Mini-projeto de Machine Learning (Aluno 3 - perfil **explorador**).
Objetivo: prever **3 alvos ao mesmo tempo** para uma venda de Black Friday,
cada previsao acompanhada do seu **grau de certeza** (`predict_proba`).

- **Alvos:** `product_category` (7 classes), `payment_method` (5 classes),
  `age_group` (7 classes).
- **Abordagem:** `MultiOutputClassifier` (UM modelo, 3 saidas) envolvendo
  `LogisticRegression`, dentro de um `Pipeline` com `ColumnTransformer`
  (one-hot nas categoricas + scaling nas numericas). Os 3 alvos sao treinados
  **juntos** (matriz `y` com 3 colunas).

---

## AVISO IMPORTANTE - DADOS SINTETICOS (PLACEHOLDER)

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                  AVISO: DADOS SINTETICOS (PLACEHOLDER)                   !!
!! O arquivo real data/black_friday.csv NAO foi encontrado.                 !!
!! Foi gerado um dataset SINTETICO (seed 42) apenas para PROVAR O PIPELINE. !!
!! NAO use estes numeros como conclusao real de negocio.                    !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Como **nao** existe `data/black_friday.csv`, o script gera automaticamente um
dataset sintetico de **4000 linhas** (`src/gerar_dados.py`, seed 42) e o salva
em `data/black_friday_sintetico.csv`. As features sao construidas com
**dependencias probabilisticas** em relacao aos alvos (ex.: ticket alto ->
Eletronicos/Cartao_Credito; genero F -> Beleza/Roupas; ocupacao alta -> faixas
mais velhas), de modo que os modelos fiquem **acima do acaso**.

Para usar dados reais: basta colocar um `data/black_friday.csv` com as colunas
do esquema abaixo; o pipeline passa a usa-lo automaticamente (e o banner some).

### Esquema dos dados

| Tipo | Colunas |
|------|---------|
| Features categoricas | `gender` (M/F), `city_category` (A/B/C) |
| Features numericas | `occupation` (0-20), `stay_years` (0-4), `marital_status` (0/1), `purchase_amount` (float), `quantity` (1-5) |
| Alvos | `product_category`, `payment_method`, `age_group` |

---

## Como executar

A partir desta pasta (`q3_black_friday/`):

```powershell
# Windows PowerShell
./run.ps1
```

```bash
# Bash
bash run.sh
```

Os scripts `run.ps1`/`run.sh` rodam o fluxo completo em 3 passos:
`[1/3]` instala as dependencias (`pip install -r requirements.txt`),
`[2/3]` roda `src/main.py` (gera/carrega dados, EDA, treina, avalia, salva
artefatos) e `[3/3]` roda `src/inferencia.py`. **Sem venv:** usa o `python` do
ambiente atual.

Ou manualmente:

```bash
python src/main.py        # gera/carrega dados, EDA, treina, avalia, salva artefatos
python src/inferencia.py  # carrega o modelo e preve 3 vendas novas de exemplo
```

A inferencia aceita flags (`argparse`):

```bash
python src/inferencia.py --exemplo 2 --top 5   # so o exemplo 2, Top-5 candidatos por alvo
python src/inferencia.py --modelo outros/m.joblib
```

---

## RESULTADOS REAIS (dados sinteticos placeholder, seed 42)

Conjunto: 4000 linhas | split **estratificado** por `age_group`
(`random_state=42`): **treino = 3000**, **teste = 1000**.
Estimador base: `LogisticRegression(max_iter=2000, random_state=42)`.

### Metricas globais por alvo (conjunto de teste)

| Alvo | Acuracia global | Acaso (uniforme) | F1 macro | F1 ponderado | Certeza media |
|------|-----------------|------------------|----------|--------------|---------------|
| `product_category` | **0.3730** | 0.1429 | 0.2025 | 0.2867 | 0.3453 |
| `payment_method`   | **0.3880** | 0.2000 | 0.2968 | 0.3320 | 0.3648 |
| `age_group`        | **0.4050** | 0.1429 | 0.3385 | 0.3672 | 0.3350 |

Os 3 alvos ficam **claramente acima do acaso** (ex.: `age_group` 0.405 vs 0.143,
~2.8x; `product_category` 0.373 vs 0.143, ~2.6x), confirmando que o pipeline
aprende o sinal injetado nos dados sinteticos.

A **certeza media** (`predict_proba`) acompanha de perto a acuracia de cada alvo
(ex.: `payment_method` 0.3648 vs 0.388; `age_group` 0.3350 vs 0.405) - sinal de
calibracao razoavel para um problema dificil, detalhado em
[`outputs/debate_Q3.md`](outputs/debate_Q3.md).

### Sensibilidade / Especificidade por classe (one-vs-rest)

Calculadas da matriz de confusao de cada alvo:
`sensibilidade = TP/(TP+FN)`, `especificidade = TN/(TN+FP)`.

**`product_category`**

| Classe | Sensibilidade | Especificidade | F1 | Suporte |
|--------|---------------|----------------|------|---------|
| Alimentos   | 0.314 | 0.891 | 0.330 | 156 |
| Beleza      | 0.629 | 0.743 | 0.484 | 210 |
| Brinquedos  | 0.022 | 0.989 | 0.040 | 89  |
| Casa        | 0.033 | 0.973 | 0.050 | 92  |
| Eletronicos | 0.766 | 0.607 | 0.514 | 244 |
| Esportes    | 0.000 | 1.000 | 0.000 | 137 |
| Roupas      | 0.000 | 1.000 | 0.000 | 72  |

**`payment_method`**

| Classe | Sensibilidade | Especificidade | F1 | Suporte |
|--------|---------------|----------------|------|---------|
| Boleto         | 0.099 | 0.971 | 0.160 | 172 |
| Cartao_Credito | 0.491 | 0.851 | 0.440 | 167 |
| Cartao_Debito  | 0.000 | 1.000 | 0.000 | 149 |
| Dinheiro       | 0.419 | 0.767 | 0.375 | 222 |
| PIX            | 0.676 | 0.601 | 0.510 | 290 |

**`age_group`**

| Classe | Sensibilidade | Especificidade | F1 | Suporte |
|--------|---------------|----------------|------|---------|
| 0-17  | 0.171 | 0.977 | 0.250 | 105 |
| 18-25 | 0.324 | 0.896 | 0.335 | 145 |
| 26-35 | 0.705 | 0.745 | 0.526 | 207 |
| 36-45 | 0.141 | 0.965 | 0.210 | 149 |
| 46-50 | 0.327 | 0.895 | 0.350 | 162 |
| 51-55 | 0.718 | 0.818 | 0.531 | 156 |
| 55+   | 0.105 | 0.988 | 0.168 | 76  |

> Leitura critica (perfil explorador): classes com **especificidade ~1.0 e
> sensibilidade ~0.0** (ex.: `Esportes`, `Roupas`, `Cartao_Debito`) NAO sao um
> bom resultado - significam que o modelo linear quase nunca preve aquela
> classe. A especificidade alta isolada engana; por isso ela e sempre lida
> junto da sensibilidade e do F1. Um estimador nao-linear (ex.: arvores) ou
> reamostragem provavelmente resgataria essas classes minoritarias.

Valores completos (inclusive TP/FN/FP/TN por classe) em
[`outputs/metricas.json`](outputs/metricas.json).

---

## Inferencia (venda nova -> 3 previsoes com % de certeza)

`src/inferencia.py` carrega **apenas** o modelo salvo (sem re-treino, **sem
vazamento**: a venda nova contem somente features, nenhum alvo) e devolve as 3
previsoes com grau de certeza. Exemplo real de saida:

```
VENDA NOVA: gender=F, occupation=3, city_category=A, stay_years=2,
            marital_status=0, purchase_amount=89.90, quantity=1

[Categoria de produto] Previsao: Beleza  (certeza 42.34%)
[Metodo de pagamento]  Previsao: PIX     (certeza 47.85%)
[Faixa etaria]         Previsao: 18-25   (certeza 41.89%)
```

As previsoes sao coerentes com o sinal sintetico (ticket baixo + genero F ->
Beleza; jovem/PIX). O modulo tambem imprime o **ranking de candidatos** (Top-N,
default 3, ajustavel via `--top`), marcando a classe vencedora com
`<== atribuido`.

---

## Multi-saida (1 modelo, 3 saidas) vs 3 modelos separados

- **Multi-saida (`MultiOutputClassifier`, usado aqui):** um unico objeto treina
  e serve os 3 alvos, um so `joblib`, um so pre-processamento. `predict_proba`
  retorna **uma lista** (uma matriz de probabilidades por alvo), exatamente o
  que precisamos para o grau de certeza de cada previsao. Menos codigo, menos
  manutencao.
- **Detalhe estatistico:** internamente o `MultiOutputClassifier` ajusta um
  estimador **independente por alvo** - ele NAO modela correlacao entre os
  alvos. Em acuracia, e **equivalente** a treinar 3 modelos iguais separados; o
  ganho e de **engenharia/organizacao**.
- **3 modelos separados:** dariam flexibilidade para algoritmo/hiperparametro
  diferente por alvo, ao custo de mais codigo e mais arquivos.
- **Evolucao:** para *explorar* a correlacao real entre alvos (no gerador,
  `age_group` influencia `payment_method`), o caminho seria `ClassifierChain`.

Discussao aprofundada (probabilidade crua vs calibrada; especificidade/
sensibilidade multiclasse; multi-saida vs separados) em
[`outputs/debate_Q3.md`](outputs/debate_Q3.md).

---

## Estrutura do projeto

```
q3_black_friday/
├── data/
│   ├── black_friday.csv                # (opcional) dataset REAL: se existir, e usado no lugar do sintetico
│   └── black_friday_sintetico.csv      # gerado (seed 42) quando nao ha o real
├── outputs/                            # SO artefatos gerados pelo pipeline
│   ├── eda_distribuicao_alvos.png
│   ├── eda_valor_por_categoria.png
│   ├── eda_pagamento_por_cidade.png
│   ├── eda_correlacao_numericas.png
│   ├── matriz_confusao_product_category.png        # contagem (fmt="d")
│   ├── matriz_confusao_product_category_normalizada.png  # recall visual (por linha)
│   ├── matriz_confusao_payment_method.png
│   ├── matriz_confusao_payment_method_normalizada.png
│   ├── matriz_confusao_age_group.png
│   ├── matriz_confusao_age_group_normalizada.png
│   ├── metricas.json                   # acuracia, F1, sensibilidade/especificidade por classe
│   ├── modelo_multisaida.joblib        # modelo multi-saida persistido (compress=3)
│   └── debate_Q3.md
├── src/
│   ├── gerar_dados.py                  # carga real ou geracao sintetica (banner placeholder)
│   ├── metricas_utils.py               # sensibilidade/especificidade one-vs-rest
│   ├── main.py                         # EDA + treino multi-saida + avaliacao + persistencia
│   └── inferencia.py                   # inferencia separada (venda nova -> 3 previsoes)
├── .gitignore
├── requirements.txt
├── run.ps1
└── run.sh
```

## Decisoes tecnicas e trade-offs

- **`SEMENTE` (random_state) = 42** em gerador, split e modelo
  (reprodutibilidade total; a constante e sempre `SEMENTE`, nunca `RANDOM_STATE`).
- **Split estratificado por `age_group`** (alvo principal) para preservar a
  proporcao das faixas no teste.
- **`OneHotEncoder(handle_unknown="ignore")`**: robusto a categorias novas na
  inferencia.
- **`LogisticRegression`** (linear) e simples e interpretavel, mas tem
  dificuldade com classes minoritarias e fronteiras nao-lineares - visivel nas
  classes com sensibilidade ~0. Trade-off consciente: priorizamos clareza do
  pipeline multi-saida sobre maximizar acuracia.
- **`matplotlib.use("Agg")`** antes de `pyplot` (backend nao-interativo).
- **pandas 3.0**: uso de `.copy()` ao separar features/rotulos (sem chained
  assignment).
- Ambiente: Python 3.14.4, scikit-learn 1.8.0 (nesta versao o parametro
  `multi_class` da `LogisticRegression` foi removido - o tratamento multiclasse
  e automatico).
