# Q2 - Wine Quality (Classificacao) - Aluno 3

Perfil EXPLORADOR: EDA detalhada com seaborn, muitos prints de exploracao,
discussao de trade-offs e inferencia em arquivo separado. Tudo em PT-BR,
`SEMENTE (random_state) = 42` em todo split/modelo, datasets por caminho
ancorado no script (pathlib).

## Problema

Prever a `quality` (nota de 3 a 9) de vinhos, UNINDO os datasets tinto
(`winequality-red.csv`, 1599 linhas) e branco (`winequality-white.csv`, 4898
linhas), preservando os nomes das colunas. Total: **6497 amostras**, 11
caracteristicas fisico-quimicas + alvo.

As classes sao **muito desbalanceadas**:

| quality | amostras | %      |
|---------|----------|--------|
| 3       | 30       | 0,46%  |
| 4       | 216      | 3,32%  |
| 5       | 2138     | 32,91% |
| 6       | 2836     | 43,65% |
| 7       | 1079     | 16,61% |
| 8       | 193      | 2,97%  |
| 9       | 5        | 0,08%  |

As classes 3 e 9 sao rarissimas (essa e a maior dificuldade do problema).

### Colunas do dataset (o que entra no modelo)

As 11 caracteristicas fisico-quimicas sao TODAS numericas e formam o `X`:

| Caracteristica         | Tipo     | Entra no X? |
|------------------------|----------|-------------|
| fixed acidity          | numerica | sim         |
| volatile acidity       | numerica | sim         |
| citric acid            | numerica | sim         |
| residual sugar         | numerica | sim         |
| chlorides              | numerica | sim         |
| free sulfur dioxide    | numerica | sim         |
| total sulfur dioxide   | numerica | sim         |
| density                | numerica | sim         |
| pH                     | numerica | sim         |
| sulphates              | numerica | sim         |
| alcohol                | numerica | sim         |
| **quality**            | alvo (ordinal 3-9) | NAO (e o `y`) |
| **tipo_vinho**         | categorica (tinto/branco) | NAO (so EDA) |

`tipo_vinho` e usado apenas na EDA (comparar tinto vs branco); ele **NAO entra
no `X`** para que a inferencia nao dependa de conhecer o tipo do vinho.

## Metodologia

- **Pipeline** (sklearn `Pipeline`) com `StandardScaler` + classificador. O
  scaler e ESSENCIAL para o MLP e inofensivo para arvores/boosting; manter o
  mesmo formato facilita a inferencia. Sem vazamento: o scaler e ajustado
  SOMENTE no treino.
- **Split estratificado** 80/20 (`stratify`, `SEMENTE (random_state) = 42`):
  treino 5197 / teste 1300, com a proporcao por classe preservada.
- **3 classificadores avaliados** (obrigatorios): `ExtraTreesClassifier`,
  `AdaBoostClassifier`, `MLPClassifier`.
- **Selecao do vencedor por F1-MACRO** (penaliza ignorar classes raras).
- Metricas reportadas: acuracia global, acuracia por classe (matriz de
  confusao / recall por classe), F1-macro e F1-weighted.

## Resultados REAIS (7 classes)

Criterio de selecao: **F1-macro**.

| Modelo         | Acuracia | F1-macro   | F1-weighted |
|----------------|----------|------------|-------------|
| **ExtraTrees** | 0,6946   | **0,3987** | 0,6830      |
| AdaBoost       | 0,5185   | 0,2107     | 0,4921      |
| MLP            | 0,5623   | 0,2687     | 0,5466      |

**Vencedor: ExtraTreesClassifier** (F1-macro = 0,3987).

Acuracia por classe do vencedor (recall):

| quality | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
|---------|------|------|------|------|------|------|------|
| recall  | 0,00 | 0,14 | 0,70 | 0,81 | 0,56 | 0,36 | 0,00 |

Mesmo o vencedor NAO acerta nenhuma amostra das classes 3 e 9 (raridade
extrema) - o que motiva o experimento de faixas. Atencao: as classes 3 e 9 tem
apenas 6 e 1 amostra no teste, respectivamente, entao o recall 0,00 delas e
estatisticamente NAO-CONFIAVEL (poucos exemplos). Matrizes de confusao:
`outputs/matriz_confusao_7classes.png` (contagens) e
`outputs/matriz_confusao_7classes_normalizada.png` (recall por classe, em que o
fracasso das classes 3 e 9 salta aos olhos).

## Experimento secundario: 7 classes vs 3 FAIXAS

Faixas definidas (e justificadas): **baixa (<=5) | media (=6) | alta (>=7)**.
A divisao foi feita nos pontos de maior massa (5 e 6 sao as duas maiores
classes), reduzindo o desbalanceamento - faixas: baixa 2384, media 2836,
alta 1277.

Comparacao (mesmo modelo ExtraTrees, mesmo split estratificado):

| Metrica     | 7 classes | 3 faixas | Ganho   |
|-------------|-----------|----------|---------|
| Acuracia    | 0,6946    | 0,7323   | +0,0377 |
| F1-macro    | 0,3987    | 0,7241   | +0,3254 |
| F1-weighted | 0,6830    | 0,7320   | +0,0490 |

Impacto: agrupar em faixas faz o **F1-macro quase dobrar** (+0,3254), porque
nenhuma faixa fica com 0% de acerto. Em troca, perde-se granularidade (nao se
distingue um 8 de um 9). Acuracia por faixa: baixa 0,74 / media 0,79 /
alta 0,60. Matriz de confusao: `outputs/matriz_confusao_3faixas.png`.

### Modelo PRINCIPAL de inferencia: **3 FAIXAS**

Justificativa: o ganho de F1-macro de +0,3254 e decisivo e a previsao
(baixa/media/alta) e mais robusta e acionavel. O modelo de 7 classes fica
salvo como opcao secundaria (granularidade fina), ciente de que falha nas
notas 3 e 9. Ver discussao completa em `outputs/debate_Q2.md`.

## Estrutura

```
Exercicio 2_wine_quality/
  data/                       winequality-red.csv, winequality-white.csv (sep=';')
  src/
    main.py                   EDA + treino dos 3 modelos + faixas + salvar artefatos
    inferencia.py             inferencia em vinho NOVO (modelo principal ou --modelo)
  outputs/
    eda_*.png                 graficos da EDA (distribuicao, correlacao, alcohol, tipo)
    comparacao_modelos.json   metricas dos 3 modelos (7 classes)
    comparacao_modelos.txt    mesma comparacao em texto
    matriz_confusao_7classes.png             matriz de confusao do vencedor (contagens)
    matriz_confusao_7classes_normalizada.png matriz normalizada (recall por classe)
    matriz_confusao_3faixas.png              matriz de confusao das 3 faixas
    metricas.json             metricas consolidadas + comparacao 7classes vs 3faixas
    modelo_campeao_7classes.joblib   pipeline vencedor (7 classes)
    modelo_campeao_faixas.joblib     pipeline vencedor (3 faixas) - PRINCIPAL
    debate_Q2.md              debate (classes raras, F1-macro vs micro/weighted, faixas)
  requirements.txt
  run.sh / run.ps1            fluxo completo
  README.md
```

## Como executar

PowerShell (Windows):

```powershell
.\run.ps1
```

Bash:

```bash
bash run.sh
```

Ou manualmente:

```bash
python src/main.py                           # EDA, treino, comparacao, salva artefatos
python src/inferencia.py                     # inferencia em vinhos novos (modelo principal)
python src/inferencia.py --modelo 7classes   # forca o modelo de 7 classes
python src/inferencia.py --top 5             # mostra Top-5 no ranking de cada vinho
python src/inferencia.py --csv vinhos.csv    # le um LOTE de vinhos novos do usuario
```

**Determinismo:** a execucao e reprodutivel — a MESMA `SEMENTE (random_state) =
42` em todo split/modelo gera SEMPRE as mesmas metricas, e `metricas.json` grava
o valor da seed (chave `random_state`) para auditoria.

## Significado das saidas (inferencia)

A inferencia imprime, por vinho, a previsao (faixa baixa/media/alta ou nota 3-9)
e um **ranking Top-N** das classes com `%`, marcando a vencedora com
`<== atribuido`. Duas notas honestas:

1. Os 3 vinhos de exemplo sao **FABRICADOS a mao** (valores fisico-quimicos
   plausiveis, porem NAO copiados de nenhuma linha dos `winequality-*.csv`).
   Servem so para demonstrar a inferencia em instancias ineditas.
2. A "probabilidade" e a **fracao das 400 arvores** do ExtraTrees que votou na
   classe vencedora (confianca do comite, NAO uma probabilidade calibrada). Como
   os exemplos sao vinhos de FRONTEIRA, o comite fica dividido e a certeza fica
   em torno de 0,52-0,56. Perfis mais extremos concentrariam mais os votos.

## Limitacoes e proximos passos

`Leitura critica (perfil explorador):` somos transparentes sobre onde o modelo
falha. As classes 3 e 9 ficam com **recall 0,00** — mas isso e limite dos DADOS
(6 e 1 amostra no teste), nao do modelo: nao ha exemplos suficientes para
aprender. Mesmo agrupando em faixas, a faixa **alta** continua a mais fraca
(recall ~0,60), pois herda a raridade das notas altas.

Proximos passos que tentariamos:
- **reamostragem / SMOTE** para as classes raras (3, 4, 8, 9);
- **classificador ordinal** ou regressao com arredondamento (a escala 3-9 e
  ORDINAL, nao apenas categorica) — penalizar errar por 1 menos que errar por 4;
- **calibracao de probabilidades** (Platt/isotonica) para que o grau de certeza
  vire probabilidade calibrada de fato, em vez de fracao de votos.

## Observacoes tecnicas

- `matplotlib.use("Agg")` antes do pyplot (backend sem janela).
- pandas 3.0: uso de `.copy()`, sem chained assignment.
- ExtraTrees e MLP usam `class_weight`/scaler conforme necessario; MLP com
  `max_iter=1000` + `early_stopping` para convergir.
- Inferencia sem vazamento: o `StandardScaler` ja foi ajustado apenas no treino
  dentro do pipeline salvo; a inferencia so aplica `transform`/`predict`.
