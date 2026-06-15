# Debate Q2 - Wine Quality (classificacao)

Tema: como avaliar e modelar um alvo `quality` (3 a 9) FORTEMENTE
desbalanceado, em que as classes 3 (30 amostras, 0,46%) e 9 (5 amostras,
0,08%) sao rarissimas. Tres perspectivas + decisao, com NUMEROS reais
obtidos nesta execucao (SEMENTE (random_state) = 42, teste = 1300 amostras).

---

## Perspectiva A - "Precisamos enxergar as classes raras" (foco em F1-macro) (ADOTADA)

Argumento: acuracia global e F1-weighted sao dominados pelas classes 5 e 6
(juntas ~76% dos dados). Um modelo pode ter 69% de acuracia e ainda assim
ACERTAR ZERO das classes 3 e 9. O F1-MACRO da peso igual a cada classe, entao
expoe esse fracasso.

Evidencia (7 classes, vencedor ExtraTrees):
- Acuracia global = 0,6946 / F1-weighted = 0,6830 (parecem "bons").
- F1-macro = 0,3987 (revela o problema).
- Acuracia por classe: quality=3 -> 0,00 ; quality=9 -> 0,00 ; quality=4 -> 0,14.

Conclusao da perspectiva A: o F1-macro e a metrica HONESTA aqui; foi por ela
que selecionamos o vencedor.

## Perspectiva B - "Acuracia global / F1-weighted refletem o uso real" (foco no negocio)

Argumento: na pratica, a maioria dos vinhos cai em 5/6/7. Se o objetivo e
classificar corretamente o GRANDE VOLUME, o F1-weighted (0,6830) e a acuracia
(0,6946) descrevem melhor a utilidade percebida. As classes 3 e 9 sao tao raras
que erra-las quase nao afeta o cliente medio.

Contraponto: se a aplicacao for justamente detectar vinhos excepcionais
(quality 8/9) ou defeituosos (3/4), a metrica weighted ESCONDE exatamente o que
importa. Micro-average, por sua vez, coincide com a acuracia em problemas de
rotulo unico e tambem ignora as raras.

## Perspectiva C - "Agrupar em FAIXAS resolve o desbalanceamento" (foco na granularidade) (ADOTADA)

Argumento: o alvo de 7 classes e, na verdade, uma escala ordinal com fronteiras
ruidosas (a diferenca entre 5 e 6 e subjetiva). Agrupar em 3 FAIXAS
(baixa <=5 / media =6 / alta >=7) junta as classes raras a vizinhas, reduz o
desbalanceamento e produz previsoes ACIONAVEIS.

Evidencia (mesmo modelo ExtraTrees, mesmo split estratificado):

| Metrica       | 7 classes | 3 faixas | Ganho   |
|---------------|-----------|----------|---------|
| Acuracia      | 0,6946    | 0,7323   | +0,0377 |
| F1-macro      | 0,3987    | 0,7241   | +0,3254 |
| F1-weighted   | 0,6830    | 0,7320   | +0,0490 |

O salto de F1-macro (+0,3254) e enorme: as faixas eliminam as classes onde o
modelo acertava 0%.

### Pros e contras de agrupar em faixas

Pros:
- F1-macro quase DOBRA (0,40 -> 0,72): nenhuma faixa fica em 0%.
- Faixas equilibradas (baixa 2384 / media 2836 / alta 1277) -> menos desbalanceamento.
- Resultado acionavel: "vinho bom/medio/ruim" e mais util que "nota 7 vs 8".

Contras:
- PERDE granularidade: deixa de distinguir um 8 de um 9 (ou um 3 de um 4).
- A escolha das fronteiras e arbitraria; outra divisao mudaria os numeros.
- A faixa "alta" ainda e a mais dificil (recall 0,5977), pois herda parte da
  raridade das notas altas.

---

## Decisao

1. Metrica de selecao: **F1-MACRO**, por ser a unica que penaliza ignorar as
   classes raras (perspectiva A). Reportamos tambem acuracia global, acuracia
   por classe (matriz de confusao) e F1-weighted para dar o quadro completo
   (perspectiva B).

2. Modelo: **ExtraTreesClassifier** vence os 3 avaliados em F1-macro nas 7
   classes (0,3987 vs AdaBoost 0,2107 vs MLP 0,2687).

3. Modelo PRINCIPAL de inferencia: **3 FAIXAS** (perspectiva C). O ganho de
   F1-macro de +0,3254 e decisivo e a saida e mais confiavel/acionavel. O
   modelo de 7 classes fica disponivel (joblib + flag `--modelo 7classes`) para
   quem precisar da granularidade fina, ciente de que ele falha nas notas 3 e 9.

Resumo: medir com F1-macro para nao se enganar com o desbalanceamento; entregar
em faixas para ter um modelo robusto; manter as 7 classes como opcao secundaria.

---

## Tabela-sintese

| Perspectiva | Foco | Adotada? | Decisao resultante |
|-------------|------|----------|--------------------|
| A - F1-macro | enxergar as classes raras | SIM | metrica de SELECAO do vencedor (penaliza ignorar 3/9) |
| B - acuracia / F1-weighted | volume real (5/6/7) | NAO (contraponto) | reportada junto, como quadro complementar de negocio |
| C - 3 faixas | reduzir desbalanceamento, saida acionavel | SIM | modelo PRINCIPAL de inferencia (baixa/media/alta) |

Em uma frase: **A** define COMO medimos, **C** define O QUE entregamos, e **B**
fica como leitura de negocio que nao deixamos esconder as classes raras.
