# Debate Q3 - Black Friday (multi-classificacao multi-saida)

> Aviso: os numeros citados vem de **dados SINTETICOS (placeholder)**, gerados
> com seed 42 apenas para provar o pipeline. Servem para discutir o *metodo*,
> nao para conclusoes de negocio.

Este documento debate tres pontos exigidos, cada um com 2-3 perspectivas e uma
**decisao** final justificada.

---

## 1. `predict_proba` cru vs calibrado

O grau de certeza de cada previsao vem do `predict_proba`. Mas essa
probabilidade e *confiavel* como "chance real de acerto"?

- **Perspectiva A - usar a probabilidade crua (ADOTADA).**
  Para `LogisticRegression`, a saida ja e uma probabilidade no sentido do
  modelo (otimiza a log-verossimilhanca / entropia cruzada). Para o objetivo
  aqui - **ordenar** candidatos e mostrar um "% de certeza" relativo - isso
  basta. E simples, sem etapa extra, sem risco de overfit de calibracao em
  conjunto pequeno.

- **Perspectiva B - calibrar (Platt/`sigmoid` ou `isotonic` via
  `CalibratedClassifierCV`).**
  Probabilidade crua pode ser **mal-calibrada**: o modelo diz "47%" mas acerta
  em proporcao diferente disso. Calibracao alinha a probabilidade declarada com
  a frequencia observada. Util quando a porcentagem alimenta uma **decisao com
  limiar** (ex.: "so recomendar PIX se certeza > 60%").

- **Perspectiva C - olhar o sintoma antes de decidir.**
  Nas nossas metricas a **certeza media** ficou perto da **acuracia global**
  por alvo (ex.: payment_method certeza media ~0.36 vs acuracia 0.388;
  age_group ~0.34 vs 0.405). Ou seja, a confianca media nao esta absurdamente
  inflada nem deflacionada - sinal de calibracao *razoavel* para um problema
  dificil e ruidoso como este.

**Decisao:** manter `predict_proba` **cru** nesta entrega, porque (i) o uso e
ranquear/exibir certeza, nao cravar limiares de negocio; (ii) a certeza media
acompanha a acuracia, indicando calibracao aceitavel; e (iii) calibrar com
`isotonic` em poucos dados por classe poderia piorar (overfit). **Recomendacao
para producao:** se a porcentagem virar gatilho de decisao, calibrar com
`CalibratedClassifierCV(method="sigmoid", cv=...)` sobre um conjunto de
validacao separado e checar com um *reliability diagram* / Brier score.

---

## 2. Especificidade e sensibilidade em problema MULTICLASSE

Sensibilidade e especificidade nascem do cenario binario. Com 5-7 classes, como
reporta-las?

- **Perspectiva A - one-vs-rest a partir da matriz de confusao (ADOTADA).**
  Para cada classe `i`: positivo = ser da classe `i`, negativo = qualquer
  outra. Da matriz `mc`:
  `TP = mc[i,i]`; `FN = soma(linha i) - TP`; `FP = soma(coluna i) - TP`;
  `TN = total - TP - FN - FP`. Entao
  `sensibilidade = TP/(TP+FN)` e `especificidade = TN/(TN+FP)`.
  Vantagem: transparente, deriva direto da matriz que ja salvamos por alvo.

- **Perspectiva B - cuidado com a especificidade "inflada".**
  Em 7 classes, os negativos sao ~6/7 das amostras. Classes que o modelo quase
  nunca preve (ex.: `Esportes` e `Roupas` com sensibilidade 0.0) exibem
  **especificidade ~1.0** - parece otimo, mas so significa "raramente erra
  chamando algo de Esportes", porque quase nunca chama. Por isso a
  especificidade isolada engana; ela **precisa** ser lida junto da
  sensibilidade e do F1.

- **Perspectiva C - qual numero priorizar.**
  Para campanha de marketing (acionar quem realmente e da faixa/categoria), a
  **sensibilidade** (recall) costuma importar mais que a especificidade. O
  **F1 macro** resume os dois e nao deixa as classes raras "sumirem" na media -
  por isso reportamos F1 macro e ponderado.

**Decisao:** reportar **sensibilidade e especificidade por classe (one-vs-rest)
sempre em conjunto**, mais F1 por classe e F1 macro/ponderado global. Nunca
celebrar especificidade alta isolada: cruzamos com sensibilidade para detectar
classes que o modelo simplesmente ignora (as de especificidade ~1.0 e
sensibilidade ~0.0).

---

## 3. Multi-saida (1 modelo, 3 saidas) vs 3 modelos separados

O enunciado pede a abordagem **multi-saida**: `MultiOutputClassifier` em volta
de um estimador base, dentro de um `Pipeline` com `ColumnTransformer`, treinando
os 3 alvos juntos.

- **Perspectiva A - multi-saida (ADOTADA).**
  Uma unica abstracao treina/serve os 3 alvos; um so `joblib`; um so
  pre-processamento. `predict_proba` devolve **uma lista** (uma matriz por
  alvo), perfeita para extrair o grau de certeza de cada previsao. Codigo e
  manutencao mais enxutos.
  *Detalhe importante:* `MultiOutputClassifier` **ajusta um estimador
  independente por alvo internamente** - ele NAO modela correlacao entre os
  alvos por si so. O ganho aqui e de **organizacao/engenharia**, nao
  estatistico.

- **Perspectiva B - 3 modelos separados.**
  Maxima flexibilidade: cada alvo poderia ter um algoritmo/hiperparametro
  diferente (ex.: KNN para um, LogisticRegression para outro) e ajuste fino
  independente. Custo: 3 pipelines, 3 arquivos, 3 rotinas de avaliacao - mais
  codigo e mais chance de divergencia entre eles.

- **Perspectiva C - e a correlacao entre alvos?**
  Existe relacao real (no nosso gerador, `age_group` influencia
  `payment_method`). Quem quisesse **explorar** essa dependencia usaria
  `ClassifierChain` (encadeia alvos, passando a previsao de um como feature do
  proximo) ou um classificador multi-rotulo nativo. Isso foge do escopo pedido,
  mas e a evolucao natural se a correlacao entre alvos for forte.

**Decisao:** usar **`MultiOutputClassifier` (multi-saida)** como pedido. Para
*este* caso, com o mesmo estimador base servindo bem aos 3 alvos e o requisito
de um `predict_proba` por alvo, a abordagem multi-saida entrega o melhor
custo/beneficio de engenharia. Em termos de *acuracia*, multi-saida com
estimadores independentes e **equivalente** a treinar 3 modelos iguais
separados - a diferenca real apareceria so com `ClassifierChain` explorando a
correlacao entre alvos, o que fica como trabalho futuro.

---

## Resumo das decisoes

| Tema | Decisao |
|------|---------|
| Probabilidade | `predict_proba` cru (calibrar so se virar gatilho de decisao) |
| Metricas multiclasse | sensibilidade + especificidade one-vs-rest, sempre juntas, + F1 macro |
| Arquitetura | multi-saida (`MultiOutputClassifier`); `ClassifierChain` como evolucao |
