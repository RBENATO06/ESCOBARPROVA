# Debate Q1 - Decisoes de modelagem no agrupamento de pacientes

Documento de discussao das principais escolhas do projeto. Para cada questao
apresentamos 2-3 perspectivas e a **decisao adotada** (com a justificativa).

---

## Questao 1: Incluir ou nao as variaveis binarias no treino?

**Perspectiva A - Incluir tudo (passthrough).**
Argumenta que `anaemia`, `diabetes`, `high_blood_pressure`, `sex` e `smoking` sao
comorbidades clinicamente relevantes e poderiam ajudar a separar perfis. Excluir
informacao seria "jogar fora" sinal.

**Perspectiva B - Excluir do treino e usar so para descrever (ADOTADA).**
O GaussianMixture modela **densidades gaussianas**. Variaveis 0/1 nao sao
gaussianas: tratar uma coluna binaria como continua distorce a verossimilhanca
e as "distancias" entre pontos. Alem disso, misturar binarias com continuas
escalonadas combina metricas incompativeis (variancia de um 0/1 nao e comparavel
a de `platelets`). Mais honesto: treinar so com as continuas e usar as binarias
**depois** para *caracterizar* os grupos (% de fumantes, % diabeticos por
cluster) — exatamente o que `perfil_clusters.csv` faz.

**Perspectiva C - Codificacao mista (Gower / k-prototypes).**
Existiriam modelos para dados mistos (k-prototypes, distancia de Gower), mas
fogem do escopo do GMM pedido e adicionam complexidade. Ficam como trabalho
futuro.

**Decisao:** **excluir as binarias do treino** e usa-las apenas para descrever os
grupos. Coerente com a natureza gaussiana do GMM.

---

## Questao 2: Manter ou excluir `time` do treino?

**Perspectiva A - Manter `time` (mais informacao).**
`time` (tempo de acompanhamento) e numerico e disponivel; descarta-lo "joga
fora" uma variavel que separa bem os pacientes. Quanto mais sinal, melhor o
agrupamento.

**Perspectiva B - Excluir do treino e so descrever (ADOTADA).**
`time` e o tempo ate o obito ou a censura: e uma variavel de **desfecho/
acompanhamento**, nao uma caracteristica clinica da triagem. Sua correlacao com
`DEATH_EVENT` e **forte (-0.527)** — incluir `time` seria deixar a informacao do
desfecho **vazar pela porta dos fundos** (proxy do alvo) para um agrupamento que
deveria ser cego ao desfecho. Por isso treinamos so com as 6 continuas clinicas
e usamos `time` **depois**, apenas para descrever os grupos (`time_media` em
`perfil_clusters.csv`).

**Perspectiva C - Usar `time` como rotulo de validacao temporal.**
Poderia servir para uma validacao do tipo "os grupos diferem no tempo de
sobrevida?". E uma analise de sobrevivencia legitima, mas foge do escopo do
clustering pedido e fica como trabalho futuro.

**Decisao:** **excluir `time` do treino** (variavel de acompanhamento, vazamento
indireto -0.527), usando-a so para caracterizar os grupos. E a decisao de
modelagem mais forte do projeto.

---

## Questao 3: KMeans (esferico) vs GMM (eliptico)?

**Perspectiva A - KMeans.**
Simples, rapido, deterministico (com seed). Mas assume clusters **esfericos** e
de tamanho parecido, e faz atribuicao **rigida** (hard). Sofre com outliers e
com variaveis de escalas/dispersoes diferentes.

**Perspectiva B - GaussianMixture (ADOTADA).**
Permite clusters **elipticos** (covariancia `full`): cada grupo tem forma e
orientacao proprias, o que casa melhor com dados clinicos heterogeneos. E
**probabilistico**: para um paciente novo, devolve a **probabilidade de pertencer
a cada grupo** (`predict_proba`) — fundamental para comunicar *confianca* e
*ambiguidade* da atribuicao (no exemplo, 99.7% no Cluster 1).

**Decisao:** **GaussianMixture com covariancia `full`**. A flexibilidade
eliptica + a saida probabilistica justificam a maior complexidade.

---

## Questao 4: Qual scaler?

**Perspectiva A - StandardScaler (media/desvio).**
Padrao usual. Problema: e **sensivel a outliers** — a media e o desvio sao
"puxados" por valores extremos. Aqui ha caudas longas fortes
(`creatinine_phosphokinase` ate 7861, `serum_creatinine` ate 9.4).

**Perspectiva B - MinMaxScaler (0-1).**
Comprime tudo no intervalo [0,1], mas usa **min e max** — ainda mais sensivel a
outliers (um unico extremo estica a escala e achata o resto).

**Perspectiva C - RobustScaler (mediana/IQR) (ADOTADA).**
Centra na **mediana** e escala pelo **IQR**: estatisticas robustas, pouco
afetadas por extremos. Mantem as variaveis comparaveis sem deixar os outliers
dominarem a densidade gaussiana.

**Decisao:** **RobustScaler**, pela presenca clara de outliers nas variaveis de
exame.

---

## Questao 5: Como escolher o numero de componentes?

**Perspectiva A - Menor BIC, puro e simples.**
O BIC penaliza complexidade e, em teoria, indica o melhor k. Mas neste dataset o
menor BIC ocorre em **k=7**, gerando **clusters minusculos (n=6, 4, 4, ...)** e
silhouette de apenas **0.10** — grupos degenerados, sem utilidade clinica.

**Perspectiva B - Maior silhouette, puro.**
A silhueta favorece **k=2** (0.4542), mas isoladamente poderia preferir solucoes
triviais.

**Perspectiva C - Regra composta BIC + silhouette (ADOTADA).**
O BIC cai de forma quase **plana** ja a partir de k=2->3 (ganho relativo < 1%):
o ganho de complexidade nao compensa. Entao localizamos o **"joelho" do BIC**
(onde a melhora relativa fica abaixo de 1%) e, dentro dessa faixa estabilizada,
escolhemos o **melhor silhouette**. Resultado: **k=2**, parcimonioso e com boa
separacao geometrica, validado externamente pela diferenca de mortalidade entre
os grupos (43.9% vs 29.3%).

**Decisao:** **k=2**, via regra composta BIC (joelho) + silhouette. Documentado
no grafico `bic_por_componentes.png`.

---

## Sintese

| Questao | Decisao |
|---------|---------|
| Binarias no treino? | **Nao** — so descrevem os grupos |
| `time` no treino? | **Nao** — variavel de desfecho/acompanhamento (vazamento) |
| Algoritmo | **GaussianMixture** (eliptico, probabilistico) |
| Scaler | **RobustScaler** (robusto a outliers) |
| Nº de componentes | **k=2** (BIC-joelho + silhouette) |

Todas as decisoes priorizam **coerencia com o GMM**, **robustez a outliers** e
**interpretabilidade clinica** dos grupos, sem deixar informacao do desfecho
vazar para o agrupamento.
