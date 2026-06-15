# Q1 - Heart Failure: Agrupamento de Pacientes com GaussianMixture

> **Perfil EXPLORADOR.** Este projeto nao entrega so um modelo: entrega uma
> **investigacao narrada em etapas**. Olhamos os dados com EDA em seaborn,
> levantamos hipoteses e trade-offs em voz alta (ver `outputs/debate_Q1.md`),
> justificamos o PORQUE de cada decisao e fechamos com uma **inferencia separada**
> que atribui o paciente novo a um grupo com **grau de certeza** (probabilidade).

Mini-projeto de **aprendizado NAO supervisionado (clustering)**. O objetivo e
**descobrir grupos (perfis) de pacientes** com base em suas caracteristicas
clinicas e, dado um paciente **desconhecido**, dizer a qual grupo ele pertence.

O desfecho clinico `DEATH_EVENT` (obito durante o acompanhamento) **NAO** entra
no treino. Ele e usado **apenas depois** para *caracterizar* os grupos (ver se
um grupo tem mais obitos que outro), funcionando como validacao externa de que
a estrutura encontrada faz sentido clinico.

---

## 1. Objetivo

- **Tarefa:** agrupar (clusterizar) os 299 pacientes em perfis homogeneos.
- **Uso pratico:** um paciente novo e atribuido a um grupo, com a **probabilidade**
  de pertencer a cada grupo (o GaussianMixture e probabilistico).
- **Restricao:** `DEATH_EVENT` fora do treino; serve so para descrever os grupos.

---

## 2. Dataset

Arquivo: `data/heart_failure_clinical_records_dataset.csv` (299 linhas, 13 colunas, **sem valores ausentes**).

| Tipo | Colunas |
|------|---------|
| Continuas (usadas no treino) | `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium` |
| Binarias 0/1 (FORA do treino) | `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking` |
| Tempo de acompanhamento (FORA do treino) | `time` |
| Desfecho clinico (FORA do treino, so descreve grupos) | `DEATH_EVENT` |

---

## 3. Decisoes de modelagem (justificadas)

### 3.1 Algoritmo: GaussianMixture (GMM)
Escolhemos o **GaussianMixture** porque:
- E um modelo **probabilistico**: alem de prever o grupo (`predict`), fornece a
  **responsabilidade/probabilidade** de cada componente (`predict_proba`) — ideal
  para mostrar a *confianca* da atribuicao de um paciente novo.
- Usa covariancia `full` (**clusters elipticos**), com cada grupo podendo ter
  forma e orientacao proprias. Em dados clinicos as variaveis tem escalas e
  dispersoes muito diferentes, entao a forma esferica do KMeans e restritiva
  demais.

### 3.2 Binarias EXCLUIDAS do treino
As variaveis binarias (`anaemia`, `diabetes`, ...) **nao entram no treino**.
Motivos: (i) o GMM modela densidades **gaussianas**, e variaveis 0/1 nao sao
gaussianas (distancia/densidade ficam distorcidas); (ii) misturar binarias com
continuas escalonadas mistura "metricas" incompativeis. Em vez disso, usamos as
binarias **depois** para *descrever* cada grupo (ex.: % de fumantes, % de
diabeticos por cluster) — interpretacao mais honesta.

### 3.3 Decisao sobre `time` (EXCLUIDA)
`time` e o **tempo de acompanhamento** ate o obito ou a censura. E uma variavel
de **desfecho/acompanhamento**, nao uma caracteristica clinica intrinseca do
paciente na triagem. Sua correlacao com `DEATH_EVENT` e **forte (-0.527)**:
incluir `time` seria deixar a informacao do desfecho "vazar" pela porta dos
fundos para o agrupamento. Por isso **excluimos `time` do treino** (so a usamos
para descrever os grupos depois).

### 3.4 Scaler: RobustScaler
Usamos **RobustScaler** (centra na mediana, escala pelo IQR). As variaveis
`creatinine_phosphokinase`, `platelets` e `serum_creatinine` tem **caudas longas
e outliers fortes** (ex.: CPK chega a 7861, creatinina a 9.4). O `StandardScaler`
(media/desvio) seria "puxado" por esses outliers; o RobustScaler e resistente a
eles, mantendo as variaveis em escala comparavel sem deixar os extremos dominarem.

### 3.5 Escolha do numero de componentes (BIC + silhouette)
Calculamos o **BIC** e o **silhouette** para `k` de 2 a 8 (grafico
`bic_por_componentes.png`):

| k | BIC | silhouette |
|---|-----|-----------|
| 2 | 4852.68 | **0.4542** |
| 3 | 4823.20 | 0.4400 |
| 4 | 4758.73 | 0.2123 |
| 5 | 4719.66 | 0.2446 |
| 6 | 4717.19 | 0.1364 |
| 7 | **4692.38** (min) | 0.1022 |
| 8 | 4736.12 | 0.1170 |

O **menor BIC isolado** ocorreria em `k=7`, mas isso gera **clusters minusculos
(n=6, 4, 4, ...)** e silhouette pessimo (0.10) — grupos degenerados e sem
utilidade clinica. Como o ganho de BIC de `k=2` ate `k=7` e **marginal** (a curva
"achata" logo no inicio), aplicamos uma regra composta e transparente: localizar
o **"joelho" do BIC** (onde a melhora relativa cai abaixo de 1%) e, dentro dessa
faixa estabilizada, escolher o **melhor silhouette**. O resultado e **k=2**,
parcimonioso e com a melhor separacao geometrica.

---

## 4. Resultados REAIS (k=2)

Metricas (de `outputs/metricas.json`):

| Metrica | Valor |
|---------|-------|
| Nº de componentes | **2** |
| BIC | 4852.68 |
| Silhouette (maior melhor) | **0.4542** |
| Davies-Bouldin (menor melhor) | 1.8074 |
| Calinski-Harabasz (maior melhor) | 68.86 |

Perfil dos grupos (de `outputs/perfil_clusters.csv`):

| Cluster | n | Taxa DEATH_EVENT | CPK (media) | serum_creatinine (media) | ejection_fraction (media) |
|---------|---|------------------|-------------|--------------------------|---------------------------|
| **0** | 57 | **43.9%** | 1631.1 | 2.39 | 36.9 |
| **1** | 242 | **29.3%** | 334.7 | 1.16 | 38.4 |

**Interpretacao:**
- **Cluster 0** (57 pacientes) = perfil de **maior estresse renal/cardiaco**:
  CPK e creatinina serica bem mais altas. Tem a **maior taxa de obito (43.9%)**.
- **Cluster 1** (242 pacientes) = perfil **metabolicamente mais estavel**:
  marcadores mais baixos. Taxa de obito menor (29.3%).
- A taxa de obito **separa os grupos (43.9% vs 29.3%)** mesmo o `DEATH_EVENT`
  **nunca** tendo entrado no treino — boa evidencia de que a estrutura nao
  supervisionada captura sinal clinicamente relevante.

Perfil das binarias por cluster (proporcao de '1'): no Cluster 0 ha mais homens
(72% vs 63%); o Cluster 1 tem mais anemia (46% vs 30%) e diabetes (44% vs 32%).
Detalhes completos em `outputs/perfil_clusters.csv`.

### Inferencia (paciente novo de exemplo)
O `src/inferencia.py` carrega o modelo salvo e classifica um paciente novo com
marcadores baixos (CPK=250, creatinina=1.1): atribuido ao **Cluster 1** com
**99.7%** de probabilidade (responsabilidade `predict_proba`). Cluster 0: 0.3%.

---

## 5. Comparacao com as outras duas abordagens

| Aspecto | **Esta versao (minha)** | Versao KMeans | Versao Hierarquica |
|---------|-------------------------|---------------|--------------------|
| Algoritmo | GaussianMixture (elipticos, probabilistico) | KMeans (esfericos, rigido) | Aglomerativo (dendrograma) |
| Binarias | **Excluidas** (so descrevem) | `passthrough` (entram no treino) | entram (MinMax) |
| Scaler | **RobustScaler** (robusto a outliers) | StandardScaler | MinMaxScaler |
| Saida por paciente | grupo **+ probabilidades** | so o grupo (hard) | so o grupo |

**Trade-offs:** o KMeans+`passthrough`+StandardScaler e mais simples e rapido,
mas assume clusters esfericos e sofre com os outliers das variaveis de cauda
longa; alem disso, misturar binarias 0/1 com continuas distorce as distancias.
O Hierarquico+MinMax da um dendrograma interpretavel, mas o MinMax e
**sensivel a outliers** (o maximo "estica" a escala) e nao fornece probabilidade
de pertinencia. Minha versao (GMM + binarias fora + RobustScaler) e a mais
adequada quando se quer **probabilidade de pertencer a cada grupo** e
**robustez a outliers**, ao custo de mais hiperparametros (tipo de covariancia,
nº de componentes).

---

## 6. Como rodar

**Windows (PowerShell):**
```powershell
./run.ps1
```

**Linux/Mac (bash):**
```bash
bash run.sh
```

Ou manualmente:
```bash
python -m pip install -r requirements.txt
python src/main.py        # EDA + treino + salva modelo e artefatos
python src/inferencia.py  # inferencia em paciente novo (carrega modelo salvo)
```

Requisitos: Python 3.14 e os pacotes em `requirements.txt` (scikit-learn 1.8,
pandas 3.0, numpy 2.4, matplotlib 3.11, seaborn 0.13, joblib 1.5). Tudo usa
`SEMENTE (random_state) = 42` (reprodutivel).

---

## 7. Significado das saidas (`outputs/`)

| Arquivo | O que e |
|---------|---------|
| `eda_histogramas.png` | Distribuicao das variaveis continuas (EDA). |
| `eda_correlacao.png` | Mapa de calor de correlacao entre todas as variaveis. |
| `eda_boxplots.png` | Boxplots das continuas evidenciando os outliers (justificam o RobustScaler). |
| `eda_alvo_vs_caracteristica.png` | Creatinina serica por `DEATH_EVENT` (continua mais ligada ao desfecho). |
| `bic_por_componentes.png` | BIC x nº de componentes, com silhouette no 2º eixo (justifica k=2). |
| `clusters_pca.png` | Clusters projetados em PCA 2D (PC1+PC2 ~76% da variancia), com centroides. |
| `taxa_obito_por_cluster.png` | Taxa de DEATH_EVENT por cluster, em % (validacao externa). |
| `perfil_clusters.csv` | Medias das continuas + media de `time` + proporcao das binarias + taxa de obito + n, por cluster. |
| `metricas.json` | Nº de componentes, BIC, silhouette, Davies-Bouldin, Calinski-Harabasz, n e taxa de obito por cluster (+ proveniencia: variancia PCA, convergencia, sensibilidade de k). |
| `modelo_gmm.joblib` | Modelo GMM + RobustScaler + ordem das colunas + mapa de clusters (para inferencia sem vazamento). |
| `debate_Q1.md` | Debate das decisoes de modelagem (Questao -> Perspectivas A/B/C -> ADOTADA). |

---

## 8. Estrutura

```
Exercicio 1_heart_failure/
├── data/heart_failure_clinical_records_dataset.csv
├── src/
│   ├── main.py        # EDA + pre-proc + selecao de k + treino + caracterizacao
│   └── inferencia.py  # inferencia SEPARADA em paciente novo (carrega modelo salvo)
├── outputs/           # artefatos gerados
├── requirements.txt
├── run.ps1 / run.sh
├── README.md
└── outputs/debate_Q1.md  # debate das decisoes de modelagem
```

---

## 9. Limitacoes e proximos passos (honestidade autocritica)

Como explorador, prefiro apontar onde o projeto e fragil a maquiar resultados:

- **Sem split held-out.** O modelo treina nos 299 pacientes (sem reservar um
  conjunto de teste). A separacao treino -> paciente novo esta **correta e sem
  vazamento** (a inferencia so faz `escalonador.transform`), mas **nao ha
  avaliacao formal em dados retidos**. Proximo passo: reservar um held-out e
  medir a estabilidade da atribuicao.
- **k=2 depende do limiar do joelho do BIC (1%).** A escolha e defensavel e bem
  documentada, mas o resultado depende desse limiar. Reportamos a
  **sensibilidade** (`sensibilidade_k_por_limiar` em `metricas.json`) para 0.5%,
  1% e 2% justamente para mostrar a robustez da decisao.
- **Grupos desbalanceados (57 vs 242).** O Cluster 0 (alto estresse) e pequeno.
  E clinicamente plausivel (pacientes criticos sao minoria), mas digno de nota:
  metricas globais podem ser dominadas pelo grupo maior.
- **GMM ignora as binarias.** Coerente com a natureza gaussiana do modelo, mas
  joga fora informacao de comorbidades. Proximo passo: modelos para dados mistos
  (k-prototypes / distancia de Gower) para aproveitar `anaemia`, `diabetes`, etc.
  sem distorcer as densidades.
