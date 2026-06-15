# TESTE — Q1 Heart Failure (Aluno 3)

Ambiente: Windows, PowerShell, `python` 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, seaborn 0.13.2.
Pontos de entrada (de `run.ps1`): `python src/main.py` e depois `python src/inferencia.py`.
Obs.: `run.ps1` começa com `python -m pip install -r requirements.txt`; como os pacotes já estão instalados, executei diretamente os dois scripts (equivalente ao restante do run.ps1).

## Comando 1 — Treino (EDA + GMM)
```
cd C:\Users\cezar\escobar_avaliacao_semestral\aluno_3\q1_heart_failure
python src/main.py
```
**EXIT CODE: 0**

Métricas obtidas (idênticas ao README e ao `metricas.json`):
- Algoritmo: GaussianMixture(covariance_type='full'); Scaler: RobustScaler
- Features de treino: 6 contínuas (binárias, time e DEATH_EVENT fora)
- Correlação medida time × DEATH_EVENT: **-0.527** (justifica excluir time)
- BIC/silhouette por k: k2=4852.68/0.4542, k3=4823.20/0.4400, k4=4758.73/0.2123, k5=4719.66/0.2446, k6=4717.19/0.1364, k7=4692.38/0.1022, k8=4736.12/0.1170
- k escolhido: **2** (BIC-joelho + silhouette); GMM convergiu (True)
- Silhouette: **0.4542** | Davies-Bouldin: **1.8074** | Calinski-Harabasz: **68.86**
- n por cluster: {0: 57, 1: 242}
- Taxa de óbito (caracterização): Cluster 0 = **43.9%**, Cluster 1 = **29.3%**
- Variância PCA: PC1=39.0%, PC2=37.5% (soma 76.5%)

Artefatos (re)gerados: eda_histogramas.png, eda_correlacao.png, bic_por_componentes.png, clusters_pca.png, taxa_obito_por_cluster.png, perfil_clusters.csv, metricas.json, modelo_gmm.joblib. Todos presentes.

## Comando 2 — Inferência
```
python src/inferencia.py
```
**EXIT CODE: 0**

### Screenshot textual da saída da inferência
```
======================================================================
INFERENCIA - paciente NOVO -> grupo (GaussianMixture)
======================================================================
Modelo carregado. Variaveis continuas esperadas: ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

Paciente novo (entrada):
  age                       : 60.0  (usada)
  creatinine_phosphokinase  : 250  (usada)
  ejection_fraction         : 38  (usada)
  platelets                 : 262000.0  (usada)
  serum_creatinine          : 1.1  (usada)
  serum_sodium              : 137  (usada)
  anaemia                   : 0  (ignorada)
  diabetes                  : 0  (ignorada)
  high_blood_pressure       : 0  (ignorada)
  sex                       : 1  (ignorada)
  smoking                   : 0  (ignorada)
  time                      : 130  (ignorada)
----------------------------------------------------------------------
GRUPO previsto para o paciente: cluster 1
Probabilidade (responsabilidade) de pertencer a cada grupo:
  cluster 0: 0.0028 (0.3%)
  cluster 1: 0.9972 (99.7%)  <== atribuido
----------------------------------------------------------------------
Confianca da atribuicao (maior probabilidade): 99.7%
```

## Conclusão do teste
Fluxo completo reprodutível, exit code 0 nos dois scripts. Métricas e saída batem com o README. Inferência mostra corretamente quais variáveis são usadas (6 contínuas) e quais são ignoradas (binárias, time). Paciente novo (marcadores baixos) atribuído ao Cluster 1 com 99.7% de probabilidade — saída probabilística do GMM funcionando.
