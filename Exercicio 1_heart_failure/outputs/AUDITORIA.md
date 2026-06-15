# AUDITORIA — Q1 Heart Failure (Aluno 3: GaussianMixture + RobustScaler, binárias excluídas)

Auditor: avaliação semestral (não corretiva — apenas avalia e testa).
Data: 2026-06-14.

## Resumo do projeto
- **Algoritmo:** `GaussianMixture(covariance_type='full', n_init=10, random_state=42)`, k=2 componentes.
- **Scaler:** RobustScaler (mediana/IQR) nas variáveis contínuas.
- **Binárias:** **excluídas** do treino; usadas só para descrever os grupos (proporção por cluster).
- **`time`:** **excluído** do treino (correlação -0.527 com o desfecho → vazamento indireto); usado só para descrever.
- **Alvo:** `DEATH_EVENT` fora do treino; usado só para caracterizar (taxa de óbito por cluster).
- **Estilo:** explorador — EDA com histogramas e mapa de correlação, saída probabilística.

## PONTOS POSITIVOS
1. **Clustering legítimo e alvo fora do treino.** Features de treino = só as 6 contínuas; `DEATH_EVENT`, binárias e `time` ficam fora. Alvo entra apenas em `caracterizar_grupos` (taxa de óbito por cluster) e no gráfico `taxa_obito_por_cluster.png`.
2. **Decisão sobre `time` é a mais rigorosa das três versões.** Único projeto que **exclui** `time` do treino, com justificativa quantitativa: mede a correlação real (`-0.527`) na EDA e argumenta que `time` é variável de acompanhamento/desfecho (vazamento indireto), não característica de triagem. Decisão explícita no código, README e debate (Tema/Questão 2).
3. **Coerência conceitual algoritmo ↔ pré-processamento.** Excluir binárias é coerente com o GMM (densidades gaussianas; 0/1 não é gaussiano) — bem justificado no debate (Questão 1). RobustScaler é defendido pela presença real de outliers (CPK até 7861, creatinina até 9.4), mostrados na EDA.
4. **Saída probabilística aproveitada.** A inferência usa `predict_proba`, comunicando confiança da atribuição (99.7% no Cluster 1) — diferencial real do GMM, bem explorado.
5. **Melhor qualidade geométrica dos clusters das três versões.** Silhouette **0.4542** (vs 0.13 e 0.17 das outras), DB 1.8074, CH 68.86 — separação interna nitidamente superior, esperada por excluir as binárias ruidosas.
6. **Separação clínica por óbito presente.** 43.9% (Cluster 0) vs 29.3% (Cluster 1), com o grupo de maior óbito tendo CPK e creatinina sérica muito mais altas — interpretação clínica coerente.
7. **Seleção de k transparente.** Regra composta BIC-joelho + silhouette, com explicação de por que NÃO usar o argmin cego do BIC (k=7 → clusters degenerados n=6,4,4 e silhouette 0.10). Gráfico `bic_por_componentes.png` gerado.
8. **EDA e artefatos completos.** Histogramas, correlação, BIC, PCA, taxa de óbito, perfil consolidado (contínuas + time + proporção das binárias + taxa de óbito + n). README com números reais que batem com a execução.

## PONTOS NEGATIVOS / RISCOS
1. **Inferência sem split treino/teste, mas sem vazamento real.** O modelo treina nos 299 pacientes (sem split). Diferente do Aluno 2, este README **não** afirma haver split — descreve corretamente "treina e salva o modelo; inferência carrega o modelo salvo e só transforma". O paciente novo passa apenas por `escalonador.transform` (sem refit), então **não há vazamento** na inferência. Apenas registra-se que não existe avaliação em held-out. Risco baixo.
2. **k=2 escolhido por regra composta — leve dependência de limiar.** A regra do "joelho do BIC" (melhora relativa < 1%) define o joelho já em k=2, então a escolha recai no melhor silhouette dentro de k≤2, ou seja, k=2. É defensável e bem documentada, mas o resultado depende do limiar de 1%; com outro limiar a faixa mudaria. Não é defeito (a regra está explícita e justificada), apenas ponto de atenção metodológico.
3. **Desbalanceamento dos grupos (57 vs 242).** Um cluster é bem menor; clinicamente faz sentido (subgrupo de alto estresse renal/cardíaco), mas vale a observação. Sem impacto na validade.

## Checklist de critérios

| Item | Status | Observação |
|------|--------|------------|
| Clustering de verdade; DEATH_EVENT fora do treino, só caracteriza | OK | Treino só com 6 contínuas; alvo só na caracterização. |
| Binárias tratadas e justificadas | OK | Excluídas; justificado pela natureza gaussiana do GMM (debate Questão 1); usadas para descrever. |
| Sem vazamento (fit só no treino; paciente novo só transformado) | OK | Paciente novo só `transform`ado; sem refit. Treina nos 299 (sem split), mas README não alega split → sem inconsistência. |
| Todo o pré-processamento demonstrado | OK | EDA + RobustScaler explícitos; etapas impressas. |
| Inferência roda, paciente novo sem DEATH_EVENT, retorna grupo | OK | Cluster 1 com 99.7% de probabilidade; exit 0. |
| README explica como rodar e o que cada saída significa; números reais | OK | Tabela de saídas + resultados reais conferem com a execução. |
| Decisão sobre 'time' explícita e justificada | OK | **Excluído**, com correlação medida (-0.527); a decisão mais bem fundamentada das três. |

## VEREDITO: **APROVADO**

Projeto conceitualmente o mais cuidadoso: exclui `time` (evita vazamento indireto via proxy do desfecho), exclui binárias coerentemente com o GMM, usa RobustScaler justificado por outliers reais e aproveita a saída probabilística. Melhor qualidade geométrica (silhouette 0.4542) e separação clínica por óbito (43.9% vs 29.3%). Reprodutível, exit 0, sem vazamento. Nenhum defeito que comprometa a tarefa.

### Correções recomendadas (não bloqueantes)
1. (Opcional) Adicionar um split treino/teste (ou validação) para demonstrar atribuição em pacientes held-out, fechando o ciclo de avaliação — embora a separação treino → paciente novo já esteja correta.
2. (Opcional) Reportar a sensibilidade da escolha de k ao limiar do "joelho" (ex.: testar 0.5% e 2%) para mostrar robustez da decisão k=2.
