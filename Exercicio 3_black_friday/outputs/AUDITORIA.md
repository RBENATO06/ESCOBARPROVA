# AUDITORIA — Q3 Black Friday (Projeto C / aluno_3)

**Estratégia esperada:** `MultiOutputClassifier` + LogisticRegression/KNN.
**Estratégia entregue:** `Pipeline(ColumnTransformer + MultiOutputClassifier(LogisticRegression(max_iter=2000)))`, treinando os 3 alvos juntos (y com 3 colunas). **Confere** (estimador base = LogisticRegression).

Data da auditoria: 2026-06-14 · Avaliador: auditor/testador Q3.

---

## PONTOS POSITIVOS

- **Arquitetura correta e fiel ao esperado.** Abordagem multi-saída: um único `MultiOutputClassifier` envolvendo `LogisticRegression`, dentro de `Pipeline` com `ColumnTransformer` (one-hot + `StandardScaler`). Verificado no `.joblib`: 3 estimadores internos (1 por alvo), `feature_names_in_` = 7 features.
- **Perfil "explorador" bem executado:** EDA com estatísticas descritivas + 4 figuras seaborn (distribuição dos alvos, valor×categoria, pagamento×cidade, correlação numérica), todas geradas.
- **Sem vazamento, comprovado.** `X = dados[COLUNAS_CARACTERISTICAS]` (7 features); `y` = as 3 colunas-alvo. Nenhum alvo entra como feature. A venda nova na inferência monta o DataFrame só com features. Confirmado pelos metadados salvos no artefato (`colunas_caracteristicas` x `colunas_alvo`).
- **Especificidade e sensibilidade por classe corretas.** `metricas_utils.metricas_por_classe` aplica one-vs-rest e ainda salva TP/FN/FP/TN explícitos por classe. Recalculei sens/espec a partir de TP/FN/FP/TN do JSON: **0 divergências**.
- **Grau de certeza via `predict_proba`.** `MultiOutputClassifier.predict_proba` devolve lista (1 matriz por alvo); a inferência usa `max` por alvo e ainda imprime **top-3 candidatos** com %. Bom diferencial.
- **Fallback de dados implementado e bem sinalizado.** Lê `data/black_friday.csv`; senão gera sintético (seed 42) → `data/black_friday_sintetico.csv` + banner `!!` BEM VISÍVEL. README documenta o placeholder com destaque. Flag `aviso_placeholder: true` + campo `observacao` no JSON.
- **Reprodutibilidade total.** `random_state=42` em geração, split estratificado (por `age_group`, `test_size=0.25`) e modelo. Reexecutei do zero: métricas batem **exatamente** com o JSON versionado (acc product=0,373, payment=0,388, age=0,405).
- **Leitura crítica honesta.** README e debate destacam que classes com espec≈1,0 / sens≈0,0 (Esportes, Roupas, Cartao_Debito) NÃO são bom resultado — interpretação correta da especificidade isolada.
- **debate_Q3.md** cobre proba cru vs calibrado, sens/espec multiclasse e multi-saída vs separados, com observação correta de que o `MultiOutputClassifier` ajusta um estimador independente por alvo (não modela correlação).
- **Sistema funcional:** 3 vendas de exemplo, cada uma → 3 alvos com % de certeza e top-3. Primeiro exemplo reproduz o README (Beleza 42,34% / PIX 47,85% / 18-25 41,89%).

## PONTOS NEGATIVOS / RISCOS

- **`f1_score` sem `labels=classes` em `metricas_globais` (risco latente, sem impacto aqui).** O F1 macro/ponderado global usa o conjunto de rótulos inferido de `y_true ∪ y_pred`, não a lista explícita de classes. Verifiquei que, com a estratificação atual, **todas** as classes de cada alvo aparecem em treino e teste, então o resultado é correto neste dataset. **Recomendação (não bloqueante):** passar `labels=classes` para robustez caso, com dataset real/outro split, alguma classe falte no teste.
- **Estratificação só por `age_group`.** Para um problema multi-saída, isso não garante estratificação dos outros 2 alvos. No dataset atual todos os alvos mantiveram todas as classes em treino/teste (verificado), mas com dados reais desbalanceados poderia faltar classe rara de `product_category`/`payment_method` no teste. Limitação consciente, documentada como "estratificar pelo alvo principal".
- **LogisticRegression (linear) colapsa classes minoritárias** (Esportes/Roupas/Cartao_Debito com sens=0,00). Já reconhecido no README/debate como trade-off de clareza vs acurácia; não é defeito de pipeline.
- **Métricas modestas** (0,37–0,41), acima do acaso uniforme (0,143–0,20). Aceitável por serem dados placeholder.

---

## CHECKLIST

| Item | Status | Observação |
|------|--------|------------|
| Trata os 3 alvos | OK | y com 3 colunas, multi-saída |
| Por alvo: pipeline, acurácia global, acurácia por classe (matriz), F1 | OK | acc global; acc_classe ovr; F1 macro+ponderado+por classe; matriz PNG |
| Especificidade E sensibilidade por classe (one-vs-rest correto) | OK | TP/FN/FP/TN salvos; 0 divergências vs recálculo |
| Grau de certeza via `predict_proba` | OK | `max` por alvo + top-3 |
| Sem vazamento (alvos não viram feature; pré-proc só no treino) | OK | `feature_names_in_` = 7 features; venda só com features |
| Fallback CSV real → sintético com aviso visível + README | OK | Banner `!!` + flag JSON + README |
| Sistema funcionando (venda → 3 previsões com %) | OK | Inferência reproduz o README |
| README explica execução e saídas; números reais | OK | Tabelas conferem com o JSON |
| Reprodutibilidade (seed 42) | OK | Reexecução bate exatamente |
| Matriz de confusão PNG dos 3 alvos | OK | 3 PNGs (+ 4 figuras de EDA) |
| metricas.json com espec E sens por classe | OK | Presente para os 3 alvos |
| Modelos salvos (.joblib) | OK | 1 `modelo_multisaida.joblib` (~8 KB) + metadados |
| Dado sintético gerado em data/ | OK | `black_friday_sintetico.csv` |
| Abordagem multi-saída (esperada) | OK | `MultiOutputClassifier(LogisticRegression)` |

---

## VEREDITO: **APROVADO**

Projeto fiel à estratégia esperada (multi-saída com LogisticRegression), sem vazamento, com especificidade/sensibilidade por classe corretas (inclui TP/FN/FP/TN), EDA rica, fallback sinalizado, inferência com top-3 e reprodutibilidade comprovada.

**Correções obrigatórias:** nenhuma.

**Sugestões opcionais (não bloqueiam):** (1) passar `labels=classes` em `f1_score` dentro de `metricas_globais` para robustez com datasets onde alguma classe possa faltar no teste; (2) considerar estratificação multi-rótulo (ou ao menos verificação de cobertura de classes dos 3 alvos) ao usar dados reais; (3) estimador não-linear/`class_weight` para resgatar classes minoritárias colapsadas.
