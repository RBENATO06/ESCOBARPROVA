# AUDITORIA - Q2 Wine Quality (Aluno 3)

Auditor: avaliacao semestral. Projeto: `aluno_3/q2_wine_quality`.
Modelos esperados: ExtraTrees, AdaBoost, MLP, + experimento de faixas.
**Confirmado** (ExtraTrees, AdaBoost, MLP, com experimento secundario de 3 faixas).

## Pontos positivos

- **Reproduz exatamente.** Treino (EDA + 3 modelos + faixas) e inferencia rodam com
  EXIT CODE 0; metricas batem com README, debate e `metricas.json`
  (vencedor ExtraTrees: acuracia 0.6946, F1-macro 0.3987, F1-weighted 0.6830).
- **Perfil EXPLORADOR cumprido:** EDA rica (estatisticas, valores ausentes,
  correlacao com o alvo ordenada por |valor|) + 5 graficos seaborn
  (`eda_distribuicao_quality.png`, `eda_quality_por_tipo.png`,
  `eda_correlacao.png`, `eda_alcohol_vs_quality.png` e
  `eda_correlacao_com_alvo.png`), todos regenerados.
- **Os 3 modelos exigidos** (ExtraTrees, AdaBoost, MLP), cada um em
  `Pipeline(StandardScaler -> clf)`. MLP com `max_iter=1000` + `early_stopping`
  (essencial para convergencia). Selecao por **F1-macro**.
- **Sem vazamento.** Scaler dentro do Pipeline; `fit` so no treino. Inferencia so
  aplica `transform`/`predict`. Inclui tambem o split estratificado das faixas com
  o mesmo `random_state`.
- **EXPERIMENTO DE FAIXAS bem feito (criterio 9):** binariza em 3 faixas
  (baixa<=5 / media=6 / alta>=7) via `pd.cut`, usa o MESMO modelo vencedor e o
  MESMO esquema de split estratificado (comparacao justa), e **quantifica o impacto
  no F1-macro: +0.3254 (0.3987 -> 0.7241)**. Discussao de pros/contras
  (granularidade perdida, fronteiras arbitrarias) presente no README e no debate.
- **Faixas reduzem o desbalanceamento** (baixa 2384 / media 2836 / alta 1277) e
  eliminam as classes com 0% de acerto — explicacao correta do porque o F1-macro
  quase dobra.
- **Dois modelos salvos** (`modelo_campeao_7classes.joblib` e
  `modelo_campeao_faixas.joblib`) com metadados; a inferencia escolhe o principal
  lendo `metricas.json` e aceita `--modelo` para forcar.
- **Split estratificado** com verificacao impressa da proporcao por classe
  treino vs teste (transparencia extra).
- **Classes raras 3 e 9 (recall 0.00) documentadas honestamente** e usadas como
  MOTIVACAO explicita do experimento de faixas — uso pedagogico exemplar do dado.

## Pontos negativos / riscos

- **Realismo do vinho de inferencia (RESOLVIDO):** os 3 vinhos de exemplo sao
  agora **FABRICADOS a mao** (valores fisico-quimicos plausiveis, porem NAO
  copiados de nenhuma linha dos `winequality-*.csv` — 0 correspondencias exatas).
  O comentario do codigo declara explicitamente que sao instancias ineditas, e o
  README repete o aviso. A ressalva anterior (Vinho 1 = 1a linha do CSV) NAO se
  aplica mais.
- **Grau de certeza documentado (RESOLVIDO):** a inferencia agora imprime um
  ranking Top-N com `%` e marca a vencedora (`<== atribuido`); a saida e o README
  explicam que a "probabilidade" e a FRACAO das 400 arvores que votaram na classe
  vencedora (confianca do comite, nao calibrada). Para os vinhos de FRONTEIRA
  fabricados a certeza fica em ~0,52-0,56 (comite dividido) — coerente e honesto.
- **`.joblib` sob controle (RESOLVIDO):** ambos os `joblib.dump` usam
  `compress=3`; tamanhos reais ~**55 MB** (`7classes`) e ~**44 MB** (`faixas`),
  bem abaixo dos ~279 MB / ~189 MB SEM compressao. Sem alterar previsoes/metricas.
- A acuracia da faixa "alta" (0.5977) continua a mais fraca — o proprio aluno
  reconhece que herda a raridade das notas altas. OK, documentado (secao
  "Limitacoes e proximos passos" do README).

## Checklist Q2

| # | Criterio | Status | Observacao |
|---|----------|--------|------------|
| 1 | Une tinto+branco preservando colunas; alvo=quality | OK | `pd.concat`; `tipo_vinho` so para EDA, NAO entra no X |
| 2 | >=3 classificadores + justificativa numerica do vencedor | OK | ExtraTrees/AdaBoost/MLP; vencedor por F1-macro com numeros |
| 3 | Acuracia global + por classe + F1 (macro/weighted) | OK | + classification_report (7 classes e faixas) |
| 4 | Matriz de confusao presente + F1 reportado | OK | `matriz_confusao_7classes.png`, `..._7classes_normalizada.png` e `..._3faixas.png` |
| 5 | Split estratificado + random_state fixo | OK | `stratify`, `SEMENTE (random_state) = 42`; proporcao impressa |
| 6 | Sem vazamento (fit so no treino, via Pipeline) | OK | StandardScaler dentro do Pipeline |
| 7 | Inferencia roda com vinho novo realista | OK | EXIT 0; 3 vinhos FABRICADOS (nao copiados do CSV); ranking Top-N com % e grau de certeza documentado |
| 8 | README explica execucao e saidas; numeros reais | OK | Tabelas 7 classes e faixas conferem com a execucao |
| 9 | Discute binarizacao em faixas e impacto no F1-macro | OK | +0.3254 quantificado; pros/contras no debate |

## VEREDITO: **APROVADO**

Projeto mais completo dos tres (EDA + experimento de faixas) e totalmente
reprodutivel. Nenhum item de checklist falha. As observacoes abaixo sao melhorias
recomendadas, **nao** correcoes obrigatorias.

### Sugestoes recomendadas — todas JA ATENDIDAS
1. **Imprecisao do comentario de inferencia — ATENDIDA:** os 3 vinhos sao agora
   FABRICADOS a mao (0 correspondencias exatas no dataset); o comentario e o
   README dizem isso explicitamente.
2. **Explicar o grau de certeza — ATENDIDA:** a saida e o README explicam que a
   "probabilidade" e a fracao de votos das 400 arvores (confianca do comite, nao
   calibrada); para vinhos de fronteira fica em ~0,52-0,56. Ranking Top-N
   acrescentado.
3. **`compress=3` nos `.joblib` — ATENDIDA:** ambos os dumps usam `compress=3`
   (~55 MB / ~44 MB).
