# MANUAL DE ESTILO DEFINITIVO — aluno_3

> Arquetipo: **CIENTISTA DE DADOS EXPLORADOR**. Referencia unica para refinar
> `q1_heart_failure`, `q2_wine_quality` e `q3_black_friday` ate que pareçam
> escritos pela MESMA pessoa — distinta do aluno_1 (didatico/iniciante) e do
> aluno_2 (engenheiro/modular).

---

## 1. Persona em 1 frase

O aluno_3 nao entrega codigo: entrega uma **INVESTIGACAO NARRADA em portugues do
Brasil**, organizada em ETAPAS numeradas, que olha os dados (EDA com seaborn),
levanta e compara hipoteses/trade-offs em voz alta, justifica o PORQUE de cada
decisao e fecha com inferencia separada que produz rankings/Top-N com grau de
certeza.

---

## 2. Estrutura de arquivos padrao (FIXA em todo `qN_*`)

```
qN_nome/
├── data/                       # CSVs de entrada (e sinteticos gerados)
├── outputs/                    # TODOS os artefatos: graficos .png, metricas.json,
│                               #   *.joblib, *.csv, debate_QN.md
├── src/
│   ├── main.py                 # pipeline NARRADO em ETAPAS numeradas (treina + salva)
│   ├── inferencia.py           # SEMPRE separado: carrega .joblib, preve (sem fit, sem vazamento)
│   └── <extra>.py              # auxiliar SO quando a complexidade justifica
│                               #   (ex.: gerar_dados.py, metricas_utils.py)
├── requirements.txt            # versoes fixadas com == + cabecalho-comentario
├── run.ps1                     # fluxo Windows
├── run.sh                      # fluxo Linux/Mac (espelho EXATO do .ps1)
└── README.md
```

REGRAS:
- `main.py` treina+salva; `inferencia.py` carrega+preve. **Nunca fundir os dois.**
- Extrair modulo auxiliar (`gerar_dados.py`, `metricas_utils.py`) apenas quando ha
  logica reutilizavel ou geracao de dados — modularidade a servico da NARRATIVA,
  nao da engenharia (isso e o que distingue do aluno_2).
- Imports locais com fallback `try/except ModuleNotFoundError` (padrao do Q3):

  ```python
  try:
      from gerar_dados import carregar_ou_gerar
  except ModuleNotFoundError:  # caso seja importado como pacote
      from src.gerar_dados import carregar_ou_gerar
  ```

---

## 3. Convencao de nomes — 100% PT-BR proprio

O leitor deve conseguir contar a historia da analise so lendo os nomes. Apenas a
API do scikit-learn/pandas permanece em ingles (inevitavel).

### Constantes (MAIUSCULAS)
- `SEMENTE = 42` ← **assinatura obrigatoria, NUNCA `RANDOM_STATE`.**
  [CORRIGIR Q2: `q2_wine_quality/src/main.py` usa `RANDOM_STATE` ~10x — destoa de
  Q1/Q3. Trocar para `SEMENTE`. Corrigir tambem o texto do README do Q2 que diz
  "random_state=42" para "SEMENTE (random_state) = 42".]
- Caminhos: padronizar UMA familia nos 3. Adotar o par do Q2/Q3:
  `RAIZ`, `DIR_DADOS`, `DIR_SAIDA` (via `Path(__file__).resolve().parent.parent`).
  [CORRIGIR Q1: usa `PASTA_BASE`/`PASTA_SAIDA`/`CAMINHO_DADOS` — unificar para
  `RAIZ`/`DIR_DADOS`/`DIR_SAIDA`.]
- Esquema de colunas em MAIUSCULA: `CARACTERISTICAS_NUMERICAS`,
  `CARACTERISTICAS_CATEGORICAS`, `COLUNAS_CARACTERISTICAS`, `COLUNAS_ALVO`,
  `COLUNAS_BINARIAS`, `COLUNA_ALVO`, `COLUNA_TEMPO`.

### Variaveis locais (snake_case PT-BR) — vocabulario canonico
`dados`, `caracteristicas`, `rotulos`, `conjunto_treino_x`, `conjunto_teste_x`,
`conjunto_treino_y`, `conjunto_teste_y`, `modelo_campeao`/`nome_campeao`,
`escalonador`, `matriz_escalonada`, `previsoes`, `y_prev`/`y_real`, `certeza`,
`relatorio`, `ranking`, `eixo`/`eixos` (NUNCA `ax`).
[CORRIGIR Q2: trocar `X_tr, y_tr, X_te, y_te`, `Xtr_f, yprev_f`, `acc`, `comp`,
`ax`, `p` por equivalentes PT-BR — eles contradizem o `conjunto_treino_X` usado no
MESMO arquivo.]

### Funcoes (verbo no infinitivo, PT-BR)
`carregar_dados`, `explorar`/`explorar_dados`/`analise_exploratoria`,
`preprocessar`, `construir_modelos`/`construir_pipeline`, `treinar_modelo_campeao`,
`avaliar_modelo`/`avaliar_alvos`, `caracterizar_grupos`, `salvar_matriz_confusao`,
`carregar_artefato`, `prever`/`prever_paciente`/`prever_venda`.
- **Helper de cabecalho de console:** PADRONIZAR o nome nos 3 como
  `imprimir_titulo(texto)` com regua `"=" * 70`.
  [CORRIGIR: Q2 usa `cabecalho`; Q3 inlina `print("=" * 70)` sem helper. Unificar.]
- **Ponto de entrada:** PADRONIZAR como `def main()` nos 3.
  [CORRIGIR Q3: usa `def principal()` — renomear para `main`.]

### Type hints (100% das assinaturas)
- Anotar TODAS as funcoes (params + retorno): `-> pd.DataFrame`, `-> dict`,
  `-> None`, `tipo: str | None = None` (Q2/Q3 ja fazem).
- Adicionar `from __future__ import annotations` no topo de TODO `.py` (Q3 ja usa).
  [CORRIGIR Q1: `main.py` e `inferencia.py` nao tem type hints nem o `__future__` —
  adicionar ambos.]

---

## 4. Voz dos comentarios e docstrings — o Narrador-Explorador

- **Docstring de modulo no topo de cada `main.py`** SEMPRE com este esqueleto:
  1. cabecalho-roteiro `"QX - <dataset> | <tarefa>"`;
  2. **fluxo numerado (1..N)** do que o script faz (carga → EDA → ... → persistencia);
  3. bloco final **"OBS de ambiente" / "Notas de ambiente"**: pandas 3.0
     copy-on-write (`.copy()`, sem chained assignment), backend matplotlib `"Agg"`
     ANTES do pyplot, `SEMENTE = 42`, "Tudo em PT-BR".
  [CORRIGIR: Q2/Q3 tem o fluxo no docstring mas nao numeram as ETAPAS no CORPO —
  ver item 4 do banner; Q1 ja numera. Unificar.]
- **Comentarios em 1a pessoa do plural, curiosos e OPINATIVOS** ("usamos",
  "escolhemos", "preferimos"). Respondem **POR QUE**, nao O QUE. Cada decisao
  nao-obvia (scaler, exclusao de coluna, metrica, k escolhido) ganha 1-3 linhas de
  justificativa. Exemplos reais a manter como tom:
  - `# aluno explorador gosta de seaborn bonitinho`
  - `# o "preco" desse ganho minimo de BIC sao clusters minusculos (n<7)`
  - `# RobustScaler: usa mediana e IQR -> resistente a OUTLIERS`
- **Avisos de vazamento/risco em CAIXA ALTA inline** (ja e assinatura, manter em
  todos): `NAO entra no treino`, `SEM vazamento`, `apenas .transform NUNCA .fit`,
  `fica FORA do treino`.
- **Frases-assinatura do narrador** nos prints de EDA e nas leituras de resultado:
  `Observacao do explorador:` e `Leitura critica (perfil explorador):`.
  [Hoje so em Q2/Q3 — levar tambem ao Q1, p.ex. comentando a separacao de
  mortalidade 43.9% vs 29.3% como "Leitura critica".]
- Docstrings de funcao curtas, explicando intencao e formato de retorno, com a
  formula quando relevante (ex.: `sensibilidade = TP/(TP+FN)`).

---

## 5. Estilo de prints / banners

- **Banner mestre de abertura** (1a linha do `main`):
  `"QX - <DATASET> | <TAREFA EM CAIXA ALTA> (Aluno 3)"`, via `imprimir_titulo`.
- **Etapas numeradas explicitas no console** e no corpo do codigo, com banner-
  comentario antes de cada uma:

  ```python
  # ======================================================================
  # ETAPA 2 - ANALISE EXPLORATORIA (EDA)
  # ======================================================================
  def analise_exploratoria(dados):
      imprimir_titulo("ETAPA 2 - ANALISE EXPLORATORIA (EDA)")
  ```

  E o marcador visual nº1 do explorador. [APLICAR a numeracao `ETAPA N` tambem em
  Q2 e Q3 — hoje tem secoes, mas sem o rotulo.]
- **Tags de I/O padronizadas nos 3 projetos:**
  - `[INFO]` para passos/progresso;
  - `[grafico salvo]` / `[arquivo salvo]` / `[modelo salvo]` para artefatos;
  - `[OK]` reservado ao encerramento.
  [Hoje convivem `[OK]` (Q2) e `[INFO]` (Q3) e as tags de artefato (Q1).
  Unificar o vocabulario igual nos tres.]
- **Encerramento ritual:** bloco final que LISTA os artefatos gerados percorrendo
  `outputs/` + frase `PIPELINE QX CONCLUIDO`. [Replicar em Q2, que hoje so imprime
  `[OK]`.]
- **Banner de alerta em moldura de `!`** para avisos fortes (placeholder, dados
  sinteticos, dado faltante) — padrao memoravel do `gerar_dados.py` do Q3:

  ```python
  linha = "!" * 78
  print(linha)
  print("!!" + " AVISO: DADOS SINTETICOS (PLACEHOLDER) ".center(74) + "!!")
  ...
  print(linha)
  ```
- Marcador de escolha em rankings/tabelas: `<== atribuido` / `<== VENCEDOR`.

---

## 6. Estilo de visualizacao / EDA — coracao da persona

- **Tema-assinatura unico nos 3:** `sns.set_theme(style="whitegrid", palette="deep")`.
  [CORRIGIR Q2: hoje so `style="whitegrid"` sem `palette` — adicionar `palette="deep"`.]
- **Gramatica cromatica consistente:**
  - barras/countplot → `viridis` ou `rocket`;
  - mapa de correlacao → `coolwarm` com `center=0`;
  - matriz de confusao → `Blues` (`fmt="d"`).
- **Toda `main.py` tem uma secao EDA que IMPRIME e SALVA:**
  - imprime: `describe()`, ausentes por coluna, distribuicao do alvo, e
    **correlacao das caracteristicas com o alvo ordenada por |valor|**;
  - salva **≥4 figuras `eda_*.png`** (histogramas/countplot, heatmap de
    correlacao, boxplot/relacao da variavel mais ligada ao alvo, etc.).
- **Sempre anotar valores nas barras** (`eixo.annotate` / `eixo.text` /
  `ax.annotate`) e **rotular eixos em PT-BR** (`set_xlabel`/`set_ylabel`).
- Backend `matplotlib.use("Agg")` ANTES de importar `pyplot`; `dpi` ~110-120;
  `tight_layout`/`bbox_inches="tight"`; sempre `plt.close(fig)` apos salvar.

---

## 7. requirements.txt + run.ps1 / run.sh

### requirements.txt
- Versoes SEMPRE fixadas com `==`, conjunto IDENTICO nos 3:

  ```
  scikit-learn==1.8.0
  pandas==3.0.2
  numpy==2.4.4
  scipy==1.17.1
  matplotlib==3.11.0
  seaborn==0.13.2
  joblib==1.5.3
  ```
- **Cabecalho-comentario** nomeando o projeto + ambiente testado:

  ```
  # Dependencias do mini-projeto QX - <Nome> (<tarefa>)
  # Versoes testadas no ambiente de desenvolvimento (Python 3.14.4).
  ```
  [CORRIGIR Q2: `requirements.txt` nao tem o cabecalho — adicionar (Q1 e Q3 tem).]

### run.ps1 / run.sh (PAR espelhado, mesmos passos e mensagens `==>`)
Estrutura obrigatoria nos dois:
1. modo estrito: `$ErrorActionPreference = "Stop"` (ps1) / `set -euo pipefail` (sh);
   [CORRIGIR: Q1 e Q2 usam so `set -e` no `.sh` — padronizar `set -euo pipefail`
   nos tres, como o Q3.]
2. ancorar no diretorio do script: `Set-Location -Path $PSScriptRoot` /
   `cd "$(dirname "$0")"`;
3. `python -m pip install -r requirements.txt`;
   [CORRIGIR Q3: `run.ps1` e `run.sh` PULAM a instalacao de requirements —
   readicionar para espelhar Q1/Q2.]
4. `python src/main.py`;
5. `python src/inferencia.py`;
6. mensagem final `==> Concluido. ... outputs/`.

---

## 8. Inferencia, debate e README (assinaturas intelectuais)

### Inferencia (`inferencia.py`)
- Sempre separada, com docstring marcando **"SEM vazamento"** e bloco `Uso:`.
- **Padronizar `argparse` nos 3** (hoje so o Q2 tem flags reais; Q3 documenta `Uso:`
  mas nao implementa; Q1 nao oferece opcoes). Levar opcoes a todos, ex.:
  `--modelo`, `--exemplo`, `--top`.
- **Ranking / Top-N como assinatura de saida:** onde houver `predict_proba`,
  imprimir o Top-N de classes com `%` e marcar a escolhida (`<== atribuido` /
  `<== VENCEDOR`). Generalizar o `ranking`/`Top-3 candidatos` do Q3 aos demais.

### debate_QX.md
- Manter em `outputs/` o arquivo `debate_QX.md` no formato:
  **"Questao N" → "Perspectiva A / B / C" → marcar "(ADOTADA)" → "Decisao" → tabela-
  sintese final**. E o diferencial mais forte da persona — exigir nos tres.

### README.md (narrativo, honesto, autocritico)
- Abertura declara o **perfil EXPLORADOR** e o que isso significa.
- Secoes fixas: Objetivo/Problema → Dataset (tabela de colunas por tipo) →
  Decisoes justificadas → **RESULTADOS REAIS** (tabelas com valores reais) →
  Comparacao/trade-offs ("X vs Y" + paragrafo "Trade-offs:") → Como rodar
  (PowerShell + Bash) → Significado das saidas → Estrutura em arvore.
- **Honestidade autocritica e marca:** apontar onde o modelo FALHA (recall ~0 em
  classes minoritarias, ex.: `Esportes`/`Roupas`/`Cartao_Debito`) e os proximos
  passos (ClassifierChain, reamostragem, modelo nao-linear). Nunca maquiar.
  Sempre rotular dado sintetico como **PLACEHOLDER** e repetir o aviso.

---

## MARCAS REGISTRADAS (idiomas-assinatura)

1. **`SEMENTE = 42`** como constante de seed em PT-BR — NUNCA `RANDOM_STATE`
   (corrigir Q2). A assinatura mais forte do aluno.
2. **Vocabulario 100% PT-BR proprio:** `dados`, `caracteristicas`, `rotulos`,
   `conjunto_treino_x/y`, `conjunto_teste_x/y`, `modelo_campeao`, `escalonador`,
   `matriz_escalonada`, `eixo`/`eixos` — zero `X_tr`/`y_te`/`acc`/`ax`.
3. **`main.py` como narrativa em ETAPAS numeradas** via banners-comentario
   (`# ETAPA 1 - CARGA`, `# ETAPA 2 - EDA`...) + docstring de modulo com fluxo
   1..N e bloco "OBS de ambiente".
4. **Frases-assinatura do narrador:** `Observacao do explorador:` e
   `Leitura critica (perfil explorador):`; comentarios opinativos em 1a pessoa do
   plural que respondem o PORQUE.
5. **Avisos de vazamento em CAIXA ALTA inline:** `NAO entra no treino`,
   `SEM vazamento`, `apenas .transform NUNCA .fit`.
6. **Layout fixo:** `src/main.py` (treina+salva) + `src/inferencia.py` (carrega+
   preve, sempre separado) + `data/` + `outputs/` + `requirements.txt` +
   `run.ps1` & `run.sh` + `README.md`.
7. **Tema seaborn fixo** `set_theme(whitegrid, deep)` + gramatica de cores
   `viridis`/`rocket` (barras), `coolwarm center=0` (correlacao), `Blues`
   (confusao); EDA com ≥4 `eda_*.png`, valores anotados nas barras e print de
   "correlacao com o alvo" ordenada por |valor|.
8. **Tags de I/O padronizadas:** `[INFO]` (passos),
   `[grafico salvo]`/`[arquivo salvo]`/`[modelo salvo]` (artefatos), `[OK]`
   (encerramento) + encerramento ritual listando `outputs/` e `PIPELINE QX
   CONCLUIDO`.
9. **Banner de alerta em moldura de `!`** para dados sinteticos/placeholder.
10. **`debate_QX.md`** no formato Questao N → Perspectiva A/B/C → (ADOTADA) →
    tabela-sintese; e README narrativo com "RESULTADOS REAIS", trade-offs e
    honestidade autocritica.
11. **Inferencia separada com `argparse`** + **Ranking/Top-N** com `%` e marcador
    `<== atribuido` / `<== VENCEDOR`, declarando "SEM vazamento" (so `.transform`).
12. **Caminhos via pathlib** ancorados no script (`RAIZ`/`DIR_DADOS`/`DIR_SAIDA`,
    `Path(__file__).resolve().parent.parent`) + `from __future__ import
    annotations` e type hints em todas as assinaturas.
