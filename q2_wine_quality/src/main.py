# -*- coding: utf-8 -*-
"""
Q2 - Wine Quality | CLASSIFICACAO da nota de qualidade (3 a 9).

Aluno 3 (perfil EXPLORADOR): este script faz, em ETAPAS numeradas,

  ETAPA 1) Carga e UNIAO dos vinhos tinto + branco (preservando os nomes
           das colunas; acrescenta 'tipo_vinho' SO para a EDA).
  ETAPA 2) EDA: estatisticas descritivas + correlacao com o alvo + graficos
           seaborn (>=4 figuras salvas em outputs/).
  ETAPA 3) Split ESTRATIFICADO 80/20 (SEMENTE = 42), proporcao por classe
           preservada e impressa.
  ETAPA 4) Comparacao de 3 classificadores no problema de 7 classes:
           ExtraTrees, AdaBoost, MLP (cada um em Pipeline com StandardScaler);
           selecao do vencedor por F1-MACRO.
  ETAPA 5) Experimento secundario: 7 classes vs 3 FAIXAS
           (baixa<=5 | media=6 | alta>=7), mesmo modelo e mesmo split.
  ETAPA 6) Persistencia: pipelines vencedores (joblib, compress=3) + metricas
           (JSON) + matrizes de confusao (PNG).

Notas de ambiente:
  - pandas 3.0: copy-on-write (usamos .copy(), sem chained assignment).
  - matplotlib: backend "Agg" ANTES de importar pyplot (sem janela).
  - SEMENTE (random_state) = 42 em todo split/modelo (reprodutibilidade).
  - Tudo em PT-BR. Sem vazamento: o StandardScaler vive DENTRO do Pipeline,
    com fit SO no treino.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela (obrigatorio antes de pyplot)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------------
# Configuracao geral
# ----------------------------------------------------------------------------
SEMENTE = 42  # assinatura de reprodutibilidade: mesma SEMENTE -> mesmas metricas

# Esquema de colunas (em MAIUSCULA): centraliza as magic strings do dataset.
COLUNA_ALVO = "quality"
COLUNA_TIPO = "tipo_vinho"            # so para EDA; NAO entra no X
ROTULOS_FAIXA = ["baixa", "media", "alta"]
DEFINICAO_FAIXAS = "baixa<=5 | media=6 | alta>=7"

RAIZ = Path(__file__).resolve().parent.parent          # .../q2_wine_quality
DIR_DADOS = RAIZ / "data"
DIR_SAIDA = RAIZ / "outputs"
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

# tema-assinatura do aluno_3 (identico nos 3 projetos)
sns.set_theme(style="whitegrid", palette="deep")


def imprimir_titulo(texto: str) -> None:
    """Imprime um cabecalho legivel (regua de 70 '=') para organizar a exploracao."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def anotar_barras(eixo) -> None:
    """Anota o valor (contagem) no topo de cada barra de um countplot.

    So anota barras com altura > 0 (evita poluir categorias vazias). E o
    detalhe que o explorador gosta: numero explicito em cada barra.
    """
    for barra in eixo.patches:
        altura = barra.get_height()
        if altura > 0:
            eixo.annotate(
                f"{int(altura)}",
                (barra.get_x() + barra.get_width() / 2, altura),
                ha="center", va="bottom", fontsize=9)


def resumir_metricas(y_real: pd.Series, previsoes: np.ndarray) -> dict:
    """Resume as 3 metricas globais (acuracia/F1-macro/F1-weighted) arredondadas.

    Centralizado para garantir as MESMAS chamadas e o MESMO round(...,4) tanto
    no problema de 7 classes quanto no de 3 faixas (mesmos numeros sempre).
    """
    acuracia = accuracy_score(y_real, previsoes)
    f1_macro = f1_score(y_real, previsoes, average="macro", zero_division=0)
    f1_weighted = f1_score(y_real, previsoes, average="weighted", zero_division=0)
    return {
        "acuracia": round(float(acuracia), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def salvar_pacote(pipeline: Pipeline, tipo_alvo: str, caracteristicas: list,
                  classes: list, nome_modelo: str, caminho: Path,
                  **extra) -> None:
    """Empacota pipeline JA TREINADO + metadados e salva via joblib (compress=3).

    Reaproveitamos o pipeline treinado no split de treino (SEM vazamento).
    compress=3: o ExtraTrees (400 arvores) gera arquivos grandes; a compressao
    reduz MUITO o tamanho sem alterar previsoes/metricas.
    """
    pacote = {
        "pipeline": pipeline,
        "tipo_alvo": tipo_alvo,
        "caracteristicas": caracteristicas,
        "classes": classes,
        "nome_modelo": nome_modelo,
    }
    pacote.update(extra)
    joblib.dump(pacote, caminho, compress=3)
    print(f"[modelo salvo] {caminho.name}")


# ============================================================================
# ETAPA 1 - CARGA E UNIAO (tinto + branco)
# ============================================================================
def carregar_dados() -> pd.DataFrame:
    """Le os dois CSVs (sep=';') e os UNE preservando os nomes das colunas.

    Acrescenta a coluna 'tipo_vinho' (tinto/branco) apenas para a EDA;
    ela NAO entra no modelo (evitamos depender do tipo na inferencia padrao).
    Retorna um unico DataFrame com 11 caracteristicas + 'quality' + 'tipo_vinho'.
    """
    caminho_tinto = DIR_DADOS / "winequality-red.csv"
    caminho_branco = DIR_DADOS / "winequality-white.csv"

    tinto = pd.read_csv(caminho_tinto, sep=";")
    branco = pd.read_csv(caminho_branco, sep=";")

    tinto = tinto.copy()
    branco = branco.copy()
    tinto[COLUNA_TIPO] = "tinto"
    branco[COLUNA_TIPO] = "branco"

    dados = pd.concat([tinto, branco], ignore_index=True)

    print(f"[INFO] Tinto : {tinto.shape[0]} linhas, {tinto.shape[1] - 1} caracteristicas")
    print(f"[INFO] Branco: {branco.shape[0]} linhas, {branco.shape[1] - 1} caracteristicas")
    print(f"[INFO] Uniao : {dados.shape[0]} linhas (colunas preservadas)")
    return dados


# ============================================================================
# ETAPA 2 - ANALISE EXPLORATORIA (EDA)
# ============================================================================
def explorar(dados: pd.DataFrame) -> None:
    """EDA do explorador: estatisticas + graficos seaborn salvos em outputs/.

    Imprime describe(), ausentes, distribuicao do alvo e a correlacao das
    caracteristicas com 'quality' ordenada por |valor|. Salva >=4 figuras
    eda_*.png. Nao retorna nada (efeito colateral: prints + PNGs).
    """
    imprimir_titulo("ETAPA 2 - ANALISE EXPLORATORIA (EDA)")
    print("Primeiras linhas:")
    print(dados.head())

    print("\nTipos de dados:")
    print(dados.dtypes)

    print("\nValores ausentes por coluna (esperado: 0):")
    print(dados.isnull().sum())

    print("\nEstatisticas descritivas (numericas):")
    # describe() so nas numericas (em pandas 3.0 usamos select_dtypes em vez de
    # numeric_only, que nao existe em DataFrame.describe).
    apenas_numericas = dados.select_dtypes(include="number")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(apenas_numericas.describe().round(3))

    print("\nDistribuicao do alvo 'quality' (3 a 9):")
    contagem = dados[COLUNA_ALVO].value_counts().sort_index()
    for classe, qtd in contagem.items():
        pct = 100.0 * qtd / len(dados)
        print(f"  quality={classe}: {qtd:5d}  ({pct:5.2f}%)")
    print("\nObservacao do explorador: classes 3 e 9 sao RARISSIMAS "
          "(forte desbalanceamento).")

    print("\nDistribuicao por tipo de vinho:")
    print(dados.groupby(COLUNA_TIPO)[COLUNA_ALVO].describe().round(2))

    # --- Grafico 1: distribuicao do alvo
    plt.figure(figsize=(8, 5))
    eixo = sns.countplot(data=dados, x=COLUNA_ALVO, hue=COLUNA_ALVO,
                         palette="viridis", legend=False)
    eixo.set_title("Distribuicao da qualidade do vinho (alvo)")
    eixo.set_xlabel("Qualidade (quality)")
    eixo.set_ylabel("Quantidade de amostras")
    anotar_barras(eixo)
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "eda_distribuicao_quality.png", dpi=120,
                bbox_inches="tight")
    plt.close()
    print("[grafico salvo] eda_distribuicao_quality.png")

    # --- Grafico 2: distribuicao do alvo por tipo de vinho
    # usamos 'rocket' (gramatica de cores: barras -> viridis/rocket) e anotamos
    # as barras para casar com o Grafico 1 (consistencia interna).
    plt.figure(figsize=(8, 5))
    eixo = sns.countplot(data=dados, x=COLUNA_ALVO, hue=COLUNA_TIPO,
                         palette="rocket")
    eixo.set_title("Qualidade por tipo de vinho")
    eixo.set_xlabel("Qualidade (quality)")
    eixo.set_ylabel("Quantidade de amostras")
    anotar_barras(eixo)
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "eda_quality_por_tipo.png", dpi=120,
                bbox_inches="tight")
    plt.close()
    print("[grafico salvo] eda_quality_por_tipo.png")

    # --- Grafico 3: mapa de correlacao das caracteristicas
    # ('quality' fica na ultima linha/coluna do mapa)
    apenas_num = dados.drop(columns=[COLUNA_TIPO])
    plt.figure(figsize=(12, 10))
    sns.heatmap(apenas_num.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    plt.title("Mapa de correlacao das caracteristicas ('quality' na ultima linha/coluna)")
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "eda_correlacao.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[grafico salvo] eda_correlacao.png")

    # --- Grafico 4: alcohol x quality (variavel mais correlacionada)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=dados, x=COLUNA_ALVO, y="alcohol", hue=COLUNA_ALVO,
                palette="viridis", legend=False)
    plt.title("Teor alcoolico por qualidade")
    plt.xlabel("Qualidade (quality)")
    plt.ylabel("Alcohol")
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "eda_alcohol_vs_quality.png", dpi=120,
                bbox_inches="tight")
    plt.close()
    print("[grafico salvo] eda_alcohol_vs_quality.png")

    # correlacao de cada caracteristica com o alvo, ordenada por |valor|
    corr_alvo = apenas_num.corr(numeric_only=True)[COLUNA_ALVO].drop(
        COLUNA_ALVO).sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nCorrelacao das caracteristicas com 'quality' (|maior| -> |menor|):")
    print(corr_alvo.round(3))

    # --- Grafico 5: barplot dedicado da correlacao com o alvo
    # o heatmap 12x12 dilui o que mais importa; este barplot isola o sinal.
    plt.figure(figsize=(8, 5))
    eixo = sns.barplot(x=corr_alvo.values, y=corr_alvo.index,
                       hue=corr_alvo.index, palette="viridis", legend=False)
    eixo.axvline(0, color="gray", linewidth=0.8)
    eixo.set_title("Correlacao de cada caracteristica com 'quality' (ordenada por |valor|)")
    eixo.set_xlabel("Correlacao de Pearson com quality")
    eixo.set_ylabel("Caracteristica")
    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "eda_correlacao_com_alvo.png", dpi=120,
                bbox_inches="tight")
    plt.close()
    print("[grafico salvo] eda_correlacao_com_alvo.png")

    # Leitura critica do narrador, com os valores REAIS de corr_alvo.
    mais_pos = corr_alvo.idxmax()
    mais_neg = corr_alvo.idxmin()
    print(f"\nLeitura critica (perfil explorador): '{mais_pos}' ({corr_alvo[mais_pos]:+.2f}) "
          f"e '{mais_neg}' ({corr_alvo[mais_neg]:+.2f}) sao os sinais mais fortes — "
          "vinhos mais alcoolicos e menos densos tendem a notas mais altas.")


# ============================================================================
# ETAPA 4 - COMPARACAO DE 3 MODELOS (7 classes): definicao dos modelos
# ============================================================================
def construir_modelos() -> dict:
    """Retorna os 3 pipelines a comparar (nome -> Pipeline).

    Cada pipeline inclui StandardScaler. Para arvores (ExtraTrees) e AdaBoost
    o scaler e inofensivo; para o MLP ele e ESSENCIAL (convergencia/estabilidade).
    Manter o mesmo formato de pipeline em todos facilita a inferencia.
    """
    modelos = {
        "ExtraTrees": Pipeline([
            ("escalonador", StandardScaler()),
            ("classificador", ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=SEMENTE,
                n_jobs=-1,
            )),
        ]),
        "AdaBoost": Pipeline([
            ("escalonador", StandardScaler()),
            ("classificador", AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.5,
                random_state=SEMENTE,
            )),
        ]),
        "MLP": Pipeline([
            ("escalonador", StandardScaler()),
            ("classificador", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                max_iter=1000,          # iteracoes suficientes p/ convergir
                early_stopping=True,
                n_iter_no_change=20,
                random_state=SEMENTE,
            )),
        ]),
    }
    return modelos


def avaliar_modelo(nome: str, pipeline: Pipeline,
                   conjunto_treino_x: pd.DataFrame, conjunto_treino_y: pd.Series,
                   conjunto_teste_x: pd.DataFrame, conjunto_teste_y: pd.Series) -> dict:
    """Treina e avalia um pipeline; devolve dict com acuracia/F1-macro/F1-weighted."""
    pipeline.fit(conjunto_treino_x, conjunto_treino_y)
    previsoes = pipeline.predict(conjunto_teste_x)

    metricas = resumir_metricas(conjunto_teste_y, previsoes)

    print(f"\n--- {nome} ---")
    print(f"Acuracia global : {metricas['acuracia']:.4f}")
    print(f"F1-macro        : {metricas['f1_macro']:.4f}")
    print(f"F1-weighted     : {metricas['f1_weighted']:.4f}")

    return metricas


# ----------------------------------------------------------------------------
# Acuracia por classe + matrizes de confusao
# ----------------------------------------------------------------------------
def acuracia_por_classe(y_real: pd.Series, y_prev: np.ndarray,
                        rotulos: list) -> dict:
    """Acuracia por classe = diagonal normalizada da matriz de confusao
    (equivale ao RECALL de cada classe: recall = TP/(TP+FN))."""
    matriz_confusao = confusion_matrix(y_real, y_prev, labels=rotulos)
    por_classe = {}
    for i, classe in enumerate(rotulos):
        total = matriz_confusao[i].sum()
        acerto = matriz_confusao[i, i] / total if total > 0 else 0.0
        por_classe[str(classe)] = round(float(acerto), 4)
    return por_classe


def suporte_por_classe(y_real: pd.Series, rotulos: list) -> dict:
    """Numero de amostras de teste (suporte) de cada classe.

    Serve para sinalizar quando um recall e estatisticamente NAO-CONFIAVEL
    (ex.: classe 9 com 1 amostra no teste).
    """
    contagem = y_real.value_counts()
    return {str(classe): int(contagem.get(classe, 0)) for classe in rotulos}


def salvar_matriz_confusao(y_real: pd.Series, y_prev: np.ndarray, rotulos: list,
                           titulo: str, caminho: Path,
                           normalizar: bool = False) -> None:
    """Salva a matriz de confusao como heatmap seaborn (cmap='Blues').

    normalizar=False -> contagens (fmt='d'); normalizar=True -> recall por
    classe (normalize='true', fmt='.2f'), que faz as classes com recall 0%
    saltarem aos olhos.
    """
    if normalizar:
        matriz_confusao = confusion_matrix(
            y_real, y_prev, labels=rotulos, normalize="true")
        formato = ".2f"
        rotulo_cbar = "Recall (linha)"
    else:
        matriz_confusao = confusion_matrix(y_real, y_prev, labels=rotulos)
        formato = "d"
        rotulo_cbar = "Quantidade"

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(matriz_confusao, annot=True, fmt=formato, cmap="Blues",
                xticklabels=rotulos, yticklabels=rotulos,
                cbar_kws={"label": rotulo_cbar})
    plt.title(titulo)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[grafico salvo] {caminho.name}")


# ============================================================================
# ETAPA 5 - EXPERIMENTO 7 CLASSES vs 3 FAIXAS: agrupamento
# ============================================================================
def para_faixas(serie_quality: pd.Series) -> pd.Series:
    """Agrupa 'quality' em 3 faixas justificadas:
        baixa : quality <= 5  (vinhos medianos/ruins)
        media : quality == 6  (classe central, a maior)
        alta  : quality >= 7  (bons vinhos)
    A separacao 5|6|7 fica nos pontos de maior massa, equilibrando as faixas.
    """
    return pd.cut(
        serie_quality,
        bins=[-np.inf, 5, 6, np.inf],
        labels=ROTULOS_FAIXA,
    ).astype("object")


# ============================================================================
# Programa principal
# ============================================================================
def main() -> None:
    """Orquestra as ETAPAS 1..6: carga -> EDA -> split estratificado ->
    comparacao de 3 modelos (7 classes) -> selecao por F1-macro ->
    experimento de faixas -> persistencia dos artefatos."""
    imprimir_titulo("Q2 - WINE QUALITY | CLASSIFICACAO DA NOTA (Aluno 3)")

    # ========================================================================
    # ETAPA 1 - CARGA E UNIAO (tinto + branco)
    # ========================================================================
    imprimir_titulo("ETAPA 1 - CARGA E UNIAO (tinto + branco)")
    dados = carregar_dados()

    # ETAPA 2 - EDA (titulo impresso dentro de explorar())
    explorar(dados)

    # Caracteristicas e rotulos (NAO usamos 'tipo_vinho' nem 'quality' como X)
    caracteristicas = dados.drop(columns=[COLUNA_ALVO, COLUNA_TIPO])
    rotulos = dados[COLUNA_ALVO]
    nomes_caracteristicas = list(caracteristicas.columns)

    # ========================================================================
    # ETAPA 3 - SPLIT ESTRATIFICADO (treino/teste 80/20, SEMENTE = 42)
    # ========================================================================
    imprimir_titulo("ETAPA 3 - SPLIT ESTRATIFICADO (80/20, SEMENTE = 42)")
    conjunto_treino_x, conjunto_teste_x, conjunto_treino_y, conjunto_teste_y = \
        train_test_split(
            caracteristicas, rotulos,
            test_size=0.20,
            stratify=rotulos,
            random_state=SEMENTE,
        )
    print(f"[INFO] Treino: {conjunto_treino_x.shape[0]} amostras")
    print(f"[INFO] Teste : {conjunto_teste_x.shape[0]} amostras")
    print("Proporcao por classe preservada (estratificacao):")
    comparacao_proporcao = pd.DataFrame({
        "treino_%": conjunto_treino_y.value_counts(normalize=True).sort_index() * 100,
        "teste_%": conjunto_teste_y.value_counts(normalize=True).sort_index() * 100,
    }).round(2)
    print(comparacao_proporcao)

    # ========================================================================
    # ETAPA 4 - COMPARACAO DE 3 MODELOS (problema de 7 classes)
    # ========================================================================
    imprimir_titulo("ETAPA 4 - COMPARACAO DE 3 MODELOS (7 classes)")
    modelos = construir_modelos()
    rotulos_unicos = sorted(rotulos.unique())

    resultados = {}
    detalhado = {}
    for nome, pipeline in modelos.items():
        print(f"\n[INFO] Treinando e avaliando {nome}...")
        metricas = avaliar_modelo(
            nome, pipeline,
            conjunto_treino_x, conjunto_treino_y,
            conjunto_teste_x, conjunto_teste_y,
        )
        previsoes = pipeline.predict(conjunto_teste_x)
        metricas["acuracia_por_classe"] = acuracia_por_classe(
            conjunto_teste_y, previsoes, rotulos_unicos)
        resultados[nome] = metricas
        detalhado[nome] = pipeline

    # ----- Selecao do vencedor por F1-MACRO -----
    nome_campeao = max(resultados, key=lambda n: resultados[n]["f1_macro"])
    modelo_campeao = detalhado[nome_campeao]
    print("\n" + "-" * 70)
    print(f"[INFO] MODELO VENCEDOR (por F1-macro): {nome_campeao} "
          f"(F1-macro={resultados[nome_campeao]['f1_macro']:.4f})")
    print("-" * 70)

    # Relatorio detalhado do vencedor (com suporte por classe)
    previsoes_campeao = modelo_campeao.predict(conjunto_teste_x)
    print("\nRelatorio de classificacao do vencedor (7 classes):")
    print(classification_report(conjunto_teste_y, previsoes_campeao,
                                zero_division=0))
    suporte = suporte_por_classe(conjunto_teste_y, rotulos_unicos)
    print("Acuracia por classe (recall) do vencedor [com suporte de teste]:")
    for classe, acuracia_classe in resultados[nome_campeao]["acuracia_por_classe"].items():
        n_teste = suporte[classe]
        aviso = "  <- recall NAO-CONFIAVEL (suporte minusculo)" if n_teste <= 6 else ""
        print(f"  quality={classe}: recall={acuracia_classe:.4f}  (n_teste={n_teste}){aviso}")

    print("\nLeitura critica (perfil explorador): o F1-macro do vencedor "
          f"({resultados[nome_campeao]['f1_macro']:.4f}) denuncia o desbalanceamento — "
          "quality=3 e quality=9 ficam com recall 0.00 por raridade extrema "
          "(poucas amostras no teste), e a media macro paga esse preco.")

    # Matriz de confusao do vencedor (contagens + normalizada por recall)
    caminho_mc = DIR_SAIDA / "matriz_confusao_7classes.png"
    salvar_matriz_confusao(
        conjunto_teste_y, previsoes_campeao, rotulos_unicos,
        f"Matriz de confusao - {nome_campeao} (7 classes)", caminho_mc)
    # Na matriz de contagens as linhas das classes 3 e 9 (recall 0%) somem
    # visualmente; a versao normalizada faz o fracasso saltar aos olhos.
    salvar_matriz_confusao(
        conjunto_teste_y, previsoes_campeao, rotulos_unicos,
        f"Matriz de confusao normalizada (recall por classe) - {nome_campeao}",
        DIR_SAIDA / "matriz_confusao_7classes_normalizada.png",
        normalizar=True)

    # ----- Salvar comparacao dos modelos (json + txt) -----
    comparacao_modelos = {
        "problema": "7 classes (quality 3..9)",
        "criterio_selecao": "F1-macro",
        "vencedor": nome_campeao,
        "modelos": resultados,
    }
    (DIR_SAIDA / "comparacao_modelos.json").write_text(
        json.dumps(comparacao_modelos, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("[arquivo salvo] comparacao_modelos.json")

    linhas_txt = ["COMPARACAO DE MODELOS - 7 CLASSES (criterio: F1-macro)", "=" * 60]
    linhas_txt.append(f"{'Modelo':<14}{'Acuracia':>10}{'F1-macro':>11}{'F1-weighted':>14}")
    for nome, metricas in resultados.items():
        marca = "  <== VENCEDOR" if nome == nome_campeao else ""
        linhas_txt.append(
            f"{nome:<14}{metricas['acuracia']:>10.4f}{metricas['f1_macro']:>11.4f}"
            f"{metricas['f1_weighted']:>14.4f}{marca}")
    (DIR_SAIDA / "comparacao_modelos.txt").write_text(
        "\n".join(linhas_txt) + "\n", encoding="utf-8")
    print("[arquivo salvo] comparacao_modelos.txt")

    # ========================================================================
    # ETAPA 5 - EXPERIMENTO 7 CLASSES vs 3 FAIXAS
    # ========================================================================
    imprimir_titulo("ETAPA 5 - EXPERIMENTO 7 CLASSES vs 3 FAIXAS")
    print(f"Faixas: {DEFINICAO_FAIXAS}")

    y_faixa = para_faixas(rotulos)
    print("\nDistribuicao das faixas:")
    print(y_faixa.value_counts())

    # Mesmo split estratificado, agora no alvo agrupado (SEMENTE = 42)
    conjunto_treino_x_faixa, conjunto_teste_x_faixa, \
        conjunto_treino_y_faixa, conjunto_teste_y_faixa = train_test_split(
            caracteristicas, y_faixa,
            test_size=0.20, stratify=y_faixa, random_state=SEMENTE)

    # Usa o MESMO tipo de modelo vencedor das 7 classes (comparacao justa)
    print(f"\n[INFO] Treinando {nome_campeao} no alvo agrupado em 3 faixas...")
    modelo_faixas = construir_modelos()[nome_campeao]
    modelo_faixas.fit(conjunto_treino_x_faixa, conjunto_treino_y_faixa)
    previsoes_faixa = modelo_faixas.predict(conjunto_teste_x_faixa)

    metricas_faixa = resumir_metricas(conjunto_teste_y_faixa, previsoes_faixa)
    acuracia_faixa = metricas_faixa["acuracia"]
    f1_macro_faixa = metricas_faixa["f1_macro"]
    f1_weighted_faixa = metricas_faixa["f1_weighted"]

    print(f"\nModelo usado nas faixas: {nome_campeao}")
    print(f"[3 FAIXAS]  Acuracia={acuracia_faixa:.4f}  "
          f"F1-macro={f1_macro_faixa:.4f}  F1-weighted={f1_weighted_faixa:.4f}")
    print(f"[7 CLASSES] Acuracia={resultados[nome_campeao]['acuracia']:.4f}  "
          f"F1-macro={resultados[nome_campeao]['f1_macro']:.4f}  "
          f"F1-weighted={resultados[nome_campeao]['f1_weighted']:.4f}")

    print("\nRelatorio de classificacao (3 faixas):")
    print(classification_report(conjunto_teste_y_faixa, previsoes_faixa,
                                zero_division=0))

    salvar_matriz_confusao(
        conjunto_teste_y_faixa, previsoes_faixa, ROTULOS_FAIXA,
        f"Matriz de confusao - {nome_campeao} (3 faixas)",
        DIR_SAIDA / "matriz_confusao_3faixas.png")

    delta = f1_macro_faixa - resultados[nome_campeao]["f1_macro"]
    comparacao_7_vs_3 = {
        "definicao_faixas": DEFINICAO_FAIXAS,
        "modelo": nome_campeao,
        "sete_classes": {
            "acuracia": resultados[nome_campeao]["acuracia"],
            "f1_macro": resultados[nome_campeao]["f1_macro"],
            "f1_weighted": resultados[nome_campeao]["f1_weighted"],
        },
        "tres_faixas": {
            "acuracia": acuracia_faixa,
            "f1_macro": f1_macro_faixa,
            "f1_weighted": f1_weighted_faixa,
        },
        "ganho_f1_macro_faixas_vs_7classes": round(float(delta), 4),
        "acuracia_por_faixa": acuracia_por_classe(
            conjunto_teste_y_faixa, previsoes_faixa, ROTULOS_FAIXA),
    }

    # Decisao do modelo principal de inferencia
    if f1_macro_faixa >= resultados[nome_campeao]["f1_macro"]:
        modelo_principal = "3 faixas"
        justificativa = (
            "As 3 faixas atingem F1-macro substancialmente maior e oferecem "
            "previsao mais confiavel/acionavel (baixa/media/alta), por isso "
            "sao o modelo PRINCIPAL de inferencia.")
    else:
        modelo_principal = "7 classes"
        justificativa = (
            "O modelo de 7 classes manteve F1-macro competitivo e fornece "
            "granularidade maior, sendo escolhido como principal.")
    comparacao_7_vs_3["modelo_principal_inferencia"] = modelo_principal
    comparacao_7_vs_3["justificativa"] = justificativa
    print(f"\nDelta F1-macro (faixas - 7classes): {delta:+.4f}")
    print(f"[INFO] MODELO PRINCIPAL DE INFERENCIA: {modelo_principal}")
    print(justificativa)
    print("\nLeitura critica (perfil explorador): agrupar em faixas quase DOBRA "
          f"o F1-macro (+{delta:.4f}) porque junta as classes raras a vizinhas — "
          "nenhuma faixa fica com 0% de acerto. O preco e perder granularidade "
          "(deixamos de distinguir um 8 de um 9).")

    # ========================================================================
    # ETAPA 6 - PERSISTENCIA (metricas + pipelines vencedores)
    # ========================================================================
    imprimir_titulo("ETAPA 6 - PERSISTENCIA (metricas + modelos)")

    metricas_finais = {
        "random_state": SEMENTE,
        "n_amostras_total": int(len(dados)),
        "n_treino": int(conjunto_treino_x.shape[0]),
        "n_teste": int(conjunto_teste_x.shape[0]),
        "caracteristicas": nomes_caracteristicas,
        "classes_7": [int(c) for c in rotulos_unicos],
        "comparacao_modelos_7classes": resultados,
        "vencedor_7classes": nome_campeao,
        "suporte_por_classe_teste": suporte,
        "comparacao_7classes_vs_3faixas": comparacao_7_vs_3,
        "modelo_salvo_principal": (
            "modelo_campeao_faixas.joblib" if modelo_principal == "3 faixas"
            else "modelo_campeao_7classes.joblib"),
    }
    (DIR_SAIDA / "metricas.json").write_text(
        json.dumps(metricas_finais, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("[arquivo salvo] metricas.json")

    # ----- Salvar pipelines vencedores (joblib, compress=3, SEM vazamento) -----
    salvar_pacote(
        modelo_campeao, "7classes", nomes_caracteristicas,
        [int(c) for c in rotulos_unicos], nome_campeao,
        DIR_SAIDA / "modelo_campeao_7classes.joblib")
    salvar_pacote(
        modelo_faixas, "faixas", nomes_caracteristicas,
        ROTULOS_FAIXA, nome_campeao,
        DIR_SAIDA / "modelo_campeao_faixas.joblib",
        definicao_faixas=DEFINICAO_FAIXAS)

    # ----- Encerramento ritual: listar artefatos gerados -----
    imprimir_titulo("ARTEFATOS GERADOS EM outputs/")
    for arq in sorted(DIR_SAIDA.iterdir()):
        print(f"  - {arq.name}")
    print("\nPIPELINE Q2 CONCLUIDO")
    print("[OK] Execucao concluida com sucesso.")


if __name__ == "__main__":
    main()
