# -*- coding: utf-8 -*-
"""
Q3 - Black Friday | Multi-classificacao (3 alvos) com abordagem MULTI-SAIDA.

Aluno 3 (perfil EXPLORADOR): este script faz, em ETAPAS numeradas:
  1) CARGA (CSV real) ou GERACAO sintetica dos dados (com banner de aviso);
  2) ANALISE EXPLORATORIA (EDA): estatisticas descritivas + associacao
     feature->alvo + graficos seaborn (salvos em outputs/);
  3) PRE-PROCESSAMENTO com ColumnTransformer (one-hot p/ categoricas,
     scaling p/ numericas);
  4) SPLIT ESTRATIFICADO, estratificando pelo alvo principal (age_group);
  5) TREINO MULTI-SAIDA: MultiOutputClassifier(estimador_base) dentro de um
     Pipeline com o ColumnTransformer, treinando os 3 alvos JUNTOS (y 3 colunas);
  6) AVALIACAO por alvo: acuracia global, F1, matriz de confusao, e
     especificidade/sensibilidade por classe (one-vs-rest);
  7) PERSISTENCIA do modelo (joblib), metricas (JSON) e figuras (PNG).

Notas de ambiente:
  - SEMENTE = 42 (random_state) no gerador, no split e no modelo. Assinatura
    de reprodutibilidade do aluno: NUNCA RANDOM_STATE.
  - pandas 3.0 (copy-on-write): usar .copy() ao fatiar, sem chained assignment.
  - matplotlib: usar backend "Agg" ANTES de importar pyplot.
  - Tudo em PT-BR (apenas a API do scikit-learn/pandas fica em ingles).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend nao-interativo ANTES de importar pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Imports locais (funciona rodando "python src/main.py" a partir da raiz).
try:
    from gerar_dados import carregar_ou_gerar
    from metricas_utils import metricas_globais, metricas_por_classe
except ModuleNotFoundError:  # caso seja importado como pacote
    from src.gerar_dados import carregar_ou_gerar
    from src.metricas_utils import metricas_globais, metricas_por_classe

SEMENTE = 42

# Definicao de colunas conforme o esquema.
CARACTERISTICAS_CATEGORICAS = ["gender", "city_category"]
CARACTERISTICAS_NUMERICAS = [
    "occupation",
    "stay_years",
    "marital_status",
    "purchase_amount",
    "quantity",
]
COLUNAS_CARACTERISTICAS = CARACTERISTICAS_CATEGORICAS + CARACTERISTICAS_NUMERICAS
COLUNAS_ALVO = ["product_category", "payment_method", "age_group"]

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_SAIDA = RAIZ / "outputs"


def imprimir_titulo(texto: str) -> None:
    """Imprime um cabecalho de console com regua (assinatura visual do aluno)."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 120


def explorar_dados(df: pd.DataFrame) -> None:
    """EDA: imprime estatisticas e salva graficos seaborn."""
    imprimir_titulo("ETAPA 2 - ANALISE EXPLORATORIA (EDA)")
    print(f"\nFormato do conjunto: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nTipos de dados:")
    print(df.dtypes.to_string())
    print("\nValores ausentes por coluna:")
    print(df.isna().sum().to_string())

    print("\nEstatisticas descritivas (numericas):")
    print(df[CARACTERISTICAS_NUMERICAS].describe().round(2).to_string())

    print("\nDistribuicao dos ALVOS (contagem):")
    for alvo in COLUNAS_ALVO:
        print(f"\n-> {alvo}")
        print(df[alvo].value_counts().to_string())

    # Observacao do explorador sobre o (des)balanceamento: classes raras como
    # Roupas/Esportes (produto) ou Cartao_Debito (pagamento) tem poucas amostras
    # -> ja antecipamos que o modelo linear pode "ignora-las" na avaliacao.
    print(
        "\nObservacao do explorador: os 3 alvos sao DESBALANCEADOS "
        "(ha classes bem mais raras que outras);"
    )
    print(
        "  isso costuma penalizar o recall das classes minoritarias - "
        "vamos confirmar isso na ETAPA 6."
    )

    # ---- Associacao feature->ALVO ordenada por |valor| (alvos sao categoricos,
    #      entao codificamos cada alvo com pd.factorize e cruzamos com as
    #      numericas via corrwith). E uma APROXIMACAO de "qual feature mais
    #      'puxa' cada alvo", complementando o heatmap feature-vs-feature abaixo.
    print("\nAssociacao das caracteristicas NUMERICAS com cada ALVO (|corr|, desc):")
    for alvo in COLUNAS_ALVO:
        alvo_codificado = pd.Series(
            pd.factorize(df[alvo])[0], index=df.index, name=alvo
        )
        associacao = (
            df[CARACTERISTICAS_NUMERICAS]
            .corrwith(alvo_codificado)
            .sort_values(key=abs, ascending=False)
        )
        print(f"\n-> {alvo}")
        print(associacao.round(4).to_string())

    # ---- Grafico 1: distribuicao de cada alvo ----
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
    for eixo, alvo in zip(eixos, COLUNAS_ALVO):
        ordem = df[alvo].value_counts().index
        sns.countplot(
            data=df,
            x=alvo,
            order=ordem,
            ax=eixo,
            hue=alvo,
            palette="viridis",
            legend=False,
        )
        eixo.set_title(f"Distribuicao de {alvo}")
        eixo.set_xlabel("")
        eixo.set_ylabel("Contagem")
        eixo.tick_params(axis="x", rotation=45)
        # anotamos o valor de cada barra (assinatura de EDA do explorador).
        for container in eixo.containers:
            eixo.bar_label(container, fontsize=8, padding=2)
    fig.suptitle("EDA - Distribuicao dos 3 alvos", fontsize=14)
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / "eda_distribuicao_alvos.png", bbox_inches="tight")
    plt.close(fig)
    print("\n[grafico salvo] outputs/eda_distribuicao_alvos.png")

    # ---- Grafico 2: relacao valor de compra x categoria de produto ----
    fig, eixo = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=df,
        x="product_category",
        y="purchase_amount",
        ax=eixo,
        hue="product_category",
        legend=False,
    )
    eixo.set_title("Valor da compra por categoria de produto")
    eixo.set_xlabel("Categoria de produto")
    eixo.set_ylabel("Valor da compra (R$)")
    eixo.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / "eda_valor_por_categoria.png", bbox_inches="tight")
    plt.close(fig)
    print("[grafico salvo] outputs/eda_valor_por_categoria.png")

    # ---- Grafico 3: metodo de pagamento por categoria de cidade ----
    fig, eixo = plt.subplots(figsize=(10, 6))
    tabela = (
        pd.crosstab(df["city_category"], df["payment_method"], normalize="index") * 100
    )
    sns.heatmap(tabela, annot=True, fmt=".1f", cmap="viridis", ax=eixo)
    eixo.set_title("% de metodo de pagamento por categoria de cidade")
    eixo.set_xlabel("Metodo de pagamento")
    eixo.set_ylabel("Categoria de cidade")
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / "eda_pagamento_por_cidade.png", bbox_inches="tight")
    plt.close(fig)
    print("[grafico salvo] outputs/eda_pagamento_por_cidade.png")

    # ---- Grafico 4: correlacao das numericas ----
    fig, eixo = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        df[CARACTERISTICAS_NUMERICAS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=eixo,
    )
    eixo.set_title("Correlacao entre caracteristicas numericas")
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / "eda_correlacao_numericas.png", bbox_inches="tight")
    plt.close(fig)
    print("[grafico salvo] outputs/eda_correlacao_numericas.png")

    print("\n[INFO] EDA concluida (4 figuras eda_*.png em outputs/).")


def construir_pipeline() -> Pipeline:
    """Pipeline = ColumnTransformer + MultiOutputClassifier(LogisticRegression).

    Abordagem MULTI-SAIDA obrigatoria: um unico objeto envolve o estimador base
    e treina os 3 alvos juntos. predict_proba devolve uma LISTA (uma matriz por
    alvo), usada para o grau de certeza.
    """
    pre_processador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                CARACTERISTICAS_CATEGORICAS,
            ),
            ("numericas", StandardScaler(), CARACTERISTICAS_NUMERICAS),
        ]
    )

    # Nota: em scikit-learn 1.8 o parametro 'multi_class' foi removido; a
    # LogisticRegression trata multiclasse automaticamente (softmax/multinomial).
    estimador_base = LogisticRegression(
        max_iter=2000,
        random_state=SEMENTE,
    )

    modelo = Pipeline(
        steps=[
            ("preprocessamento", pre_processador),
            ("multi_saida", MultiOutputClassifier(estimador_base, n_jobs=-1)),
        ]
    )
    return modelo


def salvar_matriz_confusao(
    y_verdadeiro: pd.Series,
    y_previsto: pd.Series,
    classes: list[str],
    alvo: str,
) -> None:
    """Salva a matriz de confusao multiclasse como PNG.

    Salvamos DUAS versoes: a de contagem (fmt="d") e uma normalizada por LINHA
    (normalize="true", fmt=".2f") - esta ultima e o "recall visual", evidenciando
    as classes que o modelo colapsa (sens~0) que discutimos na leitura critica.
    """
    mc = confusion_matrix(y_verdadeiro, y_previsto, labels=classes)
    fig, eixo = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        mc,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=eixo,
    )
    eixo.set_title(f"Matriz de confusao - {alvo}")
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Verdadeiro")
    eixo.tick_params(axis="x", rotation=45)
    eixo.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(DIR_SAIDA / f"matriz_confusao_{alvo}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[grafico salvo] outputs/matriz_confusao_{alvo}.png")

    # Versao normalizada por linha (recall por classe na diagonal).
    mc_norm = confusion_matrix(
        y_verdadeiro, y_previsto, labels=classes, normalize="true"
    )
    fig, eixo = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        mc_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=eixo,
    )
    eixo.set_title(f"Matriz de confusao (normalizada por linha) - {alvo}")
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Verdadeiro")
    eixo.tick_params(axis="x", rotation=45)
    eixo.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(
        DIR_SAIDA / f"matriz_confusao_{alvo}_normalizada.png", bbox_inches="tight"
    )
    plt.close(fig)
    print(f"[grafico salvo] outputs/matriz_confusao_{alvo}_normalizada.png")


def avaliar_alvos(
    modelo: Pipeline,
    conjunto_teste_x: pd.DataFrame,
    conjunto_teste_y: pd.DataFrame,
) -> dict:
    """Avalia cada alvo: metricas globais + por classe + matriz de confusao.

    Tambem usa predict_proba (lista por alvo) para reportar a certeza media do
    modelo nas previsoes do conjunto de teste.
    """
    imprimir_titulo("ETAPA 6 - AVALIACAO POR ALVO")
    previsoes = modelo.predict(conjunto_teste_x)  # array (n, 3)
    previsoes = pd.DataFrame(previsoes, columns=COLUNAS_ALVO, index=conjunto_teste_y.index)

    # predict_proba: lista com 3 matrizes (uma por alvo).
    lista_probas = modelo.predict_proba(conjunto_teste_x)
    # Estimadores internos (um por alvo): fonte de verdade das classes (ja
    # ORDENADAS por .classes_). Preferimos isso a sorted(unique()) do teste para
    # nao omitir silenciosamente uma classe vista no treino e ausente no fold.
    estimadores = modelo.named_steps["multi_saida"].estimators_

    relatorio = {}
    for i, alvo in enumerate(COLUNAS_ALVO):
        y_real = conjunto_teste_y[alvo]
        y_prev = previsoes[alvo]
        classes = list(estimadores[i].classes_)

        globais = metricas_globais(y_real, y_prev, classes)
        por_classe = metricas_por_classe(y_real, y_prev, classes)

        # Certeza media = media do maior predict_proba por amostra.
        certeza_media = float(np.mean(np.max(lista_probas[i], axis=1)))

        salvar_matriz_confusao(y_real, y_prev, classes, alvo)

        relatorio[alvo] = {
            "classes": classes,
            "metricas_globais": globais,
            "certeza_media_predict_proba": round(certeza_media, 4),
            "metricas_por_classe": por_classe,
        }

        imprimir_titulo(f"ALVO: {alvo}")
        print(
            f"  Acuracia global : {globais['acuracia_global']:.4f}  "
            f"(acaso uniforme = {globais['acaso_uniforme']:.4f})"
        )
        print(f"  F1 macro        : {globais['f1_macro']:.4f}")
        print(f"  F1 ponderado    : {globais['f1_ponderado']:.4f}")
        print(f"  Certeza media   : {certeza_media:.4f}")
        print("  Sensibilidade / Especificidade por classe:")
        classes_colapsadas = []
        for classe, m in por_classe.items():
            marca = ""
            if m["especificidade"] >= 0.99 and m["sensibilidade"] <= 0.05:
                marca = "  <== classe colapsada (sens~0)"
                classes_colapsadas.append(classe)
            print(
                f"    - {classe:<16} sens={m['sensibilidade']:.3f}  "
                f"espec={m['especificidade']:.3f}  f1={m['f1']:.3f}  "
                f"(suporte={m['suporte']}){marca}"
            )
        if classes_colapsadas:
            print(
                "  Leitura critica (perfil explorador): classes com "
                "especificidade ~1.0 e sensibilidade ~0.0 "
                f"({', '.join(classes_colapsadas)})"
            )
            print(
                "    NAO sao bom resultado - o modelo linear quase nunca preve "
                "essas classes; a especificidade alta isolada engana."
            )
    return relatorio


def main() -> None:
    configurar_estilo()
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)

    imprimir_titulo(
        "Q3 - BLACK FRIDAY | MULTI-CLASSIFICACAO MULTI-SAIDA (3 alvos) (Aluno 3)"
    )

    # ======================================================================
    # ETAPA 1 - CARGA (CSV real ou geracao sintetica)
    # ======================================================================
    imprimir_titulo("ETAPA 1 - CARGA DOS DADOS")
    dados, eh_sintetico = carregar_ou_gerar(DIR_DADOS)

    # ======================================================================
    # ETAPA 2 - ANALISE EXPLORATORIA (EDA)
    # ======================================================================
    explorar_dados(dados)

    # ======================================================================
    # ETAPA 3 - PRE-PROCESSAMENTO / PIPELINE
    # ======================================================================
    imprimir_titulo("ETAPA 3 - PRE-PROCESSAMENTO / PIPELINE")
    # Separar caracteristicas (X) e rotulos (y com 3 colunas). Usamos .copy()
    # por causa do copy-on-write do pandas 3.0 (sem chained assignment).
    caracteristicas = dados[COLUNAS_CARACTERISTICAS].copy()
    rotulos = dados[COLUNAS_ALVO].copy()
    modelo = construir_pipeline()
    print(
        "[INFO] Pipeline montado: ColumnTransformer (one-hot + scaling) + "
        "MultiOutputClassifier(LogisticRegression)."
    )

    # ======================================================================
    # ETAPA 4 - SPLIT ESTRATIFICADO (por age_group)
    # ======================================================================
    imprimir_titulo("ETAPA 4 - SPLIT ESTRATIFICADO")
    conjunto_treino_x, conjunto_teste_x, conjunto_treino_y, conjunto_teste_y = (
        train_test_split(
            caracteristicas,
            rotulos,
            test_size=0.25,
            random_state=SEMENTE,
            stratify=rotulos["age_group"],
        )
    )
    print(
        f"[INFO] Split estratificado (por age_group): "
        f"treino={len(conjunto_treino_x)} | teste={len(conjunto_teste_x)}"
    )

    # ======================================================================
    # ETAPA 5 - TREINO MULTI-SAIDA (3 alvos JUNTOS)
    # ======================================================================
    imprimir_titulo("ETAPA 5 - TREINO MULTI-SAIDA")
    print("[INFO] Treinando modelo MULTI-SAIDA (3 alvos simultaneos)...")
    modelo.fit(conjunto_treino_x, conjunto_treino_y)
    print("[INFO] Treino concluido.")

    # ======================================================================
    # ETAPA 6 - AVALIACAO
    # ======================================================================
    relatorio = avaliar_alvos(modelo, conjunto_teste_x, conjunto_teste_y)

    # ======================================================================
    # ETAPA 7 - PERSISTENCIA (modelo + metricas)
    # ======================================================================
    imprimir_titulo("ETAPA 7 - PERSISTENCIA")
    caminho_modelo = DIR_SAIDA / "modelo_multisaida.joblib"
    joblib.dump(
        {
            "modelo": modelo,
            "colunas_caracteristicas": COLUNAS_CARACTERISTICAS,
            "colunas_alvo": COLUNAS_ALVO,
            "caracteristicas_categoricas": CARACTERISTICAS_CATEGORICAS,
            "caracteristicas_numericas": CARACTERISTICAS_NUMERICAS,
            # propaga o aviso de placeholder p/ quem rodar SO a inferencia.
            "treinado_em_sintetico": bool(eh_sintetico),
        },
        caminho_modelo,
        compress=3,
    )
    print(f"[modelo salvo] {caminho_modelo}")

    metricas_completas = {
        "aviso_placeholder": bool(eh_sintetico),
        "observacao": (
            "Metricas obtidas com DADOS SINTETICOS (placeholder)."
            if eh_sintetico
            else "Metricas obtidas com dataset real."
        ),
        "n_linhas": int(len(dados)),
        "n_treino": int(len(conjunto_treino_x)),
        "n_teste": int(len(conjunto_teste_x)),
        "estimador_base": "LogisticRegression(max_iter=2000, random_state=SEMENTE) | SEMENTE = 42",
        "abordagem": "MultiOutputClassifier (multi-saida) dentro de Pipeline+ColumnTransformer",
        "alvos": relatorio,
    }
    caminho_metricas = DIR_SAIDA / "metricas.json"
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_completas, f, ensure_ascii=False, indent=2)
    print(f"[arquivo salvo] {caminho_metricas}")

    # ----- Encerramento ritual: lista os artefatos gerados em outputs/ -----
    imprimir_titulo("ARTEFATOS GERADOS EM outputs/")
    for artefato in sorted(DIR_SAIDA.iterdir()):
        if artefato.is_file():
            print(f"  - {artefato.name}")

    imprimir_titulo("PIPELINE Q3 CONCLUIDO")
    if eh_sintetico:
        print("LEMBRETE: resultados sao de DADOS SINTETICOS (placeholder).")
    print("[OK] Pipeline Q3 finalizado com sucesso.")


if __name__ == "__main__":
    main()
