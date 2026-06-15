# -*- coding: utf-8 -*-
"""
Q1 - Heart Failure | AGRUPAMENTO (clustering) NAO supervisionado.

Aluno 3 (perfil EXPLORADOR): este script faz
  1) Carga dos dados (caminho via pathlib, ancorado no script);
  2) EDA: estatisticas descritivas + ausentes por coluna + correlacao com o
     alvo ordenada por |valor| + graficos seaborn (salvos em outputs/);
  3) Pre-processamento: separar continuas x binarias; decidir sobre 'time';
     escalonar SOMENTE as continuas com RobustScaler (resistente a outliers);
  4) Escolha do numero de componentes do GaussianMixture via BIC (regra do
     joelho) apoiada pelo silhouette;
  5) Treino do modelo campeao (GaussianMixture) + reordenacao DETERMINISTICA
     dos rotulos (Cluster 0 = alto estresse), para a narrativa nunca inverter;
  6) Caracterizacao dos grupos: medias por cluster, taxa de DEATH_EVENT por
     cluster (DEATH_EVENT fica FORA do treino, so descreve grupos) e perfil das
     binarias por cluster (% de fumantes, diabeticos, etc.);
  7) Visualizacao PCA 2D dos clusters;
  8) Persistencia em outputs/ (graficos, metricas.json, modelo + scaler).

OBS de ambiente:
  - pandas 3.0 usa copy-on-write; usamos .copy() e evitamos chained assignment.
  - matplotlib em backend "Agg" (sem janela grafica), ANTES de importar pyplot.
  - SEMENTE = 42 (random_state) em tudo (GMM/PCA) para reprodutibilidade.
  - Tudo em portugues do Brasil.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # backend nao-interativo: precisa vir ANTES do pyplot

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import joblib

# ----------------------------------------------------------------------
# Configuracoes globais
# ----------------------------------------------------------------------
SEMENTE = 42  # random_state em TODO lugar -> reprodutibilidade obrigatoria

# Caminhos ancorados no arquivo deste script (robusto ao diretorio de execucao).
RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_SAIDA = RAIZ / "outputs"
CAMINHO_DADOS = DIR_DADOS / "heart_failure_clinical_records_dataset.csv"
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

# Estilo dos graficos (aluno explorador gosta de seaborn bonitinho).
sns.set_theme(style="whitegrid", palette="deep")

# Esquema de colunas (em MAIUSCULA). As CONTINUAS sao as UNICAS que treinam o
# GMM; a ordem aqui e a ordem vista pelo RobustScaler -> NAO reordenar.
CARACTERISTICAS_NUMERICAS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]

# Colunas binarias (0/1): NAO entram no treino do GMM. Servem so para
# DESCREVER os grupos depois (ex.: % de fumantes por cluster).
COLUNAS_BINARIAS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

# Alvo clinico: fica FORA do treino. Usado SO para caracterizar os grupos.
COLUNA_ALVO = "DEATH_EVENT"

# 'time' = tempo de acompanhamento (follow-up) ate o desfecho/censura.
# DECISAO: excluir do treino (ver README/debate). E uma variavel de
# ACOMPANHAMENTO fortemente ligada ao desfecho (vazamento indireto), e nao
# uma caracteristica clinica intrinseca do paciente no momento da triagem.
COLUNA_TEMPO = "time"

# Limiar do "joelho" do BIC (melhora relativa < 1% -> curva ja achatou). E uma
# regra de modelagem citada no README/debate; por isso fica nomeada e visivel.
LIMIAR_JOELHO_BIC = 0.01


def imprimir_titulo(texto: str) -> None:
    """Imprime um cabecalho legivel para separar as etapas no console."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ======================================================================
# ETAPA 1 - CARGA DOS DADOS
# ======================================================================
def carregar_dados() -> pd.DataFrame:
    """Le o CSV de pacientes e devolve o DataFrame bruto (sem transformar)."""
    imprimir_titulo("ETAPA 1 - CARGA DOS DADOS")
    dados = pd.read_csv(CAMINHO_DADOS)
    print(f"[INFO] Arquivo lido: {CAMINHO_DADOS.name}")
    print(f"[INFO] Formato (linhas, colunas): {dados.shape}")
    print(f"[INFO] Valores ausentes (total): {int(dados.isna().sum().sum())}")
    print("\nPrimeiras linhas:")
    print(dados.head())
    return dados


# ======================================================================
# ETAPA 2 - ANALISE EXPLORATORIA (EDA)
# ======================================================================
def analise_exploratoria(dados: pd.DataFrame) -> None:
    """Explora os dados (descritivas, ausentes, correlacao com o alvo) e salva
    as figuras eda_*.png. Nao retorna nada: so descreve/desenha."""
    imprimir_titulo("ETAPA 2 - ANALISE EXPLORATORIA (EDA)")

    continuas = CARACTERISTICAS_NUMERICAS

    print("Estatisticas descritivas das colunas:")
    print(dados.describe().T)

    print("\nValores ausentes por coluna:")
    ausentes = dados.isna().sum()
    print(ausentes[ausentes > 0].to_dict() or "(nenhum ausente em nenhuma coluna)")

    print("\nDistribuicao do desfecho clinico (DEATH_EVENT) - so para contexto:")
    contagem_alvo = dados[COLUNA_ALVO].value_counts().sort_index()
    print(contagem_alvo.to_dict())
    print(
        f"Taxa global de obito: {dados[COLUNA_ALVO].mean():.3f} "
        f"({dados[COLUNA_ALVO].mean()*100:.1f}%)"
    )

    print("\nResumo das variaveis binarias (proporcao de '1'):")
    for coluna in COLUNAS_BINARIAS:
        print(f"  - {coluna:<22}: {dados[coluna].mean():.3f}")

    # Correlacao das caracteristicas com o alvo, ordenada por |valor|. Mesmo com
    # DEATH_EVENT FORA do treino, e util ver quem mais "puxa" o desfecho.
    correlacao_alvo = (
        dados.corr(numeric_only=True)[COLUNA_ALVO].drop(COLUNA_ALVO)
    )
    correlacao_alvo = correlacao_alvo.reindex(
        correlacao_alvo.abs().sort_values(ascending=False).index
    )
    print("\nCorrelacao das caracteristicas com DEATH_EVENT (ordenada por |valor|):")
    for coluna, valor in correlacao_alvo.items():
        print(f"  - {coluna:<26}: {valor:+.3f}")
    print(
        "Observacao do explorador: serum_creatinine (+) e ejection_fraction (-) "
        "sao os marcadores clinicos mais ligados ao obito; 'time' aparece muito "
        "forte por ser variavel de ACOMPANHAMENTO."
    )

    print(f"\nColunas continuas usadas na exploracao/treino: {continuas}")

    # --- Grafico 1: histogramas das continuas ---------------------------
    # Rotulos PT-BR amigaveis para os eixos (evita "Count" e nomes ingleses).
    rotulos_legiveis = {
        "age": "Idade (anos)",
        "creatinine_phosphokinase": "CPK (creatina-fosfoquinase)",
        "ejection_fraction": "Fracao de ejecao (%)",
        "platelets": "Plaquetas",
        "serum_creatinine": "Creatinina serica",
        "serum_sodium": "Sodio serico",
    }
    cor_barras = sns.color_palette("deep")[0]  # 1a cor do tema (no lugar de hardcode)
    n_colunas = 3
    n_linhas = int(np.ceil(len(continuas) / n_colunas))
    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(14, 4 * n_linhas))
    eixos = np.array(eixos).reshape(-1)
    ultimo_usado = -1
    for indice, coluna in enumerate(continuas):
        eixo = eixos[indice]
        sns.histplot(dados[coluna], kde=True, ax=eixo, color=cor_barras)
        eixo.set_title(f"Distribuicao: {rotulos_legiveis.get(coluna, coluna)}")
        eixo.set_xlabel(rotulos_legiveis.get(coluna, coluna))
        eixo.set_ylabel("Contagem")
        ultimo_usado = indice
    for j in range(ultimo_usado + 1, len(eixos)):  # desliga eixos sobrando
        eixos[j].axis("off")
    fig.suptitle("EDA - Histogramas das variaveis continuas", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    caminho_hist = DIR_SAIDA / "eda_histogramas.png"
    fig.savefig(caminho_hist, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[grafico salvo] {caminho_hist.name}")

    # --- Grafico 2: mapa de calor de correlacao -------------------------
    fig, eixo = plt.subplots(figsize=(11, 9))
    correlacao = dados.corr(numeric_only=True)
    sns.heatmap(
        correlacao,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=eixo,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    eixo.set_title("EDA - Mapa de correlacao (Pearson)", fontsize=14)
    eixo.set_xticklabels(eixo.get_xticklabels(), rotation=45, ha="right")
    eixo.set_yticklabels(eixo.get_yticklabels(), rotation=0)
    fig.tight_layout()
    caminho_corr = DIR_SAIDA / "eda_correlacao.png"
    fig.savefig(caminho_corr, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grafico salvo] {caminho_corr.name}")

    # --- Grafico 3: boxplots das continuas (evidencia os OUTLIERS) ------
    # Sao esses outliers que justificam o RobustScaler (mediana/IQR).
    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(14, 4 * n_linhas))
    eixos = np.array(eixos).reshape(-1)
    ultimo_usado = -1
    for indice, coluna in enumerate(continuas):
        eixo = eixos[indice]
        sns.boxplot(y=dados[coluna], ax=eixo, color=cor_barras)
        eixo.set_title(f"Outliers: {rotulos_legiveis.get(coluna, coluna)}")
        eixo.set_ylabel(rotulos_legiveis.get(coluna, coluna))
        ultimo_usado = indice
    for j in range(ultimo_usado + 1, len(eixos)):
        eixos[j].axis("off")
    fig.suptitle(
        "EDA - Boxplots das continuas (caudas longas -> RobustScaler)", fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    caminho_box = DIR_SAIDA / "eda_boxplots.png"
    fig.savefig(caminho_box, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grafico salvo] {caminho_box.name}")

    # --- Grafico 4: alvo x caracteristica mais ligada ao obito ----------
    # serum_creatinine e a continua de maior |correlacao| com DEATH_EVENT.
    fig, eixo = plt.subplots(figsize=(7.5, 5.5))
    sns.boxplot(
        x=dados[COLUNA_ALVO].astype(str),
        y=dados["serum_creatinine"],
        hue=dados[COLUNA_ALVO].astype(str),
        legend=False,
        ax=eixo,
        palette="rocket",
    )
    eixo.set_xlabel("DEATH_EVENT (0 = sobreviveu, 1 = obito)")
    eixo.set_ylabel(rotulos_legiveis["serum_creatinine"])
    eixo.set_title("Creatinina serica por desfecho (alvo so descreve)")
    fig.tight_layout()
    caminho_alvo = DIR_SAIDA / "eda_alvo_vs_caracteristica.png"
    fig.savefig(caminho_alvo, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grafico salvo] {caminho_alvo.name}")


# ======================================================================
# ETAPA 3 - PRE-PROCESSAMENTO
# ======================================================================
def preprocessar(
    dados: pd.DataFrame, continuas: list[str]
) -> tuple[np.ndarray, RobustScaler]:
    """Escalona SOMENTE as continuas com RobustScaler. Retorna
    (matriz_escalonada, escalonador) na ORDEM exata de `continuas`."""
    imprimir_titulo("ETAPA 3 - PRE-PROCESSAMENTO")

    # Coerencia: as colunas continuas existem no CSV e estao nesta ordem.
    assert set(continuas).issubset(dados.columns), "Continuas ausentes no CSV."

    # As caracteristicas de treino sao SOMENTE as continuas (binarias fora,
    # 'time' fora, alvo fora). NAO entra binaria/time/alvo no treino.
    caracteristicas = dados[continuas].copy()
    print(f"[INFO] Caracteristicas (continuas) no treino: {list(caracteristicas.columns)}")
    print(f"[INFO] Formato da matriz de treino: {caracteristicas.shape}")

    # RobustScaler: usa mediana e IQR -> resistente a OUTLIERS. O dataset tem
    # caudas longas (creatinine_phosphokinase, platelets, serum_creatinine),
    # entao o RobustScaler e mais adequado que o StandardScaler aqui.
    escalonador = RobustScaler()
    matriz_escalonada = escalonador.fit_transform(caracteristicas)
    print("[INFO] RobustScaler aplicado (mediana/IQR) -> robusto a outliers.")

    return matriz_escalonada, escalonador


# ======================================================================
# ETAPA 4 - ESCOLHA DO NUMERO DE COMPONENTES (BIC + silhouette)
# ======================================================================
def escolher_numero_componentes(
    matriz_escalonada: np.ndarray, max_componentes: int = 8
) -> dict:
    """Escolhe k via regra composta: 'joelho' do BIC (melhora relativa < 1%) e,
    na faixa estabilizada, o melhor silhouette. Retorna dict
    {"k", "bic", "silhueta", "sensibilidade"} (silhueta = (b-a)/max(a,b))."""
    imprimir_titulo("ETAPA 4 - ESCOLHA DO NUMERO DE COMPONENTES (BIC + silhouette)")

    faixa = range(2, max_componentes + 1)  # >=2 para poder medir silhouette
    valores_bic = []
    valores_silhueta = []

    for k in faixa:
        modelo = GaussianMixture(
            n_components=k,
            covariance_type="full",  # covariancia eliptica (cada cluster sua forma)
            random_state=SEMENTE,
            n_init=10,  # varias inicializacoes -> resultado mais estavel
        )
        rotulos = modelo.fit_predict(matriz_escalonada)
        bic = modelo.bic(matriz_escalonada)
        silhueta = silhouette_score(matriz_escalonada, rotulos)
        valores_bic.append(bic)
        valores_silhueta.append(silhueta)
        print(f"  k={k}: BIC={bic:10.2f} | silhouette={silhueta:.4f}")

    lista_k = list(faixa)
    valores_bic = np.array(valores_bic)
    valores_silhueta = np.array(valores_silhueta)

    # DECISAO DE SELECAO (explorador, justificada):
    # O BIC continua caindo de forma quase plana a partir de k>=4 (a diferenca
    # entre k=4 e o minimo absoluto e pequena), e o "preco" desse ganho minimo
    # de BIC sao clusters minusculos (n<7), sem utilidade clinica. Pegar o
    # argmin cego do BIC gera grupos degenerados. Por isso usamos uma regra
    # composta e transparente:
    #   1) O BIC pelo MENOR valor indicaria o k mais complexo (referencia).
    #   2) Mas escolhemos o k DENTRO da faixa onde o BIC ja "estabilizou"
    #      (ganho marginal pequeno) que MAXIMIZA a silhouette -> grupos
    #      geometricamente mais separados e interpretaveis.
    indice_bic_min = int(np.argmin(valores_bic))
    k_bic_min = lista_k[indice_bic_min]
    print(
        f"\nReferencia: o MENOR BIC isolado ocorre em k={k_bic_min} "
        f"(BIC={valores_bic[indice_bic_min]:.2f}, "
        f"silhouette={valores_silhueta[indice_bic_min]:.4f})."
    )
    print(
        "  -> Porem esse k gera clusters minusculos e silhouette baixa; "
        "ganho de BIC e marginal. Aplicamos regra composta BIC+silhouette."
    )

    def k_para_limiar(limiar: float) -> int:
        """k escolhido reaplicando a regra (joelho do BIC + melhor silhueta)
        para um dado `limiar` de melhora relativa. So para sensibilidade."""
        k_joelho_local = lista_k[0]
        for i in range(1, len(lista_k)):
            melhora = (valores_bic[i - 1] - valores_bic[i]) / abs(valores_bic[i - 1])
            if melhora < limiar:
                k_joelho_local = lista_k[i - 1]
                break
        else:
            k_joelho_local = k_bic_min
        indices = [i for i, k in enumerate(lista_k) if k <= k_joelho_local]
        return lista_k[max(indices, key=lambda i: valores_silhueta[i])]

    # "Joelho" do BIC: primeiro k cuja melhora relativa cai abaixo do limiar
    # adotado (1%). A partir dele o BIC esta praticamente plano.
    k_joelho = k_para_limiar(LIMIAR_JOELHO_BIC)

    # Na faixa "estabilizada" (de k=2 ate o joelho), escolhemos o melhor
    # silhouette -> solucao parcimoniosa e interpretavel.
    indices_faixa = [i for i, k in enumerate(lista_k) if k <= k_joelho]
    indice_melhor = max(indices_faixa, key=lambda i: valores_silhueta[i])
    melhor_k = lista_k[indice_melhor]
    print(
        f"  -> 'Joelho' do BIC em k={k_joelho}. Dentro de k<= {k_joelho}, "
        f"o melhor silhouette esta em k={melhor_k}."
    )
    print(
        f"\n-> Numero de componentes ESCOLHIDO: k={melhor_k} "
        f"(BIC={valores_bic[indice_melhor]:.2f}, "
        f"silhouette={valores_silhueta[indice_melhor]:.4f})."
    )

    # Sensibilidade ao limiar do joelho (sem mudar o adotado de 1%): mostra que
    # a decisao k=2 e robusta a pequenas variacoes da regra. So informa.
    sensibilidade = {
        f"{limiar:.3f}": int(k_para_limiar(limiar)) for limiar in (0.005, 0.01, 0.02)
    }
    print(
        "Leitura critica (perfil explorador): sensibilidade de k ao limiar do "
        f"joelho -> {sensibilidade} (mantemos 1% como adotado)."
    )

    # --- Grafico: BIC x numero de componentes (com silhouette no 2o eixo) --
    fig, eixo1 = plt.subplots(figsize=(9, 5.5))
    cor_bic = "tab:blue"
    eixo1.plot(list(faixa), valores_bic, "o-", color=cor_bic, label="BIC")
    eixo1.set_xlabel("Numero de componentes (k)")
    eixo1.set_ylabel("BIC (menor = melhor)", color=cor_bic)
    eixo1.tick_params(axis="y", labelcolor=cor_bic)
    eixo1.axvline(melhor_k, color="gray", linestyle="--", alpha=0.7)
    eixo1.annotate(
        f"escolhido k={melhor_k}",
        xy=(melhor_k, valores_bic[indice_melhor]),
        xytext=(melhor_k + 0.2, valores_bic[indice_melhor]),
        color="black",
    )

    eixo2 = eixo1.twinx()
    cor_sil = "tab:red"
    eixo2.plot(
        list(faixa), valores_silhueta, "s--", color=cor_sil, label="silhouette"
    )
    eixo2.set_ylabel("Silhouette (maior = melhor)", color=cor_sil)
    eixo2.tick_params(axis="y", labelcolor=cor_sil)

    fig.suptitle("Selecao do numero de componentes: BIC x silhouette")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    caminho_bic = DIR_SAIDA / "bic_por_componentes.png"
    fig.savefig(caminho_bic, dpi=120)
    plt.close(fig)
    print(f"[grafico salvo] {caminho_bic.name}")

    return {
        "k": melhor_k,
        "bic": float(valores_bic[indice_melhor]),
        "silhueta": float(valores_silhueta[indice_melhor]),
        "sensibilidade": sensibilidade,
    }


# ======================================================================
# ETAPA 5 - TREINO DO MODELO CAMPEAO
# ======================================================================
def treinar_modelo_campeao(
    matriz_escalonada: np.ndarray, melhor_k: int
) -> tuple[GaussianMixture, np.ndarray]:
    """Treina o GaussianMixture campeao com k componentes e devolve
    (modelo, rotulos). Reordena os clusters em seguida (ver reordenar_clusters)."""
    imprimir_titulo("ETAPA 5 - TREINO DO MODELO CAMPEAO (GaussianMixture)")
    modelo_campeao = GaussianMixture(
        n_components=melhor_k,
        covariance_type="full",
        random_state=SEMENTE,
        n_init=10,
    )
    rotulos = modelo_campeao.fit_predict(matriz_escalonada)
    print(f"[INFO] Modelo treinado com k={melhor_k} componentes.")
    print(f"[INFO] Convergiu? {modelo_campeao.converged_}")
    distribuicao = pd.Series(rotulos).value_counts().sort_index()
    print("Tamanho de cada grupo (n por cluster, ANTES de reordenar):")
    print(distribuicao.to_dict())
    return modelo_campeao, rotulos


def reordenar_clusters(
    rotulos: np.ndarray,
    chave: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Remapeia os ids dos clusters por uma `chave` DECRESCENTE (uma medida por
    componente), tornando os ids ESTAVEIS entre versoes do sklearn. Retorna
    (rotulos_remapeados, mapa) onde mapa = {id_componente_GMM: id_narrativa}.

    O indice de componente do GMM (0,1,...) NAO e estavel entre plataformas; sem
    isto, uma atualizacao poderia inverter os ids e contradizer README/CSV/
    inferencia SILENCIOSAMENTE. Reordenacao e cosmetica: nao muda a particao
    (silhouette/BIC/DB/CH inalterados), so quem recebe o id 0 vs 1. O `mapa` e
    salvo no joblib para a inferencia aplicar a MESMA convencao.

    Aqui a `chave` e a media de serum_creatinine por cluster: o grupo de MAIOR
    estresse renal/cardiaco vira o Cluster 0 (narrativa publicada).
    """
    ordem = np.argsort(chave)[::-1]  # maior chave -> id 0
    remapeamento = {int(antigo): int(novo) for novo, antigo in enumerate(ordem)}
    rotulos_novos = np.array([remapeamento[int(r)] for r in rotulos])
    print(
        "[INFO] Clusters reordenados por serum_creatinine medio (desc): "
        f"id 0 = maior estresse. Mapeamento {remapeamento}."
    )
    return rotulos_novos, remapeamento


# ======================================================================
# ETAPA 6 - CARACTERIZACAO DOS GRUPOS
# ======================================================================
def caracterizar_grupos(
    dados: pd.DataFrame, continuas: list[str], rotulos: pd.Series
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Descreve os grupos (DEATH_EVENT so descreve, NAO entra no treino).
    Retorna (taxa_obito, contagem, perfil_consolidado) e salva o CSV de perfil."""
    imprimir_titulo("ETAPA 6 - CARACTERIZACAO DOS GRUPOS")

    # Trabalhamos em uma copia para anexar o rotulo do cluster (CoW seguro).
    dados_rotulados = dados.copy()
    dados_rotulados["cluster"] = np.asarray(rotulos)

    # Um UNICO particionamento -> todos os perfis vem do mesmo groupby (DRY:
    # deixa explicito que continuas, alvo, binarias e tempo descrevem a MESMA
    # particao).
    grupos = dados_rotulados.groupby("cluster")
    perfil_continuas = grupos[continuas].mean()
    taxa_obito = grupos[COLUNA_ALVO].mean()
    contagem = grupos.size()
    perfil_binarias = grupos[COLUNAS_BINARIAS].mean()
    perfil_tempo = grupos[COLUNA_TEMPO].mean()

    print("\nMedias das variaveis continuas por cluster:")
    print(perfil_continuas)

    print("\nTaxa de DEATH_EVENT (obito) por cluster:")
    for c in taxa_obito.index:
        print(
            f"  cluster {c}: taxa={taxa_obito[c]:.3f} "
            f"({taxa_obito[c]*100:.1f}%) | n={int(contagem[c])}"
        )

    print("\nPerfil das variaveis binarias por cluster (proporcao de '1'):")
    print(perfil_binarias)

    # Leitura critica: o alvo separa os grupos mesmo FORA do treino -> sinal de
    # que a estrutura nao supervisionada captura algo clinicamente real.
    print(
        "\nLeitura critica (perfil explorador): o Cluster 0 (alto estresse renal/"
        f"cardiaco) tem taxa de obito {taxa_obito.iloc[0]*100:.1f}% contra "
        f"{taxa_obito.iloc[1]*100:.1f}% do Cluster 1 (estavel). Como DEATH_EVENT "
        "NAO entrou no treino, essa separacao e evidencia de que os clusters "
        "capturam sinal clinico real, nao um artefato do alvo."
    )

    # --- Salva o perfil consolidado em CSV (entregavel 'perfil_clusters') --
    perfil_consolidado = perfil_continuas.copy()
    perfil_consolidado[COLUNA_TEMPO + "_media"] = perfil_tempo
    for coluna in COLUNAS_BINARIAS:
        perfil_consolidado[coluna + "_prop"] = perfil_binarias[coluna]
    perfil_consolidado["taxa_DEATH_EVENT"] = taxa_obito
    perfil_consolidado["n"] = contagem
    caminho_perfil = DIR_SAIDA / "perfil_clusters.csv"
    perfil_consolidado.to_csv(caminho_perfil, encoding="utf-8")
    print(f"\n[arquivo salvo] {caminho_perfil.name}")

    # --- Grafico: taxa de obito por cluster (em PERCENTUAL) ------------
    fig, eixo = plt.subplots(figsize=(7.5, 5))
    valores_pct = taxa_obito.values * 100
    sns.barplot(
        x=taxa_obito.index.astype(str),
        y=valores_pct,
        hue=taxa_obito.index.astype(str),
        legend=False,
        ax=eixo,
        palette="rocket",
    )
    eixo.set_xlabel("Cluster")
    eixo.set_ylabel("Taxa de obito (%)")
    eixo.set_ylim(0, 100)
    eixo.set_title("Taxa de obito por cluster (alvo so descreve os grupos)")
    for i, v in enumerate(valores_pct):
        eixo.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    caminho_obito = DIR_SAIDA / "taxa_obito_por_cluster.png"
    fig.savefig(caminho_obito, dpi=120)
    plt.close(fig)
    print(f"[grafico salvo] {caminho_obito.name}")

    return taxa_obito, contagem, perfil_consolidado


# ======================================================================
# ETAPA 7 - VISUALIZACAO PCA 2D
# ======================================================================
def visualizar_pca(
    matriz_escalonada: np.ndarray, rotulos: pd.Series
) -> tuple[float, float]:
    """Projeta as continuas em PCA 2D e salva o scatter dos clusters. Retorna
    (variancia_pc1, variancia_pc2) para registro nas metricas."""
    imprimir_titulo("ETAPA 7 - VISUALIZACAO PCA 2D")
    pca = PCA(n_components=2, random_state=SEMENTE)
    coordenadas = pca.fit_transform(matriz_escalonada)
    variancia = pca.explained_variance_ratio_
    print(
        f"[INFO] Variancia explicada pelos 2 componentes principais: "
        f"PC1={variancia[0]:.3f}, PC2={variancia[1]:.3f} "
        f"(soma={variancia.sum():.3f})"
    )

    rotulos_str = pd.Series(np.asarray(rotulos)).astype(str)
    ordem_legenda = sorted(rotulos_str.unique())  # legenda ordenada "0, 1"
    fig, eixo = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(
        x=coordenadas[:, 0],
        y=coordenadas[:, 1],
        hue=rotulos_str,
        hue_order=ordem_legenda,
        palette="deep",
        s=60,
        edgecolor="white",
        ax=eixo,
    )
    # Centroides projetados por cluster (marcador "X") reforcam a separacao.
    for grupo in ordem_legenda:
        mascara = (rotulos_str == grupo).to_numpy()
        centro = coordenadas[mascara].mean(axis=0)
        eixo.scatter(
            centro[0], centro[1], marker="X", s=240, c="black",
            edgecolor="white", zorder=5,
        )
    eixo.set_xlabel(f"PC1 ({variancia[0]*100:.1f}% var.)")
    eixo.set_ylabel(f"PC2 ({variancia[1]*100:.1f}% var.)")
    eixo.set_title("Clusters do GaussianMixture projetados em PCA 2D")
    eixo.legend(title="cluster")
    fig.tight_layout()
    caminho_pca = DIR_SAIDA / "clusters_pca.png"
    fig.savefig(caminho_pca, dpi=120)
    plt.close(fig)
    print(f"[grafico salvo] {caminho_pca.name}")

    return float(variancia[0]), float(variancia[1])


# ======================================================================
# ETAPA 8 - METRICAS + PERSISTENCIA
# ======================================================================
def salvar_metricas_e_modelo(
    matriz_escalonada: np.ndarray,
    rotulos: pd.Series,
    modelo_campeao: GaussianMixture,
    escalonador: RobustScaler,
    continuas: list[str],
    selecao: dict,
    taxa_obito: pd.Series,
    contagem: pd.Series,
    variancia_pca: tuple[float, float],
    mapa_clusters: dict[int, int],
) -> dict:
    """Calcula as metricas internas de cluster, monta o dict, grava
    metricas.json e persiste modelo + scaler + colunas. Retorna o dict."""
    imprimir_titulo("ETAPA 8 - METRICAS + PERSISTENCIA DO MODELO")

    # Metricas internas de qualidade dos clusters (nao usam o alvo). Medimos
    # sobre os rotulos do CAMPEAO -> mesma particao de 'silhouette_na_selecao',
    # garantindo que os dois campos JAMAIS divirjam.
    rotulos_array = np.asarray(rotulos)
    silhueta = float(silhouette_score(matriz_escalonada, rotulos_array))
    davies = float(davies_bouldin_score(matriz_escalonada, rotulos_array))
    calinski = float(calinski_harabasz_score(matriz_escalonada, rotulos_array))

    print(f"Silhouette  (maior melhor): {silhueta:.4f}")
    print(f"Davies-Bouldin (menor melhor): {davies:.4f}")
    print(f"Calinski-Harabasz (maior melhor): {calinski:.2f}")

    metricas = {
        "algoritmo": "GaussianMixture",
        "tipo_covariancia": "full",
        "scaler": "RobustScaler",
        "binarias_no_treino": False,
        "time_no_treino": False,
        "caracteristicas_treino": list(continuas),
        "n_componentes": int(selecao["k"]),
        "bic": float(selecao["bic"]),
        "silhouette": silhueta,
        "silhouette_na_selecao": float(selecao["silhueta"]),
        "davies_bouldin": davies,
        "calinski_harabasz": calinski,
        "n_por_cluster": {str(c): int(contagem[c]) for c in contagem.index},
        "taxa_death_event_por_cluster": {
            str(c): float(taxa_obito[c]) for c in taxa_obito.index
        },
        "semente": SEMENTE,
        # Campos informativos (proveniencia) - nao alteram os numeros acima.
        "sensibilidade_k_por_limiar": selecao["sensibilidade"],
        "variancia_pca": {
            "pc1": variancia_pca[0],
            "pc2": variancia_pca[1],
            "soma": variancia_pca[0] + variancia_pca[1],
        },
        "convergiu": bool(modelo_campeao.converged_),
    }
    caminho_metricas = DIR_SAIDA / "metricas.json"
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"[arquivo salvo] {caminho_metricas.name}")

    # Persistencia para INFERENCIA separada (sem vazamento): salvamos o
    # modelo, o scaler, a ordem das colunas continuas e o mapa de reordenacao
    # dos clusters (para a inferencia traduzir o id do GMM -> id da narrativa).
    artefato = {
        "modelo": modelo_campeao,
        "escalonador": escalonador,
        "colunas_continuas": list(continuas),
        "mapa_clusters": mapa_clusters,
    }
    caminho_modelo = DIR_SAIDA / "modelo_gmm.joblib"
    joblib.dump(artefato, caminho_modelo, compress=3)
    print(f"[modelo salvo] {caminho_modelo.name} (modelo + scaler + colunas)")

    return metricas


# ======================================================================
# FLUXO PRINCIPAL
# ======================================================================
def main() -> None:
    """Orquestra as 8 etapas: carga -> EDA -> pre-proc -> selecao de k ->
    treino -> caracterizacao -> PCA -> metricas/persistencia."""
    imprimir_titulo("Q1 - HEART FAILURE | AGRUPAMENTO COM GaussianMixture (Aluno 3)")

    continuas = CARACTERISTICAS_NUMERICAS

    dados = carregar_dados()
    analise_exploratoria(dados)
    matriz_escalonada, escalonador = preprocessar(dados, continuas)
    selecao = escolher_numero_componentes(matriz_escalonada)
    modelo_campeao, rotulos = treinar_modelo_campeao(matriz_escalonada, selecao["k"])

    # Reordena os ids dos clusters de forma DETERMINISTICA e clinica: ordenamos
    # por serum_creatinine medio decrescente -> Cluster 0 = maior estresse. Isso
    # BLINDA a narrativa (README/inferencia/CSV) contra inversoes silenciosas de
    # id entre versoes do sklearn. NAO altera a particao nem as metricas.
    indice_creatinina = continuas.index("serum_creatinine")
    media_por_cluster_inicial = np.array(
        [
            matriz_escalonada[rotulos == g, indice_creatinina].mean()
            for g in range(selecao["k"])
        ]
    )
    rotulos, mapa_clusters = reordenar_clusters(rotulos, media_por_cluster_inicial)
    rotulos = pd.Series(rotulos)

    taxa_obito, contagem, _ = caracterizar_grupos(dados, continuas, rotulos)
    variancia_pca = visualizar_pca(matriz_escalonada, rotulos)
    salvar_metricas_e_modelo(
        matriz_escalonada,
        rotulos,
        modelo_campeao,
        escalonador,
        continuas,
        selecao,
        taxa_obito,
        contagem,
        variancia_pca,
        mapa_clusters,
    )

    # --- Encerramento ritual: lista os artefatos gerados ---------------
    imprimir_titulo("PIPELINE Q1 CONCLUIDO")
    print("Artefatos disponiveis em outputs/:")
    for arquivo in sorted(DIR_SAIDA.iterdir()):
        print(f"  - {arquivo.name}")
    print("\n[OK] Pipeline Q1 finalizado com sucesso.")


if __name__ == "__main__":
    main()
