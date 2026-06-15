# -*- coding: utf-8 -*-
"""
Q2 - Wine Quality | INFERENCIA em vinho(s) NOVO(S) (sem vazamento).

Carrega o pipeline vencedor salvo (joblib) e preve a qualidade de vinhos
novos passados como dicionarios de caracteristicas. SEM vazamento: o
StandardScaler ja foi ajustado SOMENTE no conjunto de treino DENTRO do
pipeline; aqui apenas aplicamos (apenas .transform NUNCA .fit) e prevemos.

Por padrao usa o MODELO PRINCIPAL definido em outputs/metricas.json
(7 classes ou 3 faixas). Pode-se forcar via argumento --modelo.

Onde ha predict_proba, imprimimos o RANKING Top-N das classes/faixas com %
e marcamos a vencedora (<== atribuido) — assinatura de saida do explorador.

Uso:
    python src/inferencia.py                      # usa o modelo principal
    python src/inferencia.py --modelo faixas
    python src/inferencia.py --modelo 7classes
    python src/inferencia.py --top 5              # mostra Top-5 no ranking
    python src/inferencia.py --csv vinhos.csv     # le um lote de vinhos novos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

RAIZ = Path(__file__).resolve().parent.parent
DIR_SAIDA = RAIZ / "outputs"

ARQ_7CLASSES = DIR_SAIDA / "modelo_campeao_7classes.joblib"
ARQ_FAIXAS = DIR_SAIDA / "modelo_campeao_faixas.joblib"


def imprimir_titulo(texto: str) -> None:
    """Mesmo helper de cabecalho do main.py (regua de 70 '='), p/ unificar a voz."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def descobrir_modelo_principal() -> str:
    """Le metricas.json para saber qual modelo e o principal (7classes/faixas).

    Robusto a JSON ausente/ilegivel: nesse caso cai no padrao 'faixas' (o
    modelo principal do experimento) com aviso, em vez de quebrar com traceback.
    """
    caminho = DIR_SAIDA / "metricas.json"
    if not caminho.exists():
        print("[INFO] metricas.json ausente -> usando modelo principal padrao (faixas)")
        return "faixas"
    try:
        m = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("[INFO] metricas.json ilegivel -> usando modelo principal padrao (faixas)")
        return "faixas"
    principal = m.get("comparacao_7classes_vs_3faixas", {}).get(
        "modelo_principal_inferencia", "")
    if "faixa" in principal.lower():
        return "faixas"
    return "7classes"


def carregar_pacote(tipo: str) -> dict:
    """Carrega o pacote joblib (pipeline + metadados) do tipo escolhido."""
    arq = ARQ_FAIXAS if tipo == "faixas" else ARQ_7CLASSES
    if not arq.exists():
        raise FileNotFoundError(
            f"Modelo '{arq.name}' nao encontrado. Rode antes: python src/main.py")
    return joblib.load(arq)


def validar_entrada(vinhos_novos: list[dict], colunas: list[str]) -> pd.DataFrame:
    """Valida os vinhos de entrada e devolve um DataFrame com as colunas na
    ordem esperada pelo pipeline.

    (a) cada vinho deve conter TODAS as caracteristicas de 'colunas'; se faltar
        alguma, levanta ValueError em PT-BR listando exatamente as ausentes
        (em vez do KeyError cru do pandas);
    (b) os valores sao coagidos com pd.to_numeric(errors='coerce'); se sobrar
        NaN, levanta ValueError dizendo qual coluna/vinho tem valor nao-numerico.
    """
    if not vinhos_novos:
        raise ValueError("Nenhum vinho fornecido para inferencia.")

    quadro = pd.DataFrame(vinhos_novos)

    # (a) checagem de colunas faltantes, vinho a vinho
    colunas_set = set(quadro.columns)
    for i, vinho in enumerate(vinhos_novos):
        ausentes = [c for c in colunas if c not in vinho]
        if ausentes:
            raise ValueError(
                f"Caracteristicas FALTANDO no vinho {i + 1}: {ausentes}. "
                f"Esperadas (11): {colunas}.")

    # reordena/seleciona exatamente as caracteristicas do modelo
    entrada = quadro[colunas].copy()

    # (b) coercao numerica + relato de valores nao-numericos
    for coluna in colunas:
        coagida = pd.to_numeric(entrada[coluna], errors="coerce")
        nan_novos = coagida.isna() & entrada[coluna].notna()
        if nan_novos.any():
            indice = int(nan_novos.idxmax())
            valor = entrada.loc[indice, coluna]
            raise ValueError(
                f"Valor nao-numerico no vinho {indice + 1}, coluna '{coluna}': "
                f"{valor!r}.")
        if coagida.isna().any():
            indice = int(coagida.isna().idxmax())
            raise ValueError(
                f"Valor AUSENTE (vazio) no vinho {indice + 1}, coluna '{coluna}'.")
        entrada[coluna] = coagida

    return entrada


def prever(vinhos_novos: list[dict], tipo: str | None = None,
           top: int = 3) -> tuple[pd.DataFrame, str, str]:
    """Recebe lista de dicts (caracteristicas) e devolve
    (DataFrame, tipo, nome_modelo).

    O DataFrame traz a previsao e, quando disponivel, a probabilidade da classe
    prevista e o ranking Top-N (lista de (classe, %)). SEM vazamento: o pipeline
    salvo aplica apenas .transform/.predict (NUNCA .fit).
    """
    if tipo is None:
        tipo = descobrir_modelo_principal()

    pacote = carregar_pacote(tipo)
    pipeline = pacote["pipeline"]
    colunas = pacote["caracteristicas"]

    # Valida e garante a ordem/colunas corretas (sem 'quality'/'tipo_vinho')
    entrada = validar_entrada(vinhos_novos, colunas)

    previsoes = pipeline.predict(entrada)

    resultado = entrada.copy()
    coluna_prev = "faixa_prevista" if tipo == "faixas" else "quality_prevista"
    resultado[coluna_prev] = previsoes

    # Probabilidade da classe prevista + ranking Top-N (ExtraTrees/MLP tem proba)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(entrada)            # (n_amostras, n_classes)
        classes = list(pipeline.classes_)
        # indexacao VETORIZADA: posicao da classe prevista em cada linha
        indices = np.array([classes.index(p) for p in previsoes])
        prob_classe_prevista = probas[np.arange(len(previsoes)), indices]
        resultado["probabilidade"] = np.round(prob_classe_prevista, 4)

        # ranking ordenado (todas as classes, do maior % p/ o menor) por vinho
        rankings = []
        for linha_proba in probas:
            ordem = np.argsort(linha_proba)[::-1]
            rankings.append([
                (str(classes[j]), round(float(linha_proba[j]) * 100, 2))
                for j in ordem
            ])
        resultado["ranking"] = rankings
    else:
        print("[INFO] O modelo carregado NAO expoe predict_proba: a previsao "
              "sai SEM grau de certeza (sem probabilidade/ranking).")

    return resultado, tipo, pacote.get("nome_modelo", "?")


def carregar_vinhos_csv(caminho_csv: Path) -> list[dict]:
    """Le um lote de vinhos novos de um CSV do usuario (uma linha por vinho).

    NAO carregamos linhas dos winequality-*.csv como 'novas' (seria copia do
    dataset). Este caminho serve para vinhos REAIS fornecidos pelo usuario.
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(f"CSV de entrada nao encontrado: {caminho_csv}")
    # aceita ',' ou ';' como separador (sep=None + engine='python' infere)
    quadro = pd.read_csv(caminho_csv, sep=None, engine="python")
    return quadro.to_dict(orient="records")


def vinhos_exemplo() -> list[dict]:
    """3 vinhos NOVOS de exemplo: valores fisico-quimicos PLAUSIVEIS, porem
    FABRICADOS a mao (NAO copiados de nenhuma linha do dataset). Servem para
    demonstrar a inferencia em instancias ineditas.
      - Vinho 1: perfil "tinto rustico" -> acidez volatil alta, alcool baixo
                 (tende a faixa BAIXA).
      - Vinho 2: perfil "branco mediano" -> equilibrado (tende a faixa MEDIA).
      - Vinho 3: perfil "premium" -> acidez volatil baixa, alcool alto
                 (tende a faixa ALTA).
    """
    return [
        {
            "fixed acidity": 7.9, "volatile acidity": 0.62, "citric acid": 0.12,
            "residual sugar": 2.3, "chlorides": 0.081, "free sulfur dioxide": 13.0,
            "total sulfur dioxide": 41.0, "density": 0.9971, "pH": 3.39,
            "sulphates": 0.61, "alcohol": 9.7,
        },
        {
            "fixed acidity": 6.7, "volatile acidity": 0.28, "citric acid": 0.33,
            "residual sugar": 5.2, "chlorides": 0.045, "free sulfur dioxide": 35.0,
            "total sulfur dioxide": 128.0, "density": 0.9942, "pH": 3.18,
            "sulphates": 0.47, "alcohol": 10.1,
        },
        {
            "fixed acidity": 6.9, "volatile acidity": 0.21, "citric acid": 0.36,
            "residual sugar": 1.7, "chlorides": 0.038, "free sulfur dioxide": 41.0,
            "total sulfur dioxide": 118.0, "density": 0.9905, "pH": 3.28,
            "sulphates": 0.62, "alcohol": 12.7,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia de qualidade de vinho.")
    parser.add_argument(
        "--modelo", choices=["7classes", "faixas"], default=None,
        help="Forca o tipo de modelo. Padrao: modelo principal de metricas.json")
    parser.add_argument(
        "--top", type=int, default=3,
        help="Quantas classes mostrar no ranking de cada vinho (padrao: 3).")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="CSV com vinhos novos do usuario (1 linha/vinho). Padrao: exemplos.")
    args = parser.parse_args()

    try:
        if args.csv:
            vinhos_novos = carregar_vinhos_csv(Path(args.csv))
            print(f"[INFO] Lendo {len(vinhos_novos)} vinho(s) de {args.csv}")
        else:
            vinhos_novos = vinhos_exemplo()

        resultado, tipo, nome_modelo = prever(
            vinhos_novos, tipo=args.modelo, top=args.top)
    except FileNotFoundError as erro:
        # artefatos ausentes -> erro amigavel e exit code != 0 controlado
        print(f"[ERRO] {erro}")
        sys.exit(1)
    except ValueError as erro:
        print(f"[ERRO] Entrada invalida: {erro}")
        sys.exit(1)

    imprimir_titulo(f"INFERENCIA - vinhos NOVOS | modelo: {tipo} ({nome_modelo})")
    coluna_prev = "faixa_prevista" if tipo == "faixas" else "quality_prevista"
    tem_proba = "probabilidade" in resultado.columns

    for indice, linha_vinho in resultado.iterrows():
        prev = linha_vinho[coluna_prev]
        if tem_proba:
            print(f"\nVinho {indice + 1}: previsao = {prev}  "
                  f"(prob = {linha_vinho['probabilidade']:.4f})")
            print(f"  Top-{args.top} ranking de {'faixas' if tipo == 'faixas' else 'classes'}:")
            for classe, pct in linha_vinho["ranking"][:args.top]:
                marca = "  <== atribuido" if str(classe) == str(prev) else ""
                print(f"    - {classe:<8} {pct:6.2f}%{marca}")
        else:
            print(f"\nVinho {indice + 1}: previsao = {prev}")

    print("\nTabela completa (previsao + probabilidade):")
    colunas_mostrar = [coluna_prev] + (["probabilidade"] if tem_proba else [])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(resultado[colunas_mostrar])

    if tem_proba:
        print("\nLeitura critica (perfil explorador): a probabilidade aqui e a "
              "FRACAO das 400 arvores do ExtraTrees que votou na faixa vencedora "
              "(nao uma probabilidade calibrada). Como FABRICAMOS vinhos de "
              "fronteira (entre faixas vizinhas), o comite fica dividido e a "
              "certeza cai para ~0,52-0,56 — coerente com o desbalanceamento e "
              "com a faixa 'alta', a mais dificil. Vinhos com perfil mais extremo "
              "(muito alcool/baixa densidade) tenderiam a votos mais concentrados.")

    print("\nPIPELINE Q2 CONCLUIDO")
    print("[OK] Inferencia concluida com sucesso.")


if __name__ == "__main__":
    main()
