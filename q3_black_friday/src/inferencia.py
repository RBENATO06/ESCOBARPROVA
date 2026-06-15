# -*- coding: utf-8 -*-
"""
Q3 - Black Friday | INFERENCIA SEPARADA (sem vazamento).

Carrega o modelo MULTI-SAIDA treinado (joblib) e, para uma VENDA NOVA
(dicionario apenas com CARACTERISTICAS), produz as 3 PREVISOES
(product_category, payment_method, age_group), CADA UMA com seu GRAU DE CERTEZA
(% via predict_proba).

Importante (SEM vazamento): a venda nova contem SOMENTE features; nenhum alvo
e fornecido. O modelo e o mesmo objeto salvo no treino - aqui ele apenas preve
(apenas .predict/.predict_proba, NUNCA .fit).

Uso:
  python src/inferencia.py                       -> roda os exemplos pre-definidos
  python src/inferencia.py --exemplo 2           -> roda so o exemplo 2
  python src/inferencia.py --top 5               -> Top-5 candidatos por alvo
  python src/inferencia.py --modelo <caminho>    -> usa outro .joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = RAIZ / "outputs" / "modelo_multisaida.joblib"

# Rotulos amigaveis (PT-BR) dos alvos tecnicos - constante de modulo (DRY).
ROTULOS_AMIGAVEIS = {
    "product_category": "Categoria de produto",
    "payment_method": "Metodo de pagamento",
    "age_group": "Faixa etaria",
}


def imprimir_titulo(texto: str) -> None:
    """Imprime um cabecalho de console com regua (assinatura visual do aluno)."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def carregar_artefato(caminho: Path = CAMINHO_MODELO) -> dict:
    """Carrega o dicionario salvo com o modelo e metadados das colunas."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {caminho}. "
            "Rode primeiro: python src/main.py"
        )
    return joblib.load(caminho)


def validar_venda(artefato: dict, venda_nova: dict) -> None:
    """Valida a venda nova ANTES de prever, com mensagens claras do narrador.

    (1) checa features FALTANTES (senao o sklearn estoura um KeyError cru);
    (2) coage as numericas com pd.to_numeric, transformando o erro opaco do
        sklearn ("could not convert string to float: caro") numa mensagem que
        aponta a COLUNA culpada. Caminho feliz (entrada valida) passa intacto.
    """
    colunas_features = artefato["colunas_caracteristicas"]
    colunas_numericas = artefato.get("caracteristicas_numericas", [])

    faltantes = [c for c in colunas_features if c not in venda_nova]
    if faltantes:
        raise ValueError(
            "Venda invalida: faltam as caracteristicas "
            f"{faltantes}. Esperadas: {colunas_features}."
        )

    for coluna in colunas_numericas:
        try:
            pd.to_numeric(venda_nova[coluna])
        except (TypeError, ValueError):
            raise ValueError(
                f"Venda invalida: a caracteristica '{coluna}' deveria ser "
                f"numerica, mas recebeu {venda_nova[coluna]!r}."
            ) from None


def prever_venda(artefato: dict, venda_nova: dict) -> dict:
    """Recebe UMA venda nova (so features) e retorna 3 previsoes com certeza.

    Retorna um dicionario:
      { alvo: {"previsao": classe, "certeza_pct": xx.x,
               "ranking": [(classe, pct), ...]} }
    """
    validar_venda(artefato, venda_nova)

    modelo = artefato["modelo"]
    colunas_features = artefato["colunas_caracteristicas"]
    colunas_alvo = artefato["colunas_alvo"]

    # Monta o DataFrame de 1 linha apenas com as caracteristicas (sem alvos).
    linha = pd.DataFrame([{c: venda_nova[c] for c in colunas_features}])

    # predict_proba -> lista (uma matriz por alvo).
    lista_probas = modelo.predict_proba(linha)
    # estimadores individuais (um por alvo) -> usados p/ recuperar classes_.
    estimadores = modelo.named_steps["multi_saida"].estimators_

    resultado = {}
    for i, alvo in enumerate(colunas_alvo):
        probas = lista_probas[i][0]  # vetor de probabilidades da unica linha
        classes = estimadores[i].classes_
        idx_ordenado = np.argsort(probas)[::-1]

        previsao = classes[idx_ordenado[0]]
        certeza_pct = round(float(probas[idx_ordenado[0]]) * 100, 2)
        ranking = [
            (str(classes[j]), round(float(probas[j]) * 100, 2)) for j in idx_ordenado
        ]
        resultado[alvo] = {
            "previsao": str(previsao),
            "certeza_pct": certeza_pct,
            "ranking": ranking,
        }
    return resultado


def imprimir_previsao(venda_nova: dict, previsoes: dict, top: int = 3) -> None:
    imprimir_titulo("VENDA NOVA (apenas caracteristicas - SEM vazamento de alvos):")
    for k, v in venda_nova.items():
        print(f"  {k:<18}: {v}")

    print("\n--> 3 PREVISOES com GRAU DE CERTEZA:")
    for alvo, info in previsoes.items():
        nome = ROTULOS_AMIGAVEIS.get(alvo, alvo)
        print(f"\n  [{nome}]")
        print(f"    Previsao : {info['previsao']}  (certeza {info['certeza_pct']:.2f}%)")
        print(f"    Top-{top} candidatos:")
        for posicao, (classe, pct) in enumerate(info["ranking"][:top]):
            # a 1a linha do ranking e exatamente a classe atribuida (vencedora).
            marca = "  <== atribuido" if posicao == 0 else ""
            print(f"      - {classe:<16} {pct:5.2f}%{marca}")


def exemplos_padrao() -> list[dict]:
    """Algumas vendas novas de exemplo (somente features)."""
    return [
        {
            "gender": "F",
            "occupation": 3,
            "city_category": "A",
            "stay_years": 2,
            "marital_status": 0,
            "purchase_amount": 89.90,
            "quantity": 1,
        },
        {
            "gender": "M",
            "occupation": 17,
            "city_category": "C",
            "stay_years": 4,
            "marital_status": 1,
            "purchase_amount": 540.00,
            "quantity": 2,
        },
        {
            "gender": "M",
            "occupation": 10,
            "city_category": "B",
            "stay_years": 1,
            "marital_status": 1,
            "purchase_amount": 320.50,
            "quantity": 4,
        },
    ]


def analisar_argumentos() -> argparse.Namespace:
    """Define as opcoes de linha de comando da inferencia."""
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia Q3 - Black Friday: carrega o modelo multi-saida e preve "
            "3 alvos (com grau de certeza) para vendas novas. SEM vazamento."
        )
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=CAMINHO_MODELO,
        help="Caminho do .joblib do modelo (padrao: outputs/modelo_multisaida.joblib).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Quantidade de candidatos no ranking de cada alvo (padrao: 3).",
    )
    parser.add_argument(
        "--exemplo",
        type=int,
        default=None,
        help="Indice 1..N do exemplo a rodar (padrao: roda todos os exemplos).",
    )
    return parser.parse_args()


def main() -> None:
    args = analisar_argumentos()

    artefato = carregar_artefato(args.modelo)
    print("[INFO] Modelo multi-saida carregado para INFERENCIA.")
    # Aviso propagado do treino: se o modelo nasceu de dados sinteticos, lembramos
    # quem roda SO a inferencia (fallback silencioso p/ artefatos antigos).
    if artefato.get("treinado_em_sintetico", False):
        print(
            "[INFO] Previsoes baseadas em modelo treinado com DADOS SINTETICOS "
            "(placeholder) - NAO use como conclusao real."
        )

    exemplos = exemplos_padrao()
    if args.exemplo is not None:
        if not 1 <= args.exemplo <= len(exemplos):
            raise SystemExit(
                f"--exemplo deve estar entre 1 e {len(exemplos)} "
                f"(recebido: {args.exemplo})."
            )
        exemplos = [exemplos[args.exemplo - 1]]

    for venda in exemplos:
        previsoes = prever_venda(artefato, venda)
        imprimir_previsao(venda, previsoes, top=args.top)

    print("\n[OK] Inferencia concluida.")


if __name__ == "__main__":
    main()
