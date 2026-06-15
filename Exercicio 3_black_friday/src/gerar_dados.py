# -*- coding: utf-8 -*-
"""
Gerador de dados sinteticos para o mini-projeto Q3 - Black Friday.

Contexto: multi-classificacao com 3 alvos (product_category, payment_method,
age_group). Caso o arquivo real `data/black_friday.csv` NAO exista, este modulo
gera um dataset sintetico (~4000 linhas) com dependencias probabilisticas entre
features e alvos, de modo que os modelos fiquem ACIMA do acaso.

IMPORTANTE: dados sinteticos servem APENAS para provar o pipeline. Um banner
bem visivel e impresso e o aviso e documentado no README.

SEMENTE = 42 (random_state) em todo o gerador, conforme exigido (NUNCA RANDOM_STATE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 42

# Esquema exato exigido pelo enunciado.
CATEGORIAS_PRODUTO = [
    "Eletronicos",
    "Roupas",
    "Casa",
    "Alimentos",
    "Beleza",
    "Brinquedos",
    "Esportes",
]
METODOS_PAGAMENTO = [
    "Cartao_Credito",
    "Cartao_Debito",
    "PIX",
    "Dinheiro",
    "Boleto",
]
FAIXAS_ETARIAS = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]

CATEGORIAS_CIDADE = ["A", "B", "C"]
GENEROS = ["M", "F"]

# Indices nome->posicao: deterministicos, entao preferimos calcular UMA vez aqui
# em vez de reconstruir dentro do loop de 4000 linhas (DRY + um pouco mais rapido).
IDX_PRODUTO = {c: k for k, c in enumerate(CATEGORIAS_PRODUTO)}
IDX_PAGAMENTO = {c: k for k, c in enumerate(METODOS_PAGAMENTO)}


def _amostra_categorica(rng: np.random.Generator, opcoes, pesos) -> str:
    """Amostra uma categoria a partir de pesos nao-normalizados."""
    pesos = np.asarray(pesos, dtype=float)
    pesos = pesos / pesos.sum()
    return rng.choice(opcoes, p=pesos)


def gerar_dataframe_sintetico(n_linhas: int = 4000, semente: int = SEMENTE) -> pd.DataFrame:
    """Gera o DataFrame sintetico com dependencias features -> alvos.

    A ideia e construir, para cada linha, vetores de pesos para cada alvo que
    dependem das features. Assim os modelos conseguem aprender um sinal real e
    ficar acima do acaso (acaso = 1/n_classes).
    """
    rng = np.random.default_rng(semente)

    registros = []
    for _ in range(n_linhas):
        # ----- Features independentes -----
        genero = rng.choice(GENEROS, p=[0.55, 0.45])
        ocupacao = int(rng.integers(0, 21))  # 0..20
        cidade = rng.choice(CATEGORIAS_CIDADE, p=[0.35, 0.40, 0.25])
        anos_cidade = int(rng.integers(0, 5))  # 0..4
        estado_civil = int(rng.integers(0, 2))  # 0/1
        # purchase_amount: faixa tipica de compras Black Friday (R$).
        valor_compra = float(np.round(rng.gamma(shape=2.0, scale=120.0) + 15.0, 2))
        quantidade = int(rng.integers(1, 6))  # 1..5

        # ----- Alvo age_group (depende de ocupacao, estado_civil, anos_cidade) -----
        # Jovens tendem a ocupacao baixa e solteiros; mais velhos o contrario.
        base_idade = np.ones(len(FAIXAS_ETARIAS)) * 0.3
        # ocupacao alta empurra para faixas mais velhas (pico gaussiano forte)
        deslocamento = (ocupacao / 20.0) * 6.0  # 0..6 (cobre todas as faixas)
        for i in range(len(FAIXAS_ETARIAS)):
            base_idade[i] += np.exp(-((i - deslocamento) ** 2) / 1.2) * 8.0
        if estado_civil == 1:  # casado -> faixas intermediarias/altas
            base_idade[2:6] *= 2.4
        else:  # solteiro -> faixas jovens
            base_idade[0:3] *= 2.2
        base_idade += anos_cidade * 0.10  # mais tempo na cidade -> leve idade+
        faixa_etaria = _amostra_categorica(rng, FAIXAS_ETARIAS, base_idade)

        # ----- Alvo product_category (depende de genero, valor, quantidade, cidade) -----
        base_prod = np.ones(len(CATEGORIAS_PRODUTO)) * 0.3
        idx = IDX_PRODUTO
        if valor_compra > 250:  # ticket alto -> eletronicos/esportes
            base_prod[idx["Eletronicos"]] *= 6.0
            base_prod[idx["Esportes"]] *= 3.0
        else:  # ticket baixo -> alimentos/beleza
            base_prod[idx["Alimentos"]] *= 5.0
            base_prod[idx["Beleza"]] *= 3.5
        if genero == "F":
            base_prod[idx["Beleza"]] *= 3.0
            base_prod[idx["Roupas"]] *= 3.0
        else:
            base_prod[idx["Eletronicos"]] *= 2.2
            base_prod[idx["Esportes"]] *= 2.2
        if quantidade >= 4:  # compra de muitos itens -> casa/brinquedos
            base_prod[idx["Casa"]] *= 4.5
            base_prod[idx["Brinquedos"]] *= 4.0
        if cidade == "A":
            base_prod[idx["Eletronicos"]] *= 1.8
        categoria_produto = _amostra_categorica(rng, CATEGORIAS_PRODUTO, base_prod)

        # ----- Alvo payment_method (depende de valor, cidade, idade ja sorteada) -----
        base_pag = np.ones(len(METODOS_PAGAMENTO)) * 0.3
        idxp = IDX_PAGAMENTO
        if valor_compra > 300:  # ticket alto -> credito/boleto
            base_pag[idxp["Cartao_Credito"]] *= 6.0
            base_pag[idxp["Boleto"]] *= 3.0
        else:  # ticket baixo -> PIX/dinheiro/debito
            base_pag[idxp["PIX"]] *= 4.5
            base_pag[idxp["Dinheiro"]] *= 3.0
            base_pag[idxp["Cartao_Debito"]] *= 2.5
        if cidade == "C":  # cidade menor -> mais dinheiro/boleto
            base_pag[idxp["Dinheiro"]] *= 2.2
            base_pag[idxp["Boleto"]] *= 1.8
        if faixa_etaria in ("0-17", "18-25"):  # jovens -> PIX/debito
            base_pag[idxp["PIX"]] *= 2.4
            base_pag[idxp["Cartao_Debito"]] *= 1.8
        elif faixa_etaria in ("51-55", "55+"):  # mais velhos -> boleto/dinheiro
            base_pag[idxp["Boleto"]] *= 2.4
            base_pag[idxp["Dinheiro"]] *= 2.0
        metodo_pagamento = _amostra_categorica(rng, METODOS_PAGAMENTO, base_pag)

        registros.append(
            {
                "gender": genero,
                "occupation": ocupacao,
                "city_category": cidade,
                "stay_years": anos_cidade,
                "marital_status": estado_civil,
                "purchase_amount": valor_compra,
                "quantity": quantidade,
                "age_group": faixa_etaria,
                "product_category": categoria_produto,
                "payment_method": metodo_pagamento,
            }
        )

    return pd.DataFrame.from_records(registros)


def imprimir_banner_placeholder() -> None:
    """Banner BEM VISIVEL avisando que os dados sao sinteticos (placeholder)."""
    linha = "!" * 78
    print(linha)
    print("!!" + " AVISO: DADOS SINTETICOS (PLACEHOLDER) ".center(74) + "!!")
    print("!!" + " O arquivo real data/black_friday.csv nao foi encontrado.".ljust(74) + "!!")
    print("!!" + " Foi gerado um dataset SINTETICO (seed 42) apenas para".ljust(74) + "!!")
    print("!!" + " PROVAR O PIPELINE. NAO use estes numeros como conclusao real.".ljust(74) + "!!")
    print("!!" + " Salvo em data/black_friday_sintetico.csv.".ljust(74) + "!!")
    print(linha)


def carregar_ou_gerar(diretorio_dados: Path) -> tuple[pd.DataFrame, bool]:
    """Carrega o CSV real se existir; caso contrario gera o sintetico.

    Retorna (dataframe, eh_sintetico).
    """
    caminho_real = diretorio_dados / "black_friday.csv"
    caminho_sintetico = diretorio_dados / "black_friday_sintetico.csv"

    if caminho_real.exists():
        print(f"[INFO] Dataset real encontrado em: {caminho_real}")
        df = pd.read_csv(caminho_real)
        return df, False

    imprimir_banner_placeholder()
    df = gerar_dataframe_sintetico(n_linhas=4000, semente=SEMENTE)
    diretorio_dados.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_sintetico, index=False, encoding="utf-8")
    print(f"[INFO] Dataset sintetico salvo em: {caminho_sintetico}")
    return df, True


if __name__ == "__main__":
    # Execucao direta apenas para gerar/visualizar os dados.
    raiz = Path(__file__).resolve().parent.parent
    dados, sintetico = carregar_ou_gerar(raiz / "data")
    print(f"\nLinhas: {len(dados)} | Sintetico: {sintetico}")
    print(dados.head())
