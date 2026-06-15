# -*- coding: utf-8 -*-
"""
Q1 - Heart Failure | INFERENCIA em paciente NOVO (clustering).

Script SEPARADO do treino (SEM vazamento de dados): carrega o modelo
GaussianMixture + RobustScaler ja treinados (salvos em outputs/modelo_gmm.joblib)
e atribui um paciente DESCONHECIDO a um dos grupos, com o GRAU DE CERTEZA de
cada grupo (Ranking/Top-N via predict_proba).

Como o GaussianMixture e PROBABILISTICO, alem de prever o grupo (predict)
mostramos a RESPONSABILIDADE/probabilidade de pertencer a CADA componente
(predict_proba) -> medida de confianca/ambiguidade.

IMPORTANTE (SEM vazamento):
- O paciente novo passa pelo MESMO escalonador (apenas .transform, NUNCA .fit).
- Usamos SOMENTE as variaveis continuas que treinaram o modelo (binarias,
  'time' e DEATH_EVENT NAO entram na inferencia).
- Os ids de cluster seguem a convencao do treino (mapa_clusters salvo no joblib):
  Cluster 0 = alto estresse renal/cardiaco, Cluster 1 = estavel.

Uso:
  python src/inferencia.py                      -> roda o exemplo 'ambos'
  python src/inferencia.py --exemplo critico    -> so o paciente critico
  python src/inferencia.py --exemplo estavel     -> so o paciente estavel
  python src/inferencia.py --top 1              -> mostra so o Top-1 do ranking
  python src/inferencia.py --modelo caminho.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# Caminhos ancorados no script (mesma familia de nomes do main.py).
RAIZ = Path(__file__).resolve().parent.parent
DIR_SAIDA = RAIZ / "outputs"
CAMINHO_MODELO = DIR_SAIDA / "modelo_gmm.joblib"


def imprimir_titulo(texto: str) -> None:
    """Cabecalho de console com regua de 70 '=' (mesma largura do main.py)."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def carregar_artefato(
    caminho: Path = CAMINHO_MODELO,
) -> tuple[GaussianMixture, RobustScaler, list[str], dict[int, int]]:
    """Carrega modelo + scaler + ordem das colunas continuas + mapa de clusters."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {caminho}. "
            "Rode antes o src/main.py para treinar e salvar o modelo."
        )
    artefato = joblib.load(caminho)
    # mapa_clusters traduz o id do componente do GMM -> id da narrativa (treino).
    # Identidade como padrao, para compatibilidade com artefatos antigos.
    mapa = artefato.get("mapa_clusters") or {}
    return (
        artefato["modelo"],
        artefato["escalonador"],
        artefato["colunas_continuas"],
        {int(k): int(v) for k, v in mapa.items()},
    )


def validar_paciente(paciente: dict, colunas_continuas: list[str]) -> None:
    """Falha CEDO e em PT-BR: confere que todas as continuas estao presentes e
    sao numericas (nao NaN/None). Avisa (sem abortar) sobre coluna desconhecida."""
    faltantes = [c for c in colunas_continuas if c not in paciente]
    if faltantes:
        raise ValueError(
            f"Faltam variaveis continuas obrigatorias: {faltantes}. "
            f"Esperadas: {colunas_continuas}."
        )
    for coluna in colunas_continuas:
        valor = paciente[coluna]
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            raise ValueError(
                f"Variavel '{coluna}' deve ser numerica (int/float); "
                f"recebido {type(valor).__name__} = {valor!r}."
            )
        if valor != valor:  # NaN
            raise ValueError(f"Variavel '{coluna}' esta ausente (NaN/None).")
    desconhecidas = [c for c in paciente if c not in colunas_continuas]
    if desconhecidas:
        print(
            f"[INFO] Colunas ignoradas na inferencia (nao continuas de treino): "
            f"{desconhecidas}."
        )


def prever_paciente(
    paciente: dict,
    modelo: GaussianMixture,
    escalonador: RobustScaler,
    colunas_continuas: list[str],
    mapa_clusters: dict[int, int],
) -> tuple[int, np.ndarray]:
    """Recebe UM paciente novo (dict) e devolve (cluster, probabilidades) ja na
    convencao de ids do treino. Saida: Top-N de grupos com % e marcador
    '<== atribuido' no topo. SEM vazamento: apenas .transform, NUNCA .fit."""
    validar_paciente(paciente, colunas_continuas)

    # Garante a MESMA ordem de colunas usada no treino.
    linha = pd.DataFrame([paciente])[colunas_continuas]

    # MESMO escalonador do treino -> apenas .transform NUNCA .fit.
    linha_escalonada = escalonador.transform(linha)

    grupo_gmm = int(modelo.predict(linha_escalonada)[0])
    probabilidades_gmm = modelo.predict_proba(linha_escalonada)[0]

    # Traduz os ids do GMM para a convencao da narrativa (Cluster 0 = estresse).
    n = len(probabilidades_gmm)
    mapa = mapa_clusters or {i: i for i in range(n)}
    probabilidades = np.zeros(n)
    for id_gmm, prob in enumerate(probabilidades_gmm):
        probabilidades[mapa.get(id_gmm, id_gmm)] = prob
    grupo = mapa.get(grupo_gmm, grupo_gmm)
    return grupo, probabilidades


def exemplos_padrao() -> dict[str, dict]:
    """Pacientes de exemplo FABRICADOS (nao copiados de linha do CSV), dentro das
    faixas reais mas longe da mediana, para exercitar a discriminacao do GMM."""
    return {
        # Perfil ESTAVEL (esperado Cluster 1): marcadores baixos.
        "estavel": {
            "age": 50.0,
            "creatinine_phosphokinase": 180,
            "ejection_fraction": 45,
            "platelets": 265000.0,
            "serum_creatinine": 0.9,
            "serum_sodium": 140,
            # Abaixo NAO entram na inferencia (so registro):
            "anaemia": 0,
            "diabetes": 0,
            "high_blood_pressure": 0,
            "sex": 1,
            "smoking": 0,
            "time": 180,
        },
        # Perfil CRITICO (esperado Cluster 0): alto estresse renal/cardiaco.
        "critico": {
            "age": 72.0,
            "creatinine_phosphokinase": 1600,
            "ejection_fraction": 25,
            "platelets": 180000.0,
            "serum_creatinine": 2.6,
            "serum_sodium": 131,
            "anaemia": 1,
            "diabetes": 0,
            "high_blood_pressure": 1,
            "sex": 1,
            "smoking": 0,
            "time": 40,
        },
    }


def imprimir_resultado(
    nome_exemplo: str,
    paciente: dict,
    colunas_continuas: list[str],
    grupo: int,
    probabilidades: np.ndarray,
    top: int | None,
) -> None:
    """Imprime a entrada e o Ranking de pertinencia (Top-N) com '<== atribuido'."""
    imprimir_titulo(f"PACIENTE NOVO [{nome_exemplo}] -> grupo (GaussianMixture)")
    print("Entrada (so as continuas entram; as demais sao ignoradas):")
    for chave, valor in paciente.items():
        marca = "  (usada)" if chave in colunas_continuas else "  (ignorada)"
        print(f"  {chave:<26}: {valor}{marca}")

    rotulo_grupo = {0: "alto estresse renal/cardiaco", 1: "estavel"}
    print(f"\n[INFO] GRUPO previsto: cluster {grupo} ({rotulo_grupo.get(grupo, '?')}).")

    # Ranking de pertinencia ORDENADO por probabilidade DECRESCENTE.
    ordem = np.argsort(probabilidades)[::-1]
    if top is not None:
        ordem = ordem[:top]
    print("Ranking de pertinencia (Top-N):")
    for posicao, cluster in enumerate(ordem):
        prob = probabilidades[cluster]
        destaque = "  <== atribuido" if posicao == 0 else ""
        print(f"  cluster {cluster}: {prob*100:5.1f}%{destaque}")
    print(
        f"Confianca da atribuicao (maior probabilidade): "
        f"{probabilidades.max()*100:.1f}%"
    )


def main() -> None:
    """Carrega o artefato e roda a inferencia nos pacientes selecionados."""
    analisador = argparse.ArgumentParser(
        description="Inferencia Q1 (GaussianMixture) em paciente novo - SEM vazamento."
    )
    analisador.add_argument(
        "--modelo",
        type=Path,
        default=CAMINHO_MODELO,
        help="Caminho do .joblib (default: outputs/modelo_gmm.joblib).",
    )
    analisador.add_argument(
        "--exemplo",
        choices=["estavel", "critico", "ambos"],
        default="ambos",
        help="Paciente de exemplo a avaliar (default: ambos).",
    )
    analisador.add_argument(
        "--top",
        type=int,
        default=None,
        help="Quantos clusters mostrar no ranking (default: todos).",
    )
    args = analisador.parse_args()

    imprimir_titulo("INFERENCIA Q1 - paciente NOVO -> grupo (GaussianMixture)")
    modelo, escalonador, colunas_continuas, mapa_clusters = carregar_artefato(args.modelo)
    print(f"[INFO] Modelo carregado. Variaveis continuas esperadas: {colunas_continuas}")

    exemplos = exemplos_padrao()
    nomes = list(exemplos) if args.exemplo == "ambos" else [args.exemplo]
    for nome in nomes:
        paciente = exemplos[nome]
        grupo, probabilidades = prever_paciente(
            paciente, modelo, escalonador, colunas_continuas, mapa_clusters
        )
        imprimir_resultado(
            nome, paciente, colunas_continuas, grupo, probabilidades, args.top
        )

    print("\n[OK] Inferencia concluida.")


if __name__ == "__main__":
    main()
