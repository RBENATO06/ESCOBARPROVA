# -*- coding: utf-8 -*-
"""
Utilitarios de metricas para multi-classificacao.

Calcula, a partir da matriz de confusao multiclasse, as metricas
one-vs-rest por classe:
  - sensibilidade (recall) = TP / (TP + FN)
  - especificidade        = TN / (TN + FP)
Alem de acuracia por classe, F1 por classe e metricas globais.

Tudo em PT-BR.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def metricas_por_classe(
    y_verdadeiro: pd.Series, y_previsto: pd.Series, classes: list[str]
) -> dict:
    """Calcula metricas one-vs-rest por classe a partir da matriz de confusao.

    Para cada classe i (tratada como positiva), considerando o restante negativo:
      TP = mc[i, i]
      FN = soma da linha i menos TP   (eram i, foram previstos como outra coisa)
      FP = soma da coluna i menos TP  (nao eram i, mas previstos como i)
      TN = total - TP - FN - FP
    """
    mc = confusion_matrix(y_verdadeiro, y_previsto, labels=classes)
    total = mc.sum()

    f1_classes = f1_score(
        y_verdadeiro, y_previsto, labels=classes, average=None, zero_division=0
    )

    resultado = {}
    for i, classe in enumerate(classes):
        tp = int(mc[i, i])
        fn = int(mc[i, :].sum() - tp)
        fp = int(mc[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # acuracia por classe (one-vs-rest): proporcao de acertos olhando essa classe
        acuracia_classe = (tp + tn) / total if total > 0 else 0.0

        resultado[str(classe)] = {
            "suporte": int(mc[i, :].sum()),
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "sensibilidade": round(float(sensibilidade), 4),
            "especificidade": round(float(especificidade), 4),
            "acuracia_classe": round(float(acuracia_classe), 4),
            "f1": round(float(f1_classes[i]), 4),
        }
    return resultado


def metricas_globais(
    y_verdadeiro: pd.Series, y_previsto: pd.Series, classes: list[str]
) -> dict:
    """Metricas agregadas (globais) do alvo.

    Resumo das 4 chaves devolvidas:
      - acuracia_global = acertos / total (proporcao de previsoes corretas);
      - f1_macro        = media SIMPLES do F1 por classe (toda classe pesa
        igual -> nao deixa as raras "sumirem");
      - f1_ponderado    = media do F1 ponderada pelo suporte de cada classe;
      - acaso_uniforme  = 1/n_classes (linha de base do chute uniforme).
    Passamos labels=classes nos f1_score para FIXAR o universo de classes: se
    uma classe rara nao aparecer no teste, ela ainda entra no F1 macro (senao o
    numero se deslocaria silenciosamente num dataset real desbalanceado).
    """
    return {
        "acuracia_global": round(float(accuracy_score(y_verdadeiro, y_previsto)), 4),
        "f1_macro": round(
            float(
                f1_score(
                    y_verdadeiro,
                    y_previsto,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_ponderado": round(
            float(
                f1_score(
                    y_verdadeiro,
                    y_previsto,
                    labels=classes,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "acaso_uniforme": round(1.0 / len(classes), 4),
    }
