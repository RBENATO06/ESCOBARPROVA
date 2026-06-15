#!/usr/bin/env bash
# Fluxo completo Q2 - Wine Quality (Aluno 3)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Instalando dependencias..."
python -m pip install -r requirements.txt

echo "==> Executando treino + EDA + comparacao (src/main.py)..."
python src/main.py

echo "==> Executando inferencia em vinhos novos (src/inferencia.py)..."
python src/inferencia.py

echo "==> Concluido. Veja a pasta outputs/."
