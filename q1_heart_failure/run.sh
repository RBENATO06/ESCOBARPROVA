#!/usr/bin/env bash
# run.sh - Fluxo completo do Q1 (Heart Failure / GaussianMixture).
# Executa o treino+EDA (main.py) e depois a inferencia em paciente novo.
set -euo pipefail

# Garante que estamos na pasta do projeto (onde este script esta).
cd "$(dirname "$0")"

echo "==> Instalando dependencias (requirements.txt)..."
python -m pip install -r requirements.txt

echo "==> Executando pipeline principal (EDA + treino GMM)..."
python src/main.py

echo "==> Executando inferencia em paciente novo..."
python src/inferencia.py

echo "==> Concluido. Veja os artefatos na pasta outputs/."
