#!/usr/bin/env bash
# run.sh - Fluxo completo Q3 Black Friday (Bash)
# Executa: instala dependencias, treino/EDA/avaliacao (main.py) e a inferencia separada.
set -euo pipefail

# Garante execucao a partir da pasta do projeto (raiz = pasta deste script).
cd "$(dirname "$0")"

echo "==> [1/3] Instalando dependencias (requirements.txt)"
python -m pip install -r requirements.txt

echo ""
echo "==> [2/3] Treino + EDA + avaliacao (src/main.py)"
python src/main.py

echo ""
echo "==> [3/3] Inferencia separada (src/inferencia.py)"
python src/inferencia.py

echo ""
echo "==> Concluido. Artefatos em outputs/."
