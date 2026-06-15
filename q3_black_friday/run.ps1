# run.ps1 - Fluxo completo Q3 Black Friday (Windows PowerShell)
# Executa: instala dependencias, treino/EDA/avaliacao (main.py) e a inferencia separada.
$ErrorActionPreference = "Stop"

# Garante execucao a partir da pasta do projeto (raiz = pasta deste script).
Set-Location -Path $PSScriptRoot

Write-Host "==> [1/3] Instalando dependencias (requirements.txt)"
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "==> [2/3] Treino + EDA + avaliacao (src/main.py)"
python src/main.py

Write-Host ""
Write-Host "==> [3/3] Inferencia separada (src/inferencia.py)"
python src/inferencia.py

Write-Host ""
Write-Host "==> Concluido. Artefatos em outputs/."
