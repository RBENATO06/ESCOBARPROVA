# Fluxo completo Q2 - Wine Quality (Aluno 3) - PowerShell
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==> Instalando dependencias..."
python -m pip install -r requirements.txt

Write-Host "==> Executando treino + EDA + comparacao (src/main.py)..."
python src/main.py

Write-Host "==> Executando inferencia em vinhos novos (src/inferencia.py)..."
python src/inferencia.py

Write-Host "==> Concluido. Veja a pasta outputs/."
