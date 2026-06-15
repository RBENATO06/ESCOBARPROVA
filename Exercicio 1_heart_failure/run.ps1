# run.ps1 - Fluxo completo do Q1 (Heart Failure / GaussianMixture) no PowerShell.
# Executa o treino+EDA (main.py) e depois a inferencia em paciente novo.
$ErrorActionPreference = "Stop"

# Garante que estamos na pasta do projeto (onde este script esta).
Set-Location -Path $PSScriptRoot

Write-Host "==> Instalando dependencias (requirements.txt)..."
python -m pip install -r requirements.txt

Write-Host "==> Executando pipeline principal (EDA + treino GMM)..."
python src/main.py

Write-Host "==> Executando inferencia em paciente novo..."
python src/inferencia.py

Write-Host "==> Concluido. Veja os artefatos na pasta outputs/."
