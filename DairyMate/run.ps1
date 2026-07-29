# Dairy Mate launcher (Windows PowerShell)
# Usage:  .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
} elseif (Test-Path "..\venv\Scripts\Activate.ps1") {
    & ..\venv\Scripts\Activate.ps1
}

streamlit run app.py
