$Host.UI.RawUI.WindowTitle = "Realtime Transcriber"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Realtime Transcriber" -ForegroundColor Cyan
Write-Host "  Ctrl+C для остановки" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python main.py @args
