Write-Host "Starting LegalDocGPT Backend Server..." -ForegroundColor Green
Write-Host "Working Directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host "Python Version: $(python --version)" -ForegroundColor Yellow

try {
    Write-Host "Starting server on http://127.0.0.1:8001..." -ForegroundColor Cyan
    python simple_server.py
} catch {
    Write-Host "Error starting server: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Press any key to continue..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}



