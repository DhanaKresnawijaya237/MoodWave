# MoodWave Launcher
# Opens the OSC bridge and backend in separate terminals, waits for the
# backend to be healthy, then opens the browser.
#
# Usage:
#   Double-click start_moodwave.bat  (bypasses PowerShell execution policy)
#   Or from PowerShell: .\start_moodwave.ps1

$ErrorActionPreference = "Stop"

function Write-ColorLine($text, $color = "White") {
    Write-Host $text -ForegroundColor $color
}

Write-Host ""
Write-ColorLine "========================================" "Cyan"
Write-ColorLine "  MoodWave Launcher" "Cyan"
Write-ColorLine "========================================" "Cyan"
Write-Host ""

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PROJECT_ROOT) { $PROJECT_ROOT = Get-Location }
$BACKEND_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Auto-detect conda environment "CS330_v2"
# ---------------------------------------------------------------------------
$ENV_NAME = "CS330_v2"
$pythonExe = $null

$condaBases = @(
    "$env:USERPROFILE\miniconda3"
    "$env:USERPROFILE\Anaconda3"
    "$env:LOCALAPPDATA\miniconda3"
    "C:\ProgramData\miniconda3"
)

foreach ($base in $condaBases) {
    $candidate = Join-Path $base "envs\$ENV_NAME\python.exe"
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        Write-ColorLine "Found conda environment '$ENV_NAME' at: $base" "Green"
        break
    }
}

if (-not $pythonExe) {
    $pythonExe = "python"
    Write-ColorLine "WARNING: Could not find conda environment '$ENV_NAME'." "Red"
    Write-ColorLine "         Using system Python. If this fails, run manually:" "Red"
    Write-ColorLine "         conda activate $ENV_NAME" "Yellow"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# 1. OSC Bridge
# ---------------------------------------------------------------------------
Write-ColorLine "[1/4] Starting OSC Bridge  ->  ws://localhost:7011 / osc://127.0.0.1:7000" "Yellow"
$bridgeCmd = "cd `"$PROJECT_ROOT\touch_designer`"; `"$pythonExe`" osc_bridge.py"
Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $bridgeCmd
Write-ColorLine "      OSC Bridge terminal opened." "Green"
Write-Host ""

# ---------------------------------------------------------------------------
# 2. FastAPI Backend
# ---------------------------------------------------------------------------
Write-ColorLine "[2/4] Starting FastAPI Backend  ->  $BACKEND_URL" "Yellow"
$backendCmd = "cd `"$PROJECT_ROOT\backend`"; `"$pythonExe`" -m uvicorn main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd
Write-ColorLine "      Backend terminal opened." "Green"
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Wait for health check
# ---------------------------------------------------------------------------
Write-ColorLine "[3/4] Waiting for backend health check..." "Yellow"
$maxAttempts = 90
$attempt = 0
$ready = $false
while ($attempt -lt $maxAttempts -and -not $ready) {
    Start-Sleep -Seconds 1
    $attempt++
    try {
        $response = Invoke-WebRequest -Uri "$BACKEND_URL/health" `
            -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
        }
    } catch {
        # still starting
    }
}

if ($ready) {
    Write-ColorLine "      Backend is healthy! ($attempt s)" "Green"
} else {
    Write-ColorLine "      Backend is still booting (90 s elapsed)... check its terminal." "DarkYellow"
}
Write-Host ""

# ---------------------------------------------------------------------------
# 4. Open Browser
# ---------------------------------------------------------------------------
Write-ColorLine "[4/4] Opening browser..." "Yellow"
$frontendUrl = "$BACKEND_URL/"
try {
    $browserCmd = "start `"`" `"$frontendUrl`""
    Start-Process -FilePath "$env:ComSpec" -ArgumentList "/c", $browserCmd -WindowStyle Hidden
} catch {
    Start-Process -FilePath "rundll32.exe" `
        -ArgumentList "url.dll,FileProtocolHandler", $frontendUrl
}
Write-ColorLine "      Browser launched." "Green"
Write-ColorLine "      URL: $frontendUrl" "Gray"
Write-Host ""

# ---------------------------------------------------------------------------
# 5. TouchDesigner reminder
# ---------------------------------------------------------------------------
$toeFiles = Get-ChildItem -Path "$PROJECT_ROOT\touch_designer" -Filter "*.toe" `
    -ErrorAction SilentlyContinue

if ($toeFiles) {
    Write-ColorLine "TouchDesigner project(s) found:" "Green"
    foreach ($f in $toeFiles) {
        Write-Host "  $($f.FullName)" -ForegroundColor Gray
    }
} else {
    Write-ColorLine "No .toe file found in touch_designer/." "DarkYellow"
}

Write-ColorLine "Please open your TouchDesigner project manually if needed." "DarkYellow"
Write-Host ""
Write-ColorLine "========================================" "Cyan"
Write-ColorLine "  All services launched!" "Green"
Write-ColorLine "========================================" "Cyan"
Write-Host ""
Write-ColorLine "Press Ctrl+C to close this launcher." "DarkGray"
try {
    while ($true) {
        Start-Sleep -Seconds 3600
    }
} finally {
    Write-Host ""
}
