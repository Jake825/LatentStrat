param(
    [string]$VaultRoot,
    [string]$OutputPath,
    [string]$ConfigPath,
    [string]$VaultName = "LatentStrat",
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"

if (-not $VaultRoot) {
    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\docs")).Path
}

$PythonScript = Join-Path $PSScriptRoot "vault_diagnostics.py"

if (-not $OutputPath) {
    $OutputPath = Join-Path $VaultRoot "obsidian-vault-diagnostics.md"
}

if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $VaultRoot $OutputPath
}

$ReportAbsolutePath = [System.IO.Path]::GetFullPath($OutputPath)

if (-not $ConfigPath) {
    $DefaultConfigPath = Join-Path $PSScriptRoot "..\references\latentstrat-diagnostics-config.json"
    if (Test-Path $DefaultConfigPath) {
        $ConfigPath = (Resolve-Path $DefaultConfigPath).Path
    }
}

if (-not (Test-Path $PythonScript)) {
    throw "Diagnostics engine not found: $PythonScript"
}

if (-not (Test-Path $VaultRoot)) {
    throw "Vault root does not exist: $VaultRoot"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonArgs = @()

if (-not $pythonCommand) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $pythonArgs += "-3"
    }
}

if (-not $pythonCommand) {
    throw "Python is not available on PATH. Install Python 3 and retry."
}

Write-Host "Running vault diagnostics..."
$pythonArgs += @(
    $PythonScript,
    "--vault", $VaultRoot,
    "--output", $ReportAbsolutePath
)

if ($ConfigPath) {
    if (-not (Test-Path $ConfigPath)) {
        throw "Diagnostics config does not exist: $ConfigPath"
    }
    $pythonArgs += @("--config", $ConfigPath)
}

& $pythonCommand.Source @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Diagnostics engine failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path $ReportAbsolutePath)) {
    throw "Diagnostics report was not written: $ReportAbsolutePath"
}

$VaultFullPath = [System.IO.Path]::GetFullPath($VaultRoot).TrimEnd([char[]]@("\", "/"))
$ReportDisplayPath = $ReportAbsolutePath
if ($ReportAbsolutePath.StartsWith($VaultFullPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    $ReportDisplayPath = $ReportAbsolutePath.Substring($VaultFullPath.Length).TrimStart([char[]]@("\", "/"))
}

Write-Host "Diagnostics report replaced: $ReportDisplayPath"

if ($OpenReport) {
    $obsidianCommand = Get-Command obsidian -ErrorAction SilentlyContinue
    if ($obsidianCommand) {
        Write-Host "Opening report in Obsidian..."
        & $obsidianCommand.Source vault="$VaultName" open path="$ReportDisplayPath" | Out-Null
    } else {
        Write-Warning "Obsidian CLI not found on PATH. Report was written to disk but not opened."
    }
}
