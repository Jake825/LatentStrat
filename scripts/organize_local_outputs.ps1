param(
    [string]$Root = (Get-Location).Path,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path

function Join-RootPath {
    param([string]$RelativePath)
    return Join-Path -Path $rootPath -ChildPath $RelativePath
}

function Get-RootRelativePath {
    param([string]$FullPath)
    $base = [Uri]((Join-Path -Path $rootPath -ChildPath ".") + [System.IO.Path]::DirectorySeparatorChar)
    $target = [Uri]$FullPath
    return [Uri]::UnescapeDataString($base.MakeRelativeUri($target).ToString()).Replace("/", [System.IO.Path]::DirectorySeparatorChar)
}

function Add-Move {
    param(
        [System.Collections.Generic.List[object]]$Moves,
        [string]$Source,
        [string]$Destination
    )
    $Moves.Add([pscustomobject]@{ Source = $Source; Destination = $Destination }) | Out-Null
}

$moves = [System.Collections.Generic.List[object]]::new()
Add-Move $moves "tba_cache.sqlite" "data/cache/tba.sqlite"
Add-Move $moves "tba_cache.sqlite-shm" "data/cache/tba.sqlite-shm"
Add-Move $moves "tba_cache.sqlite-wal" "data/cache/tba.sqlite-wal"
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite" "data/cache/openai_embeddings.sqlite"
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite-shm" "data/cache/openai_embeddings.sqlite-shm"
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite-wal" "data/cache/openai_embeddings.sqlite-wal"
Add-Move $moves "data/scouting.db" "data/scouting/scouting.db"
Add-Move $moves "data/latentstrat_embeddings.sqlite" "data/embeddings/latentstrat_embeddings.sqlite"
Add-Move $moves "data/pretrained_prior_2026.pt" "artifacts/prior/pretrained_prior_2026.pt"
Add-Move $moves "data/v57_sidecars_2026" "data/sidecars/v57_2026"
Add-Move $moves "data/v58_sidecars_2026" "data/sidecars/v58_2026"

$dataRoot = Join-RootPath "data"
if (Test-Path -LiteralPath $dataRoot) {
    Get-ChildItem -LiteralPath $dataRoot -File -Filter "prior_features*.parquet" | ForEach-Object {
        Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/prior/" + $_.Name)
    }
    Get-ChildItem -LiteralPath $dataRoot -File -Filter "features*.parquet" | ForEach-Object {
        if ($_.BaseName -match "^features_[0-9]{4}$" -or $_.BaseName -match "^features_v[0-9]+_[0-9]{4}$") {
            Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/season/" + $_.Name)
        } else {
            Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/event/" + $_.Name)
        }
    }
}

$planned = 0
$skipped = 0
foreach ($move in $moves) {
    $sourcePath = Join-RootPath $move.Source
    $destinationPath = Join-RootPath $move.Destination
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        continue
    }
    if (Test-Path -LiteralPath $destinationPath) {
        Write-Host "[skip exists] $($move.Source) -> $($move.Destination)"
        $skipped += 1
        continue
    }
    $planned += 1
    if ($Apply) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
        Move-Item -LiteralPath $sourcePath -Destination $destinationPath
        Write-Host "[moved] $($move.Source) -> $($move.Destination)"
    } else {
        Write-Host "[dry-run] $($move.Source) -> $($move.Destination)"
    }
}

if (Test-Path -LiteralPath (Join-RootPath "statbotics_offline_cache")) {
    Write-Host "[manual] statbotics_offline_cache cannot be converted to SQLite; warm data/cache/statbotics.sqlite with new commands, then clear legacy cache if desired."
}

if (Test-Path -LiteralPath $dataRoot) {
    $knownTopLevel = @(
        "cache",
        "embeddings",
        "features",
        "prior_cache",
        "scouting",
        "sidecars",
        "v57_sidecars_2026",
        "v58_sidecars_2026",
        "scouting.db",
        "latentstrat_embeddings.sqlite",
        "pretrained_prior_2026.pt"
    )
    $unknown = Get-ChildItem -LiteralPath $dataRoot | Where-Object {
        $name = $_.Name
        -not ($knownTopLevel -contains $name) -and
        -not ($name -like "features*.parquet") -and
        -not ($name -like "prior_features*.parquet")
    }
    foreach ($item in $unknown) {
        Write-Host "[manual review] data/$($item.Name)"
    }
}

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply to move known generated files."
}
Write-Host "Planned moves: $planned; skipped existing destinations: $skipped"
