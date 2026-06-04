param(
    [string]$Root = (Get-Location).Path,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$rootPrefix = $rootPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar

function Resolve-SafePath {
    param([string]$RelativePath)
    $fullPath = [System.IO.Path]::GetFullPath((Join-Path -Path $rootPath -ChildPath $RelativePath))
    if (-not $fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside workspace: $RelativePath -> $fullPath"
    }
    return $fullPath
}

function Get-RootRelativePath {
    param([string]$FullPath)
    $base = [Uri]$rootPrefix
    $target = [Uri]$FullPath
    return [Uri]::UnescapeDataString($base.MakeRelativeUri($target).ToString()).Replace(
        "/",
        [System.IO.Path]::DirectorySeparatorChar
    )
}

function Get-PathDigest {
    param([string]$Path)
    if ((Get-Item -LiteralPath $Path).PSIsContainer) {
        $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File | Sort-Object FullName)
        $builder = [System.Text.StringBuilder]::new()
        [long]$bytes = 0
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($Path.Length).TrimStart(
                [System.IO.Path]::DirectorySeparatorChar
            )
            $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            [void]$builder.Append($relative.Replace("\", "/")).Append(":").Append($hash).Append("`n")
            $bytes += $file.Length
        }
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $payload = [System.Text.Encoding]::UTF8.GetBytes($builder.ToString())
            $digest = [System.BitConverter]::ToString($sha.ComputeHash($payload)).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
        return [pscustomobject]@{ Sha256 = $digest; Bytes = $bytes }
    }
    $item = Get-Item -LiteralPath $Path
    return [pscustomobject]@{
        Sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        Bytes = $item.Length
    }
}

function Add-Move {
    param(
        [System.Collections.Generic.List[object]]$Moves,
        [string]$Source,
        [string]$Destination,
        [string]$Kind = "canonical",
        [switch]$Disposable
    )
    $Moves.Add(
        [pscustomobject]@{
            Source = $Source
            Destination = $Destination
            Kind = $Kind
            Disposable = [bool]$Disposable
        }
    ) | Out-Null
}

function Expand-ArchiveRecords {
    param([object]$Value)
    $expanded = [System.Collections.Generic.List[object]]::new()

    function Add-ExpandedRecord {
        param(
            [object]$Item,
            [System.Collections.Generic.List[object]]$Destination
        )
        if ($null -eq $Item) {
            return
        }
        if ($Item -is [System.Array]) {
            foreach ($child in $Item) {
                Add-ExpandedRecord $child $Destination
            }
            return
        }
        $propertyNames = @($Item.PSObject.Properties.Name)
        if ($propertyNames -contains "value" -and -not ($propertyNames -contains "source")) {
            Add-ExpandedRecord $Item.value $Destination
            return
        }
        $Destination.Add($Item) | Out-Null
    }

    Add-ExpandedRecord $Value $expanded
    return @($expanded)
}

$moves = [System.Collections.Generic.List[object]]::new()

# Durable canonical moves.
Add-Move $moves "data/world_model/match_breakdowns.sqlite" "data/pretraining/match-breakdown/corpus.sqlite"
Add-Move $moves "data/features/world_model/match_breakdown_alliances_2015_2026.parquet" "data/features/pretraining/match-breakdown/match_breakdown_alliances_2015_2026.parquet"
Add-Move $moves "artifacts/world_model/match_breakdown" "artifacts/pretraining/match-breakdown"
Add-Move $moves "artifacts/prior_v564_latent16" "artifacts/pretraining/prior/prior_v564_latent16"
Add-Move $moves "artifacts/prior_elbow_grid" "artifacts/pretraining/prior-grid/prior_elbow_grid"
Add-Move $moves "data/pretrained_prior_2026.pt" "artifacts/pretraining/prior/pretrained_prior_2026.pt"
Add-Move $moves "data/scouting.db" "data/scouting/scouting.db"
Add-Move $moves "data/latentstrat_embeddings.sqlite" "data/embeddings/latentstrat_embeddings.sqlite"
Add-Move $moves "data/v57_sidecars_2026" "data/sidecars/v57_2026"
Add-Move $moves "data/v58_sidecars_2026" "data/sidecars/v58_2026"

# Provider caches are transport data. Existing canonical copies win.
Add-Move $moves "tba_cache.sqlite" "data/cache/tba.sqlite" "cache" -Disposable
Add-Move $moves "tba_cache.sqlite-shm" "data/cache/tba.sqlite-shm" "cache" -Disposable
Add-Move $moves "tba_cache.sqlite-wal" "data/cache/tba.sqlite-wal" "cache" -Disposable
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite" "data/cache/openai_embeddings.sqlite" "cache" -Disposable
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite-shm" "data/cache/openai_embeddings.sqlite-shm" "cache" -Disposable
Add-Move $moves "data/prior_cache/openai_embeddings.sqlite-wal" "data/cache/openai_embeddings.sqlite-wal" "cache" -Disposable

$dataRoot = Resolve-SafePath "data"
if (Test-Path -LiteralPath $dataRoot) {
    Get-ChildItem -LiteralPath $dataRoot -File -Filter "prior_features*.parquet" | ForEach-Object {
        Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/pretraining/prior/" + $_.Name)
    }
    Get-ChildItem -LiteralPath $dataRoot -File -Filter "features*.parquet" | ForEach-Object {
        if ($_.BaseName -match "^features_[0-9]{4}$" -or $_.BaseName -match "^features_v[0-9]+_[0-9]{4}$") {
            Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/season/" + $_.Name)
        }
        else {
            Add-Move $moves (Get-RootRelativePath $_.FullName) ("data/features/event/" + $_.Name)
        }
    }
}

# Archive superseded root-level experiment directories after active moves are selected.
$artifactRoot = Resolve-SafePath "artifacts"
$reservedArtifactNames = @(
    "archive",
    "baselines",
    "evidence",
    "experimental",
    "inspection",
    "pretraining",
    "season",
    "smoke",
    "walk-forward",
    "world_model",
    "prior_v564_latent16",
    "prior_elbow_grid"
)
if (Test-Path -LiteralPath $artifactRoot) {
    Get-ChildItem -LiteralPath $artifactRoot | Where-Object {
        -not ($reservedArtifactNames -contains $_.Name)
    } | ForEach-Object {
        Add-Move $moves ("artifacts/" + $_.Name) ("artifacts/archive/legacy/" + $_.Name) "archive"
    }
}

$records = [System.Collections.Generic.List[object]]::new()
$planned = 0
$skipped = 0
foreach ($move in $moves) {
    $sourcePath = Resolve-SafePath $move.Source
    $destinationPath = Resolve-SafePath $move.Destination
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        continue
    }
    if (Test-Path -LiteralPath $destinationPath) {
        if ($move.Disposable) {
            Write-Host "[skip canonical cache exists] $($move.Source) -> $($move.Destination)"
            $skipped += 1
            continue
        }
        throw "Destination already exists; refusing collision: $($move.Destination)"
    }
    $planned += 1
    if (-not $Apply) {
        Write-Host "[dry-run] [$($move.Kind)] $($move.Source) -> $($move.Destination)"
        continue
    }
    $before = Get-PathDigest $sourcePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destinationPath) | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $destinationPath
    $after = Get-PathDigest $destinationPath
    if ($before.Sha256 -ne $after.Sha256 -or $before.Bytes -ne $after.Bytes) {
        throw "Hash verification failed after move: $($move.Source) -> $($move.Destination)"
    }
    $records.Add(
        [pscustomobject]@{
            kind = $move.Kind
            source = $move.Source
            destination = $move.Destination
            bytes = $after.Bytes
            sha256 = $after.Sha256
        }
    ) | Out-Null
    Write-Host "[moved $($move.Kind)] $($move.Source) -> $($move.Destination)"
}

if ($Apply) {
    $indexPath = Resolve-SafePath "artifacts/archive/index.json"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $indexPath) | Out-Null
    $existingRecords = @()
    if (Test-Path -LiteralPath $indexPath) {
        $existingJson = Get-Content -LiteralPath $indexPath -Raw
        if (-not [string]::IsNullOrWhiteSpace($existingJson)) {
            $existingRecords = @(Expand-ArchiveRecords ($existingJson | ConvertFrom-Json))
        }
    }
    $combinedRecords = @($existingRecords) + @($records)
    if ($combinedRecords.Count -eq 0) {
        $json = "[]"
    }
    elseif ($combinedRecords.Count -eq 1) {
        $json = "[" + ($combinedRecords[0] | ConvertTo-Json -Depth 5) + "]"
    }
    else {
        $json = $combinedRecords | ConvertTo-Json -Depth 5
    }
    [System.IO.File]::WriteAllText(
        $indexPath,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "[indexed] artifacts/archive/index.json records=$($combinedRecords.Count)"
}
else {
    Write-Host "Dry run only. Re-run with -Apply to move verified generated files."
}

if (Test-Path -LiteralPath (Resolve-SafePath "statbotics_offline_cache")) {
    Write-Host "[manual] statbotics_offline_cache cannot be converted to SQLite; warm data/cache/statbotics.sqlite, then clear the legacy cache."
}
Write-Host "Planned moves: $planned; skipped canonical caches: $skipped"
