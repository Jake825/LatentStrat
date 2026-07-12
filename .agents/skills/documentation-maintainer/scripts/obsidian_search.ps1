param(
    [Parameter(Mandatory = $true)][string]$Vault,
    [Parameter(Mandatory = $true)][string]$Query
)

$ErrorActionPreference = "Stop"
obsidian vault="$Vault" search query="$Query"
