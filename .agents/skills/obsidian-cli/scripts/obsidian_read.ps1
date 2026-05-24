param(
    [Parameter(Mandatory = $true)][string]$Vault,
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = "Stop"
obsidian vault="$Vault" read path="$Path"
