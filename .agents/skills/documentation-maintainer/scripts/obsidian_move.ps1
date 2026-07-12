param(
    [Parameter(Mandatory = $true)][string]$Vault,
    [Parameter(Mandatory = $true)][string]$From,
    [Parameter(Mandatory = $true)][string]$To
)

$ErrorActionPreference = "Stop"
obsidian vault="$Vault" move path="$From" to="$To"
