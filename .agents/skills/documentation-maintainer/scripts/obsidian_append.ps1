param(
    [Parameter(Mandatory = $true)][string]$Vault,
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Content
)

$ErrorActionPreference = "Stop"
obsidian vault="$Vault" append path="$Path" content="$Content"
