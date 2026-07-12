param(
    [Parameter(Mandatory = $true)][string]$Vault,
    [Parameter(Mandatory = $true)][string]$Name
)

$ErrorActionPreference = "Stop"
obsidian vault="$Vault" create name="$Name"
