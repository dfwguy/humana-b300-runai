<#
.SYNOPSIS
  Save the logs of a numbered group of Run:ai workloads, then summarize them.

.DESCRIPTION
  UNTESTED. The logs subcommand differs between CLI versions. Check:
      runai training standard --help
  and set -LogsCommand if needed (default: "training standard logs").

.EXAMPLE
  .\scripts\collect-logs.ps1 -Prefix nccl-nvlink- -Count 16
#>
param(
  [string]$Prefix = "nccl-nvlink-",
  [int]$Count = 16,
  [string]$Project = "nvis-validation",
  [string]$LogsCommand = "training standard logs",
  [string]$OutDir = ""
)
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root ("results\" + (Get-Date -Format "yyyyMMdd")) }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$sub = $LogsCommand -split " "
for ($i = 1; $i -le $Count; $i++) {
  $name = "{0}{1:d2}" -f $Prefix, $i
  Write-Host "logs for $name"
  & runai @sub $name -p $Project *> (Join-Path $OutDir "$name.log")
}
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) {
  python (Join-Path $PSScriptRoot "summarize.py") $OutDir --csv (Join-Path $OutDir "summary.csv")
} else {
  Write-Host "python not found: open the .log files, or run scripts/summarize.py on the jump box."
  Get-ChildItem $OutDir -Filter *.log | ForEach-Object {
    $t = Get-Content $_.FullName -Raw
    "{0}: {1}" -f $_.Name, (($t -split "`n" | Select-String "Avg bus bandwidth|Out of bounds|RESULT=") -join " | ")
  }
}
