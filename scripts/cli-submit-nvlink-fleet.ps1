<#
.SYNOPSIS
  Submit one 8-GPU single-node NCCL (NVLink) workload per node with the Run:ai CLI.

.DESCRIPTION
  Each workload asks for all 8 GPUs of a node, so 16 workloads land on 16 different nodes
  (when the nodes are free). The node name is printed in each log (NODE=...), and
  scripts/summarize.py groups results by it.

  UNTESTED. Written from the Run:ai 2.23/2.25 CLI reference. Before relying on it, run:
      runai training standard submit --help
  and fix any flag that differs on this cluster's CLI version. Use -DryRun to print commands.
  PowerShell 7.3+ is assumed (PSNativeCommandArgumentPassing=Standard) so the multi-line
  script is passed to the container as one argument.

.EXAMPLE
  .\scripts\cli-submit-nvlink-fleet.ps1 -DryRun
  .\scripts\cli-submit-nvlink-fleet.ps1 -Count 16
#>
param(
  [int]$Count = 16,
  [string]$Project = "nvis-validation",
  [string]$Image = "nvcr.io/nvidia/pytorch:26.04-py3",
  [string]$Prefix = "nccl-nvlink-",
  [switch]$DryRun
)
$root = Split-Path -Parent $PSScriptRoot
$script = Get-Content -Raw (Join-Path $root "cli\nccl-nvlink.sh")
$envs = Get-Content (Join-Path $root "cli\env-nvlink.txt") | Where-Object { $_ -match "=" }

for ($i = 1; $i -le $Count; $i++) {
  $name = "{0}{1:d2}" -f $Prefix, $i
  $cliArgs = @("training", "standard", "submit", $name, "-p", $Project, "-i", $Image, "-g", "8", "--large-shm")
  foreach ($e in $envs) { $cliArgs += @("-e", $e) }
  $cliArgs += @("--", "bash", "-c", $script)
  if ($DryRun) {
    Write-Host ("runai " + ($cliArgs[0..11] -join " ") + " ... -- bash -c <script>")
  } else {
    Write-Host "submitting $name"
    & runai @cliArgs
  }
}
