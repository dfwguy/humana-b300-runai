<#
.SYNOPSIS
  Submit one 8-GPU single-node Spectrum-X (NICs forced, NVLink excluded) NCCL workload per node.

.DESCRIPTION
  Needs the secondary networks and their resource names. Confirm them first (docs/TONIGHT-PLAN.md,
  step 0) and put the real values in config/site.json, then run python scripts/render.py.
  Reads cli/networks-annotation.txt and cli/extended-resources.txt.

  UNTESTED. Verify flags with:  runai training standard submit --help
  (--annotation, --extended-resource and --capability are listed in the CLI reference.)
#>
param(
  [int]$Count = 16,
  [string]$Project = "nvis-validation",
  [string]$Image = "nvcr.io/nvidia/pytorch:26.04-py3",
  [string]$Prefix = "nccl-spx-",
  [switch]$DryRun
)
$root = Split-Path -Parent $PSScriptRoot
$script = Get-Content -Raw (Join-Path $root "cli\nccl-spx.sh")
$envs = Get-Content (Join-Path $root "cli\env-spx.txt") | Where-Object { $_ -match "=" }
$nets = (Get-Content -Raw (Join-Path $root "cli\networks-annotation.txt")).Trim()
$res = Get-Content (Join-Path $root "cli\extended-resources.txt") | Where-Object { $_ -match "=" }

for ($i = 1; $i -le $Count; $i++) {
  $name = "{0}{1:d2}" -f $Prefix, $i
  $cliArgs = @("training", "standard", "submit", $name, "-p", $Project, "-i", $Image, "-g", "8", "--large-shm")
  $cliArgs += @("--annotation", "k8s.v1.cni.cncf.io/networks=$nets")
  foreach ($r in $res) { $cliArgs += @("--extended-resource", $r) }
  foreach ($c in @("NET_ADMIN", "NET_RAW", "IPC_LOCK", "SYS_RESOURCE")) { $cliArgs += @("--capability", $c) }
  foreach ($e in $envs) { $cliArgs += @("-e", $e) }
  $cliArgs += @("--", "bash", "-c", $script)
  if ($DryRun) {
    Write-Host "runai training standard submit $name -p $Project ... ($($res.Count) network resources) -- bash -c <script>"
  } else {
    Write-Host "submitting $name"
    & runai @cliArgs
  }
}
