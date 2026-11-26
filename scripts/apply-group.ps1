<#
.SYNOPSIS
  Apply (or delete) a folder of manifests from yaml/ with kubectl. Use only where kubectl works
  against the cluster (for example from the jump box or head node, as the Run:ai validation tests were run).

.EXAMPLE
  .\scripts\apply-group.ps1 -Group 00-smoke
  .\scripts\apply-group.ps1 -Group 10-nccl-1node-nvlink
  .\scripts\apply-group.ps1 -Group 10-nccl-1node-nvlink -Delete
#>
param([Parameter(Mandatory)][string]$Group, [string]$Namespace = "runai-nvis-validation", [switch]$Delete)
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "yaml\$Group"
if (-not (Test-Path $dir)) { throw "no such group: $dir" }
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) { throw "kubectl not found; use the Run:ai CLI scripts instead" }
if ($Delete) { kubectl delete -f $dir --ignore-not-found } else { kubectl apply -f $dir }
kubectl get jobs,mpijobs,pytorchjobs -n $Namespace 2>$null
