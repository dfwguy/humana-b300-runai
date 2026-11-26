# Humana B300 Run:ai validation kit

Ready-to-apply manifests and helper scripts for validating the Humana DGX B300 cluster
(16 nodes, 8 GPUs each, 128 GPUs) through Run:ai. Built on 26 Nov 2026 from:

- the Run:ai NCCL tutorial and the Run:ai 2.23 BCM "Validation Tests" page (the Spectrum-X RoCE test that passed here)
- Michael Schwarz's 24 and 25 Nov status emails (cluster state, image, networks, baselines)
- the `nvis-ova/auto-action-trigger` Slurm jobs (NCCL, non-NVLink NCCL, mnubergemm) re-expressed for Run:ai
- an earlier NVIDIA validation bundle (used for pod structure only; none of its files are included)

## Read this first: what is verified and what is not

| Area | Status |
|---|---|
| YAML syntax of all 86 files, and the shell/Python scripts inside them | Checked offline (PyYAML, `bash -n`, `ast.parse`) |
| Single-node NCCL over NVLink (`yaml/10-*`) | Not run. Uses the same flags as the Slurm job; no site-specific names beyond the node names |
| Node names `dgx001`..`dgx016` | **Unverified.** Confirm with `runai node list` |
| Network names and resource names (`default/pX-rY`, `nvidia.com/pX-rY`) | **Unverified.** Michael's email says `default/pX-rY`; the doc uses `rX-pY`. Confirm before `yaml/11`, `yaml/20`, `yaml/40` |
| Multi-node MPIJob (`yaml/20-*-mpijob.yaml`) | Doc pattern, but needs `sshd` in the image. The cluster is air-gapped (see `docs/AIR-GAP-SSH.md`) |
| Multi-node PyTorchJob (`yaml/20-*-pytorchjob.yaml`) | Needs no sshd. The torch benchmark is new code and has never run |
| NeMo 56B (`yaml/40-nemo56b`) | **Not ready.** Wrapper only. It needs a NeMo image, a recipe and data that are not confirmed on the cluster |
| mnubergemm (`yaml/30-*`) | Needs an NVIDIA-internal image that is not confirmed on the cluster |
| PowerShell scripts | **Untested** (no PowerShell was available when written). Check flags with `--help` first |

## Layout

```
config/site.json        every site-specific value; edit then run: python scripts/render.py
yaml/                   rendered manifests, apply in numeric order
  00-smoke              one unpinned pod: image pull, 8 GPUs, nccl binary present
  10-nccl-1node-nvlink  16 jobs, one per node
  11-nccl-1node-spx     16 jobs, one per node, NVLink excluded, NICs forced
  20-nccl-multinode     4-node and 16-node, MPIJob and PyTorchJob versions
  30-health-mnubergemm  16 health snapshots, 16 short mnubergemm runs
  40-nemo56b            wrapper jobs, NOT READY
cli/                    the same single-node tests as plain files for the Run:ai CLI
scripts/                render.py, summarize.py, PowerShell and jump-box helpers
docs/                   TONIGHT-PLAN, FINDINGS, AUTO-ACTION-TRIGGER-MAPPING, AIR-GAP-SSH
CLAUDE.md               instructions for Claude Code on the CloudPC
```

## Two ways to submit

1. **Run:ai CLI** (what you have on the CloudPC): the CLI takes flags, not YAML. `scripts/cli-submit-*.ps1`
   submit the single-node tests that way. Claude can translate the multi-node YAML into
   `runai training pytorch submit` / `runai training mpi submit` flags.
2. **kubectl apply** (what the validation tests used): `kubectl apply -f yaml/10-nccl-1node-nvlink/` from a
   machine with cluster access, such as the head node or jump box.

## Quick start

See `docs/TONIGHT-PLAN.md`. In short: confirm names (step 0), smoke test, 16 single-node NVLink runs,
summarize with `python scripts/summarize.py <log_dir>`, then Spectrum-X and multi-node.

Do not put passwords, tokens, keys or API secrets in this repository or in prompts.
