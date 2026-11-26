# Repurposing auto-action-trigger (Slurm) for Run:ai

Source: `gitlab-master.nvidia.com/nvis-ova/auto-action-trigger` (default branch `main`). I read `AGENTS.md`,
`jobs/b300-spx/allreduce-pytorch.sbatch`, `allreduce-nonvlink-pytorch.sbatch`, `mnubergemm.sbatch`, `build_llmb.sh`
and the file listing. I did not read `settings.sh`, the parsers, the agent, the health checks or the docs folder.
The repo is a Slurm bring-up framework: a systemd timer and watchdog on each node run health checks, cache container
images, sequence benchmarks and burn-in, and write verdicts.

## Job mapping

| Slurm job | Run:ai equivalent here | What changed |
|---|---|---|
| `allreduce-pytorch.sbatch` | `yaml/10-nccl-1node-nvlink` | Same flags (`all_reduce_perf_mpi -b 8 -e 16G -f 2 -g 1`), P2P level NVL, NVLS on. `NCCL_MNNVL_ENABLE=0` because MNNVL does not apply to B300. The `srun`/enroot/PMIx wrapper became `mpirun` inside one pod. |
| `allreduce-nonvlink-pytorch.sbatch` | `yaml/11-nccl-1node-spx` | Same toggles (P2P and SHM disabled, NVLS off, `spcx` plugin, GID index 3). HCA discovery (`nccl-hca.sh`) is replaced by `NCCL_IB_HCA=roce_` plus the attached networks, as in the Run:ai validation manifest. |
| `alltoall-*` | not included | Same pattern: swap `all_reduce_perf_mpi` for `alltoall_perf_mpi` in the scripts in `scripts/render.py`. |
| `dcgmi-r1..r4` | not ported | Host-level diagnostics. Bryan Reese's guidance in Slack is to `pdsh` them from the head node. `yaml/30-health-*` takes an `nvidia-smi` snapshot only. |
| `mnubergemm.sbatch` | `yaml/30-mnubergemm-*` | Burn-in is 6.5 h x 8 runs in the original; here 600 s per node. Needs the NVIDIA-internal image `mnubergemm-26.03-v1.17` (x86), which is not confirmed on this cluster. |
| LLMB / NeMo (`build_llmb.sh`) | `yaml/40-nemo56b` | Wrapper only. The collection is installed on a networked head node, then staged offline; none of that is confirmed here. |

## What the framework does that this kit does not

- Always-on scheduling (systemd timer, watchdog), queue priority and `nice` ordering, drain and retry policy.
- Verdict parsing against baselines (`verdict-scan.sh`, `parsers/`). `scripts/summarize.py` replaces it with a simple rule: a result must finish with no wrong values, and any single-node run more than 10 percent below the fleet median is flagged.
- Copying images to each node (`file_copy.sh`). Run:ai pulls from the registry instead.
- Fleet reports (`report_*.sh`).

## A sensible next step

Port the framework's cycle as a Kubernetes CronJob or a small controller that submits the `yaml/10`, `yaml/30` jobs,
reads results with `summarize.py`, and writes a fleet report. Read the repo's `docs/explanation/architecture.md` and
`verdicts-and-the-cache.md` first; they carry the design rules that the Slurm version depends on.
