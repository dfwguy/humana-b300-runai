# Instructions for Claude Code: Humana B300 Run:ai validation

You are helping validate a Humana DGX B300 cluster through Run:ai from a Humana-managed CloudPC.
Work from this folder. Be concise and show commands before running anything that changes cluster state.

## Facts
- Run:ai UI/API: `https://runai.dgxsd01.adc.humana.com`. Project `nvis-validation` (128 GPU quota). Namespace `runai-nvis-validation`.
- 16 DGX B300 nodes x 8 GPUs = 128 GPUs. Air-gapped: no internet inside the cluster, images come from the cluster registry.
- Default image `nvcr.io/nvidia/pytorch:26.04-py3` (contains `/usr/local/bin/all_reduce_perf_mpi`). Everything site-specific is in `config/site.json`.
- Node names (`dgx001`..`dgx016`), network names and resource names are **unverified**. Confirm before depending on them.
- Read `docs/TONIGHT-PLAN.md` for the order of tests and `docs/FINDINGS.md` for open questions.

## Rules
1. Never write passwords, tokens, API keys, private keys or the Run:ai admin login into any file, commit or prompt. The user runs `runai login` themselves.
2. Humana policy hooks run on this machine. Do not try to get around a hook, rename a binary, or use a wrapper to run something that was blocked. Tell the user what was blocked and why it matters.
3. You cannot run `ssh`. If a step needs the jump box, give the user the exact command to run with `! ssh jb "..."` or in their own terminal, and ask them to paste the output.
4. No downloads from the internet. If something is missing, say what it is and where it should come from.
5. Use the `runai` CLI for submissions. Before using any subcommand or flag, run its `--help` and use what this CLI version actually accepts. The CLI takes flags, not YAML. Translate `yaml/` into flags when needed (`--workers`, `-g`, `--annotation`, `--extended-resource`, `--capability`, `-e`).
6. Only delete workloads you created in this session (names starting `smoke`, `nccl-`, `health-`, `mnubergemm-`, `nemo56b-`). Ask first about anything else in the project.
7. The project quota is 128 GPUs and each test pod takes a whole node. Run one group at a time; do not start the 16-node job while single-node jobs are running.
8. Do not put customer data from other engagements into this folder, a prompt or a workload.

## Workflow for each test group
1. State what you will run and which nodes or workloads it touches.
2. Submit, then watch status until it finishes or fails.
3. Save each log to `results/<yyyymmdd>/<workload>.log`.
4. Run `python scripts/summarize.py results/<yyyymmdd> --csv results/<yyyymmdd>/summary.csv` if Python exists; otherwise read the logs for `Avg bus bandwidth`, `Out of bounds values`, `RESULT=`, `SPCX_ASSIGNED=`.
5. Report: passed, flagged, failed, with the node for each, and the number to compare with the table in `docs/TONIGHT-PLAN.md`.

## When a pod does not start
Check in this order: GPU count (another workload holding a GPU on that node), image pull error, a network or resource name that does not exist (Pending with "Insufficient nvidia.com/..."), quota, a policy rejecting `hostPath` or capabilities. Report what the CLI or UI shows; do not guess.

## Editing the manifests
Edit `config/site.json`, then run `python scripts/render.py`. If Python is missing, search and replace the same value across `yaml/` and `cli/` and tell the user you did so. Keep the single-node and multi-node settings consistent.
