# Tonight's plan

Goal: 16 single-node NCCL runs, one 4-node run, one 16-node run, and NeMo 56B if it can run.
Order matters: each step proves something the next one needs.

## Step 0: confirm the facts (10 to 20 minutes)

1. `runai login` (type credentials yourself), then `runai version`. Set the project: `runai project set nvis-validation`.
2. Node names: `runai node list`. If they are not `dgx001`..`dgx016`, edit `config/site.json` and run `python scripts/render.py`.
3. Leftovers: the OVA job `nvis-test-single-node` held a GPU on nodes 013 and 014 and could not pull `nvcr.io/nvidia/pytorch:25.06-py3`. Delete it or those nodes cannot run an 8-GPU pod.
4. Image: the Run:ai validation passed with `nvcr.io/nvidia/pytorch:26.04-py3`. Open one of Michael Schwarz's workloads in the Run:ai UI and copy the exact Image value. Put it in `config/site.json` if it differs.
5. Network names and resources (only needed for steps 3 to 5): copy them from Michael's passing workload, or ask him. His email says networks are written `default/pX-rY`.
6. Is there an image with `sshd`? Ask Michael how his multi-node MPI test started its workers on an air-gapped cluster.

## Step 1: smoke test

```
runai training standard submit smoke -p nvis-validation -i nvcr.io/nvidia/pytorch:26.04-py3 -g 8 -- bash -c 'nvidia-smi -L; ls -l /usr/local/bin/all_reduce_perf_mpi; which mpirun'
```
or `kubectl apply -f yaml/00-smoke/`. It must show 8 GPUs and the nccl binary. If the image fails to pull, stop and fix that first.

## Step 2: single-node NCCL over NVLink on all 16 nodes

`.\scripts\cli-submit-nvlink-fleet.ps1 -DryRun`, then without `-DryRun`. Collect and summarize:

```
.\scripts\collect-logs.ps1 -Prefix nccl-nvlink- -Count 16
python scripts\summarize.py results\<date>
```
There is no published B300 NVLink baseline in the files I read. The summarizer flags any node whose largest-size bus bandwidth is below 90 percent of the fleet median. Treat an outlier or a failed run as a hardware or config lead.

## Step 3: single-node NCCL forced onto Spectrum-X (NVLink excluded)

Needs step 0.5. `.\scripts\cli-submit-spx-fleet.ps1`. Check `SPCX_ASSIGNED` is not 0 in each log.

## Step 4: 4-node NCCL, then step 5: 16-node NCCL

Use `yaml/20-nccl-4node-pytorchjob.yaml` (no sshd needed) or the MPIJob version if you have an sshd image. Run one test at a time.
Reference numbers from Michael Schwarz's runs on this cluster (Spectrum-X, pod networking):

| Run | Result |
|---|---|
| doc RoCE test, 2 nodes x 8 GPUs | avg bus bandwidth 166.72 (sweep 8 B to 8 GB) |
| node pairs | about 673 to 674 GB/s bus bandwidth at 8 GB |
| 14 nodes (112 GPUs) | 568 GB/s at 16 GB |
| 16 nodes (128 GPUs) | 576.6 GB/s at 16 GB; avg 164.6 inside a Run:ai project |

Treat results within a few percent of these as in line. The torch benchmark reports "avg bus bandwidth" averaged across all sizes, like nccl-tests does.

## Step 6: NeMo 56B (only if prerequisites exist)

Read `yaml/40-nemo56b/recipe/README.txt`. Nothing in the emails or Slack says a NeMo image or recipe is staged. If not, say so in the report instead of running it.

## Step 7: clean up and report

Delete test workloads, keep the logs and `summary.csv`. Report: nodes passed, nodes flagged, multi-node numbers against the table above, and what was not run and why.
