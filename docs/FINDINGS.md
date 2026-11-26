# Where the containers are: what is known and what is not

Short answer: **not confirmed anywhere yet.** The evidence points to three places. Check them with the commands below.

| Question | Best evidence | Confidence |
|---|---|---|
| Where did the team's containers go? | Alyssa and Dejaun downloaded them and uploaded about 88 GB in 3 folders of about 29 GB to the Humana "cluster-onboarding" share (Slack, 23 Nov). Mike Griffin said Steve Pierce or Andrew Scott would move the data to the jump server. | High that they were uploaded and a move was requested; **not confirmed that it happened** |
| Where would they be on the jump box? | Dejaun (Slack, 25 Nov): project files live under `/opt/data/nvis` (1 TB volume); the Run:ai CLI under `/opt/data/nvis/humana-airgapped/runai-bin`; results under `/opt/data/nvis/humana-airgapped/results`. Your `df` on 25 Nov showed 236 GB used and 764 GB free on `/opt/data`, which fits a large staged set. | Medium |
| Are they on VAST? | Matthew Pitterle (Teams, 24 Nov): VAST CSI is not set up yet; use bind mounts and a 1777 directory `/vast/validation`. Harish (Slack): the jump box may not be on the VAST NFS network. Nothing says containers are on VAST. | Low |
| Which PyTorch container does Michael Schwarz use? | His 24 Nov email: the Run:ai RoCE NCCL validation passed using the doc manifest, which uses `nvcr.io/nvidia/pytorch:26.04-py3`; the Run:ai quick starts used the "local pytorch:26.04-py3 image". He also says the OVA job needs `nvcr.io/nvidia/pytorch:25.06-py3`, which is **not** in the air-gapped registry. Slack mentions an "older version", which conflicts. | Medium: 26.04 is in the cluster registry; the exact registry path is not stated |
| How do pods get images? | The cluster pulls from the BCM registry (Michael fixed the registry CA for Knative and imported `library/alpine:3.10` and the Kourier images). Because a missing `nvcr.io/...:25.06-py3` is reported as "not in the registry", `nvcr.io/...` names probably resolve through a mirror. | Medium |

## Check it yourself

1. **Jump box (read-only):** `bash scripts/jumpbox-locate-containers.sh | tee /opt/data/nvis/locate.txt`.
   It lists large files and archives under `/opt/data`, mounts (is VAST visible?), disk space and tools.
2. **Run:ai UI:** Workload manager, Workloads, open one of Michael's workloads (created by his user), and copy the Image field.
3. **Ask Michael or Matthew:** "What is the registry host and repository path for `pytorch:26.04-py3`, and how do `nvcr.io/...` names map to it?"

## Hosting options for your own containers

| Option | Works for | Notes |
|---|---|---|
| Push to the BCM registry | Images | Recommended. `scripts/jumpbox-stage-containers.sh` (not run). Needs the registry path, a login and the registry CA from Matthew or Michael. Use `skopeo` so nothing lands in the 10 GB `/home`. |
| VAST at `/vast/validation` via hostPath | Data, recipes, results | Works for files a container reads. It does **not** make an image runnable: Kubernetes cannot start a container from a tarball on NFS without a per-node import. |
| Per-node `ctr -n k8s.io images import` | Images | Admin-only, 16 nodes by hand. Last resort. |

So "host in VAST or tell Run:ai to use it" is workable for the NeMo recipe and datasets (`config/site.json: nemo_recipe_hostpath`) but not for container images.

## Other things learned

- The jump box `/home` is 10 GB and shared; put everything large under `/opt/data/nvis`.
- 16 nodes x 8 GPUs = 128 GPUs (BOM, HLD, Run:ai dashboard). "256 GPUs" appears only in project titles.
- Run:ai project `nvis-validation` has a 128-GPU quota; its Kubernetes namespace is `runai-nvis-validation`.
- Multi-node MPI normally installs `openssh-server` with `apt-get` inside the pod, which cannot work air-gapped.
