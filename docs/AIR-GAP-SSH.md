# Multi-node MPI on an air-gapped cluster

`MPIJob` launches ranks by SSH from the launcher pod into the worker pods. The NGC PyTorch image has no SSH server.
The Run:ai doc's workers run `apt-get install openssh-server`, which needs internet. Michael's multi-node RoCE test passed,
so he solved this somehow; the emails do not say how. Options, best first:

1. **Ask Michael.** One question: "What image or apt mirror did the RoCE MPIJob workers use for sshd?"
2. **Use the PyTorchJob versions** (`yaml/20-nccl-4node-pytorchjob.yaml`, `yaml/20-nccl-16node-pytorchjob.yaml`).
   They run a small `torch.distributed` all_reduce benchmark launched with `torchrun`, so they need no sshd and no MPI.
   Same fabric, same NCCL, same bus-bandwidth formula (2(n-1)/n x size / time). The script is new and has not been run;
   compare its numbers with the nccl-tests reference numbers before trusting it.
3. **An image that already has sshd**, staged in the registry (see `docs/FINDINGS.md`). Then put its name in `config/site.json`.
4. **An apt mirror** on a head node reachable from the pods. Needs Humana/NVIDIA admin work.
