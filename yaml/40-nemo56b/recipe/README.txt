NeMo 56B needs three things that are not confirmed on the Humana cluster:

1. A NeMo container in the cluster registry (config/site.json default: nvcr.io/nvidia/nemo:25.07.01,
   the tag used on an earlier cluster; not confirmed present here).
2. The recipe (for example from the LLM Benchmarking Collection, github.com/NVIDIA/dgxc-benchmarking),
   plus its tokenizer and dataset, staged offline. The cluster has no internet.
3. A launcher script that reads the variables the YAML sets:
     NNODES, NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT
   and starts training, typically:
     torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE --node_rank $NODE_RANK \
              --master_addr $MASTER_ADDR --master_port $MASTER_PORT ...

Put the launcher at <nemo_recipe_hostpath>/run_nemo56b.sh (default /vast/validation/nemo56b/run_nemo56b.sh).
Until it exists, every nemo56b job exits with RESULT=NOT_READY and does not use the GPUs.
