echo "RUN=nccl-nvlink-cli NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU')
echo "preflight: $(hostname) gpus=${GPUS} cores=$(nproc)"
if [ "$GPUS" -ne 8 ]; then echo "RESULT=FAIL reason=gpu_count_${GPUS}"; exit 1; fi
nvidia-smi topo -m
mpirun --allow-run-as-root --bind-to none -np 8 \
  -x NCCL_DEBUG -x NCCL_P2P_LEVEL -x NCCL_NVLS_ENABLE -x NCCL_IB_DISABLE -x NCCL_MNNVL_ENABLE \
  /usr/local/bin/all_reduce_perf_mpi -b 8 -e 16G -f 2 -g 1 -w 2 --iters 10 -c 10 2>&1 | tee /tmp/nccl.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mpirun_rc_${rc}"; fi
exit $rc
