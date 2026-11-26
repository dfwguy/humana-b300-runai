echo "RUN=nccl-spx-cli NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU')
echo "preflight: $(hostname) gpus=${GPUS} cores=$(nproc)"
if [ "$GPUS" -ne 8 ]; then echo "RESULT=FAIL reason=gpu_count_${GPUS}"; exit 1; fi
export LD_LIBRARY_PATH=/opt/hpcx/nccl_spectrum-x_plugin/lib:/usr/local/lib:$LD_LIBRARY_PATH
mpirun --allow-run-as-root --bind-to none -np 8 \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME -x UCX_TLS -x NCCL_IB_TC -x NCCL_CROSS_NIC -x NCCL_IB_MERGE_NICS -x NCCL_NET_MERGE_LEVEL -x NCCL_NCHANNELS_PER_NET_PEER -x NCCL_P2P_NET_CHUNKSIZE -x NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE -x NCCL_NVLS_ENABLE -x NCCL_MNNVL_ENABLE -x NCCL_PXN_DISABLE -x NCCL_IB_GID_INDEX -x LD_LIBRARY_PATH \
  /usr/local/bin/all_reduce_perf_mpi -b 8 -e 16G -f 2 -g 1 -w 2 --iters 4 -c 10 2>&1 | tee /tmp/nccl.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
echo "SPCX_ASSIGNED=$(grep -c 'Assigned NET plugin SPCX' /tmp/nccl.log)"
echo "GDRDMA_CHANNELS=$(grep -c 'GDRDMA' /tmp/nccl.log)"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mpirun_rc_${rc}"; fi
exit $rc
