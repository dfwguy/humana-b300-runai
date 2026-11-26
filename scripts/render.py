#!/usr/bin/env python3
"""Render every manifest under yaml/ from config/site.json.

Usage:   python scripts/render.py
Needs only the Python standard library. Re-run after editing config/site.json.
Generated files carry a header saying so; edit the config, not the YAML.
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = json.load(open(os.path.join(ROOT, "config", "site.json")))
OUT = os.path.join(ROOT, "yaml")

NS, PROJ, SCHED = CFG["namespace"], CFG["project"], CFG["scheduler_name"]
G = CFG["gpus_per_node"]
IMG, NEMO, MNU = CFG["image_pytorch"], CFG["image_nemo"], CFG["image_mnubergemm"]
NODES, FOUR = CFG["nodes"], CFG["four_node_set"]
NETS, NETRES = ",".join(CFG["networks"]), CFG["network_resources"]
MAXB = CFG["nccl_max_bytes"]


# ----------------------------------------------------------------- helpers
def y(text, n):
    """Indent every non-blank line by n spaces."""
    pad = " " * n
    return "\n".join((pad + l) if l.strip() else l for l in text.split("\n"))


def block(script, n):
    """YAML literal block scalar for a list item whose '-' sits at column n."""
    return "|\n" + y(script, n + 2)


def env_block(env, n):
    lines = [f"- name: {k}\n  value: {json.dumps(str(v))}" for k, v in env.items()]
    return y("\n".join(lines), n)


def labels(test, extra=None):
    d = {"project": PROJ, "app.kubernetes.io/part-of": "humana-b300-validation", "test": test}
    if extra:
        d.update(extra)
    return d


def labels_block(d, n):
    return y("\n".join(f"{k}: {json.dumps(v)}" for k, v in d.items()), n)


def host_affinity(nodes, n):
    vals = "\n".join(f"          - {x}" for x in nodes)
    text = (
        "affinity:\n"
        "  nodeAffinity:\n"
        "    requiredDuringSchedulingIgnoredDuringExecution:\n"
        "      nodeSelectorTerms:\n"
        "      - matchExpressions:\n"
        "        - key: kubernetes.io/hostname\n"
        "          operator: In\n"
        "          values:\n" + vals
    )
    return y(text, n)


def resources(gpus, with_nets, n):
    lines = [f"    nvidia.com/gpu: {gpus}"]
    if with_nets:
        lines += [f'    {r}: "1"' for r in NETRES]
    return y("resources:\n  limits:\n" + "\n".join(lines), n)


def volumes(extra_hostpaths=None):
    vols = ["- name: dshm\n  emptyDir:\n    medium: Memory\n    sizeLimit: 32Gi"]
    mounts = ["- name: dshm\n  mountPath: /dev/shm"]
    for name, path, mp in (extra_hostpaths or []):
        vols.append(f"- name: {name}\n  hostPath:\n    path: {path}\n    type: DirectoryOrCreate")
        mounts.append(f"- name: {name}\n  mountPath: {mp}")
    return "\n".join(vols), "\n".join(mounts)


def results_mount():
    return [("results", CFG["results_hostpath"], "/results")] if CFG.get("results_hostpath") else []


def write(path, text):
    full = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", newline="\n") as f:
        f.write(text.rstrip("\n") + "\n")


# ----------------------------------------------------------------- environments
NVLINK_ENV = {"NCCL_DEBUG": "INFO", "NCCL_P2P_LEVEL": "NVL", "NCCL_NVLS_ENABLE": "1",
              "NCCL_IB_DISABLE": "1", "NCCL_MNNVL_ENABLE": "0"}

# Spectrum-X RoCE tuning from the Run:ai 2.23 validation page (roce-nccl-test.yaml)
SPX_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET", "NCCL_NET_PLUGIN": "spcx",
           "NCCL_IB_HCA": "roce_", "NCCL_SOCKET_IFNAME": "eth0", "UCX_TLS": "tcp",
           "NCCL_IB_TC": "96", "NCCL_CROSS_NIC": "0", "NCCL_IB_MERGE_NICS": "0",
           "NCCL_NET_MERGE_LEVEL": "PIX", "NCCL_NCHANNELS_PER_NET_PEER": "4",
           "NCCL_P2P_NET_CHUNKSIZE": "262144"}

# Single-node Spectrum-X test: force traffic onto the NICs. Same toggles as the Slurm job
# jobs/b300-spx/allreduce-nonvlink-pytorch.sbatch in auto-action-trigger.
SPX_1NODE_EXTRA = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                   "NCCL_MNNVL_ENABLE": "0", "NCCL_PXN_DISABLE": "1", "NCCL_IB_GID_INDEX": "3"}

SPX_EXPORTS = "export LD_LIBRARY_PATH=/opt/hpcx/nccl_spectrum-x_plugin/lib:/usr/local/lib:$LD_LIBRARY_PATH"
SPX_XFLAGS = " ".join(f"-x {k}" for k in list(SPX_ENV) + list(SPX_1NODE_EXTRA) + ["LD_LIBRARY_PATH"])

# ----------------------------------------------------------------- shell scripts run inside the pods
PREFLIGHT = """echo "RUN=__RUN__ NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU')
echo "preflight: $(hostname) gpus=${GPUS} cores=$(nproc)"
if [ "$GPUS" -ne __G__ ]; then echo "RESULT=FAIL reason=gpu_count_${GPUS}"; exit 1; fi"""

NVLINK_SCRIPT = PREFLIGHT + """
nvidia-smi topo -m
mpirun --allow-run-as-root --bind-to none -np __G__ \\
  -x NCCL_DEBUG -x NCCL_P2P_LEVEL -x NCCL_NVLS_ENABLE -x NCCL_IB_DISABLE -x NCCL_MNNVL_ENABLE \\
  /usr/local/bin/all_reduce_perf_mpi -b 8 -e __MAXB__ -f 2 -g 1 -w 2 --iters 10 -c 10 2>&1 | tee /tmp/nccl.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mpirun_rc_${rc}"; fi
exit $rc"""

SPX_SCRIPT = PREFLIGHT + "\n" + SPX_EXPORTS + """
mpirun --allow-run-as-root --bind-to none -np __G__ \\
  __XFLAGS__ \\
  /usr/local/bin/all_reduce_perf_mpi -b 8 -e __MAXB__ -f 2 -g 1 -w 2 --iters 4 -c 10 2>&1 | tee /tmp/nccl.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
echo "SPCX_ASSIGNED=$(grep -c 'Assigned NET plugin SPCX' /tmp/nccl.log)"
echo "GDRDMA_CHANNELS=$(grep -c 'GDRDMA' /tmp/nccl.log)"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mpirun_rc_${rc}"; fi
exit $rc"""

HEALTH_SCRIPT = """echo "RUN=__RUN__ NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU')
echo "gpus=${GPUS} cores=$(nproc) mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
nvidia-smi --query-gpu=index,name,pci.bus_id,temperature.gpu,power.draw,clocks.sm,clocks.max.sm,memory.total,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi topo -m
nvidia-smi nvlink -s 2>&1 | head -80
nvidia-smi -q -d PERFORMANCE 2>&1 | grep -iE 'GPU [0-9A-F]|Active|Reasons' | head -80
ls /dev/infiniband 2>/dev/null | head -40
if [ "$GPUS" -eq __G__ ]; then echo "RESULT=PASS"; else echo "RESULT=FAIL reason=gpu_count_${GPUS}"; exit 1; fi"""

MNU_SCRIPT = PREFLIGHT + """
export NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL NCCL_CUMEM_ENABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_MNNVL_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_MIN_CTAS=16 NCCL_NVLS_ENABLE=1 UCX_TLS=tcp
BIN=$(command -v mnubergemm || echo ./mnubergemm)
if [ ! -x "$BIN" ]; then echo "RESULT=NOT_READY reason=mnubergemm_binary_not_found_in_image"; exit 2; fi
mpirun --allow-run-as-root --bind-to none -np __G__ -x NCCL_DEBUG -x NCCL_P2P_LEVEL -x NCCL_CUMEM_ENABLE -x NCCL_CUMEM_HOST_ENABLE -x NCCL_MNNVL_ENABLE -x NCCL_IB_DISABLE -x NCCL_MIN_CTAS -x NCCL_NVLS_ENABLE -x UCX_TLS \\
  "$BIN" --time_to_run __SECS__ --workload GNC --freq 500 --duty 1.0 --max_workload 65536 --MM_max_workload 65536 \\
  --NET_sm_count=32 --MM_M 0 --MM_N 0 --MM_type SX_SX_SSS --MM_sm_count 112 --NET_link_order scatter_sum \\
  --NET_size 20480000 --CE_type H --CE_size 1048576 2>&1 | tee /tmp/mnu.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mnubergemm_rc_${rc}"; fi
exit $rc"""

NEMO_SCRIPT = """echo "RUN=__RUN__ NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
if [ ! -f /recipe/run_nemo56b.sh ]; then
  echo "RESULT=NOT_READY reason=/recipe/run_nemo56b.sh_missing (see yaml/40-nemo56b/recipe/README.txt)"
  exit 2
fi
export NNODES=__NNODES__ NPROC_PER_NODE=__G__
export NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-localhost} MASTER_PORT=${MASTER_PORT:-29500}
bash /recipe/run_nemo56b.sh 2>&1 | tee /tmp/nemo.log
rc=${PIPESTATUS[0]}
echo "exit status: $rc"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=nemo_rc_${rc}"; fi
exit $rc"""

# torch.distributed all_reduce benchmark: needs no sshd and no MPI launcher.
# Output rows copy the nccl-tests columns so scripts/summarize.py reads both.
TORCH_BENCH = r'''import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
maxb = os.environ.get("BENCH_MAX", "16G")
mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
top = int(maxb[:-1]) * mult[maxb[-1]] if maxb[-1] in mult else int(maxb)
sizes, s = [], 8
while s <= top:
    sizes.append(s)
    s *= 2
warm, iters, bws, bad = 2, 10, [], 0
if rank == 0:
    print("# torch.distributed all_reduce bench: ranks=%d (in-place; columns duplicated for parser compatibility)" % world, flush=True)
    print("#  size count type redop root time algbw busbw #wrong time algbw busbw #wrong", flush=True)
for size in sizes:
    n = max(size // 4, 1)
    x = torch.ones(n, dtype=torch.float32, device="cuda")
    for _ in range(warm):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    ok = abs(x[0].item() - float(world) ** warm) <= 1e-3 * float(world) ** warm
    bad += 0 if ok else 1
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    alg = size / dt / 1e9
    bus = alg * 2 * (world - 1) / world
    bws.append(bus)
    if rank == 0:
        us = dt * 1e6
        w = 0 if ok else 1
        print("%d %d float sum -1 %.2f %.2f %.2f %d %.2f %.2f %.2f %d" % (size, n, us, alg, bus, w, us, alg, bus, w), flush=True)
    del x
if rank == 0:
    print("# Out of bounds values : %d %s" % (bad, "OK" if bad == 0 else "FAILED"), flush=True)
    print("# Avg bus bandwidth    : %.3f" % (sum(bws) / len(bws)), flush=True)
dist.destroy_process_group()
'''


def fill(s, **kw):
    s = s.replace("__G__", str(G)).replace("__MAXB__", MAXB).replace("__XFLAGS__", SPX_XFLAGS)
    s = s.replace("__SECS__", str(CFG["mnubergemm_seconds"]))
    for k, v in kw.items():
        s = s.replace(f"__{k}__", str(v))
    return s


# ----------------------------------------------------------------- Job: one pod pinned to one node
def job_one(name, test, node, image, script, env, nets=False, extra_hostpaths=None, caps=False):
    vols, mounts = volumes(extra_hostpaths)
    sec = ("securityContext:\n          runAsUser: 0\n          capabilities:\n"
           "            add: [\"NET_ADMIN\",\"NET_RAW\",\"IPC_LOCK\",\"SYS_RESOURCE\"]"
           if caps else "securityContext:\n          runAsUser: 0")
    ann = f"      annotations:\n        k8s.v1.cni.cncf.io/networks: \"{NETS}\"\n" if nets else ""
    return f"""# Generated by scripts/render.py from config/site.json. Edit the config, not this file.
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NS}
  labels:
{labels_block(labels(test, {"node": node}), 4)}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 172800
  template:
    metadata:
      labels:
{labels_block(labels(test, {"node": node}), 8)}
{ann}    spec:
      schedulerName: {SCHED}
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: {node}
      containers:
      - name: test
        image: {image}
        imagePullPolicy: IfNotPresent
        {sec}
        command: ["/bin/bash", "-lc"]
        args:
        - {block(script, 8)}
        env:
{env_block(env, 8)}
        {resources(G, nets, 8).lstrip()}
        volumeMounts:
{y(mounts, 8)}
      volumes:
{y(vols, 6)}
"""


# ----------------------------------------------------------------- MPIJob (Run:ai doc pattern)
def mpijob(name, test, nodes):
    n = len(nodes)
    launcher_script = f"""echo "RUN={name} NODES={n} RANKS={n*G} DATE=$(date -u +%FT%TZ)"
{SPX_EXPORTS}
mpirun --allow-run-as-root \\
  --mca plm_rsh_args "-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \\
  --bind-to core --map-by ppr:{G}:node -np {n*G} \\
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET -x NCCL_NET_PLUGIN=spcx -x LD_LIBRARY_PATH \\
  -x NCCL_IB_HCA=roce_ -x NCCL_SOCKET_IFNAME=eth0 -x UCX_TLS=tcp -x NCCL_IB_TC=96 -x NCCL_CROSS_NIC=0 \\
  -x NCCL_IB_MERGE_NICS=0 -x NCCL_NET_MERGE_LEVEL=PIX -x NCCL_NCHANNELS_PER_NET_PEER=4 -x NCCL_P2P_NET_CHUNKSIZE=262144 \\
  /usr/local/bin/all_reduce_perf_mpi -b 8 -e {MAXB} -f 2 -g 1 -w 2 --iters 4 -c 10 2>&1 | tee /tmp/nccl.log
rc=${{PIPESTATUS[0]}}
echo "exit status: $rc"
echo "SPCX_ASSIGNED=$(grep -c 'Assigned NET plugin SPCX' /tmp/nccl.log)"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=mpirun_rc_${{rc}}"; fi
exit $rc"""
    worker_script = """if ! command -v sshd >/dev/null 2>&1; then
  echo "sshd not in image; trying apt (fails on an air-gapped cluster - see docs/AIR-GAP-SSH.md)"
  apt-get update -qq && apt-get install -y -q openssh-server || { echo "RESULT=FAIL reason=no_sshd"; sleep 600; exit 1; }
fi
mkdir -p /run/sshd && ssh-keygen -A && exec /usr/sbin/sshd -De -p 2222 -o StrictModes=no"""
    vols, mounts = volumes()
    return f"""# Generated by scripts/render.py. Multi-node NCCL over Spectrum-X RoCE: {n} nodes x {G} GPUs = {n*G} ranks.
# Pattern from the Run:ai 2.23 BCM validation page (roce-nccl-test.yaml), which passed on this cluster.
# Needs an image that contains sshd (docs/AIR-GAP-SSH.md). Without one, use the PyTorchJob variant.
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: {name}
  namespace: {NS}
  labels:
{labels_block(labels(test), 4)}
spec:
  slotsPerWorker: {G}
  launcherCreationPolicy: WaitForWorkersReady
  runPolicy:
    cleanPodPolicy: Running
  sshAuthMountPath: /root/.ssh
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        metadata:
          labels:
{labels_block(labels(test), 12)}
        spec:
          schedulerName: {SCHED}
          restartPolicy: OnFailure
          containers:
          - name: mpi-launcher
            image: {IMG}
            securityContext:
              runAsUser: 0
            env:
            - name: OMPI_ALLOW_RUN_AS_ROOT
              value: "1"
            - name: OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
              value: "1"
            command: ["/bin/bash", "-lc"]
            args:
            - {block(launcher_script, 12)}
            volumeMounts:
{y(mounts, 12)}
          volumes:
{y(vols, 10)}
    Worker:
      replicas: {n}
      template:
        metadata:
          labels:
{labels_block(labels(test), 12)}
          annotations:
            k8s.v1.cni.cncf.io/networks: "{NETS}"
        spec:
          schedulerName: {SCHED}
          restartPolicy: OnFailure
{host_affinity(nodes, 10)}
          containers:
          - name: mpi-worker
            image: {IMG}
            securityContext:
              runAsUser: 0
              capabilities:
                add: ["NET_ADMIN","NET_RAW","IPC_LOCK","SYS_RESOURCE"]
            command: ["/bin/bash", "-lc"]
            args:
            - {block(worker_script, 12)}
{resources(G, True, 12)}
            volumeMounts:
{y(mounts, 12)}
          volumes:
{y(vols, 10)}
"""


# ----------------------------------------------------------------- PyTorchJob (no sshd needed)
def pytorchjob(name, test, nodes, image, script, env, nets, note):
    n = len(nodes)
    extra = [("recipe", CFG["nemo_recipe_hostpath"], "/recipe")] if test.startswith("nemo") else None
    vols, mounts = volumes(extra)
    ann = f"          annotations:\n            k8s.v1.cni.cncf.io/networks: \"{NETS}\"\n" if nets else ""

    def replica(role, count):
        return f"""    {role}:
      replicas: {count}
      restartPolicy: Never
      template:
        metadata:
          labels:
{labels_block(labels(test), 12)}
{ann}        spec:
          schedulerName: {SCHED}
{host_affinity(nodes, 10)}
          containers:
          - name: pytorch
            image: {image}
            imagePullPolicy: IfNotPresent
            securityContext:
              runAsUser: 0
              capabilities:
                add: ["NET_ADMIN","NET_RAW","IPC_LOCK","SYS_RESOURCE"]
            command: ["/bin/bash", "-lc"]
            args:
            - {block(script, 12)}
            env:
{env_block(env, 12)}
{resources(G, nets, 12)}
            volumeMounts:
{y(mounts, 12)}
          volumes:
{y(vols, 10)}"""

    workers = replica("Worker", n - 1) if n > 1 else ""
    return f"""# Generated by scripts/render.py. {note}
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {name}
  namespace: {NS}
  labels:
{labels_block(labels(test), 4)}
spec:
  pytorchReplicaSpecs:
{replica("Master", 1)}
{workers}
"""


def torch_script(name, n):
    return f"""echo "RUN={name} NODES={n} RANKS={n*G} NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
{SPX_EXPORTS}
cat > /tmp/nccl_bench.py <<'PYEOF'
{TORCH_BENCH}PYEOF
export BENCH_MAX={MAXB}
torchrun --nnodes={n} --nproc_per_node={G} \\
  --node_rank=${{PET_NODE_RANK:-${{RANK:-0}}}} \\
  --master_addr=${{MASTER_ADDR}} --master_port=${{MASTER_PORT:-29500}} \\
  /tmp/nccl_bench.py 2>&1 | tee /tmp/nccl.log
rc=${{PIPESTATUS[0]}}
echo "exit status: $rc"
echo "SPCX_ASSIGNED=$(grep -c 'Assigned NET plugin SPCX' /tmp/nccl.log)"
if [ "$rc" -eq 0 ]; then echo "RESULT=DONE"; else echo "RESULT=FAIL reason=torchrun_rc_${{rc}}"; fi
exit $rc"""


# ----------------------------------------------------------------- main
def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    results = results_mount()
    recipe = [("recipe", CFG["nemo_recipe_hostpath"], "/recipe")]

    smoke = job_one(
        "smoke-1node", "smoke", "ANY", IMG,
        fill("""echo "RUN=smoke NODE=$(hostname) DATE=$(date -u +%FT%TZ)"
nvidia-smi -L
GPUS=$(nvidia-smi -L | grep -c '^GPU')
ls -l /usr/local/bin/all_reduce_perf_mpi
which mpirun
echo "gpus=${GPUS}"
if [ "$GPUS" -eq __G__ ]; then echo "RESULT=PASS"; else echo "RESULT=FAIL"; exit 1; fi"""),
        {"NCCL_DEBUG": "WARN"})
    smoke = smoke.replace("      nodeSelector:\n        kubernetes.io/hostname: ANY\n", "")
    write("00-smoke/smoke-1node.yaml", smoke)

    for node in NODES:
        write(f"10-nccl-1node-nvlink/nccl-nvlink-{node}.yaml", job_one(
            f"nccl-nvlink-{node}", "nccl-1node-nvlink", node, IMG,
            fill(NVLINK_SCRIPT, RUN=f"nccl-nvlink-{node}"), NVLINK_ENV, extra_hostpaths=results))
        write(f"11-nccl-1node-spx/nccl-spx-{node}.yaml", job_one(
            f"nccl-spx-{node}", "nccl-1node-spx", node, IMG,
            fill(SPX_SCRIPT, RUN=f"nccl-spx-{node}"), {**SPX_ENV, **SPX_1NODE_EXTRA},
            nets=True, extra_hostpaths=results, caps=True))
        write(f"30-health-mnubergemm/health-{node}.yaml", job_one(
            f"health-{node}", "health", node, IMG, fill(HEALTH_SCRIPT, RUN=f"health-{node}"),
            {"NCCL_DEBUG": "WARN"}, extra_hostpaths=results))
        write(f"30-health-mnubergemm/mnubergemm-{node}.yaml", job_one(
            f"mnubergemm-{node}", "mnubergemm", node, MNU, fill(MNU_SCRIPT, RUN=f"mnubergemm-{node}"),
            {"NCCL_DEBUG": "INFO"}, extra_hostpaths=results))
        write(f"40-nemo56b/nemo56b-1node-{node}.yaml", job_one(
            f"nemo56b-1node-{node}", "nemo56b-1node", node, NEMO,
            fill(NEMO_SCRIPT, RUN=f"nemo56b-1node-{node}", NNODES=1),
            {"NCCL_DEBUG": "WARN", "NCCL_NVLS_ENABLE": "0", "TRANSFORMERS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "False", "PYTHONUNBUFFERED": "1"},
            extra_hostpaths=recipe + results))

    write("20-nccl-multinode/nccl-4node-mpijob.yaml", mpijob("nccl-4node-mpi", "nccl-4node", FOUR))
    write("20-nccl-multinode/nccl-16node-mpijob.yaml", mpijob("nccl-16node-mpi", "nccl-16node", NODES))
    write("20-nccl-multinode/nccl-4node-pytorchjob.yaml", pytorchjob(
        "nccl-4node-torch", "nccl-4node", FOUR, IMG, torch_script("nccl-4node-torch", len(FOUR)), SPX_ENV, True,
        "4-node NCCL all_reduce with torchrun. Needs no sshd or MPI launcher, so it works air-gapped."))
    write("20-nccl-multinode/nccl-16node-pytorchjob.yaml", pytorchjob(
        "nccl-16node-torch", "nccl-16node", NODES, IMG, torch_script("nccl-16node-torch", len(NODES)), SPX_ENV, True,
        "16-node NCCL all_reduce with torchrun. Needs no sshd or MPI launcher, so it works air-gapped."))

    write("40-nemo56b/nemo56b-16node-pytorchjob.yaml", pytorchjob(
        "nemo56b-16node", "nemo56b-16node", NODES, NEMO,
        fill(NEMO_SCRIPT, RUN="nemo56b-16node", NNODES=len(NODES)),
        {**SPX_ENV, "NCCL_NVLS_ENABLE": "0", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "False",
         "PYTHONUNBUFFERED": "1"}, True,
        "NeMo 56B wrapper. NOT READY until the NeMo image and /recipe/run_nemo56b.sh are staged."))

    write("40-nemo56b/recipe/README.txt", """NeMo 56B needs three things that are not confirmed on the Humana cluster:

1. A NeMo container in the cluster registry (config/site.json default: nvcr.io/nvidia/nemo:25.07.01,
   the tag used on an earlier cluster; not confirmed present here).
2. The recipe (for example from the LLM Benchmarking Collection, github.com/NVIDIA/dgxc-benchmarking),
   plus its tokenizer and dataset, staged offline. The cluster has no internet.
3. A launcher script that reads the variables the YAML sets:
     NNODES, NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT
   and starts training, typically:
     torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE --node_rank $NODE_RANK \\
              --master_addr $MASTER_ADDR --master_port $MASTER_PORT ...

Put the launcher at <nemo_recipe_hostpath>/run_nemo56b.sh (default /vast/validation/nemo56b/run_nemo56b.sh).
Until it exists, every nemo56b job exits with RESULT=NOT_READY and does not use the GPUs.
""")
    # Files for the Run:ai CLI route (flags instead of YAML): see scripts/cli-submit-*.ps1
    cli = os.path.join(ROOT, "cli")
    if os.path.isdir(cli):
        shutil.rmtree(cli)
    os.makedirs(cli)

    def cw(name, text):
        with open(os.path.join(cli, name), "w", newline="\n") as f:
            f.write(text.rstrip("\n") + "\n")

    cw("nccl-nvlink.sh", fill(NVLINK_SCRIPT, RUN="nccl-nvlink-cli"))
    cw("env-nvlink.txt", "\n".join(f"{k}={v}" for k, v in NVLINK_ENV.items()))
    cw("nccl-spx.sh", fill(SPX_SCRIPT, RUN="nccl-spx-cli"))
    cw("env-spx.txt", "\n".join(f"{k}={v}" for k, v in {**SPX_ENV, **SPX_1NODE_EXTRA}.items()))
    cw("networks-annotation.txt", NETS)
    cw("extended-resources.txt", "\n".join(f"{r}=1" for r in NETRES))
    print("rendered into", OUT)


if __name__ == "__main__":
    main()
