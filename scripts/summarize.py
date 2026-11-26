#!/usr/bin/env python3
"""Summarize NCCL / health / burn-in logs saved from the Run:ai workloads.

Usage:   python scripts/summarize.py <log_dir> [--csv out.csv] [--outlier 0.90]

Reads every *.log or *.txt in <log_dir> (one file per workload, as written by
scripts/collect-logs.ps1 or by `kubectl logs`). Understands:
  - nccl-tests output (all_reduce_perf_mpi) and the torch bench in the PyTorchJob YAMLs
  - the RESULT= / NODE= / SPCX_ASSIGNED= marker lines the YAMLs print
Flags any single-node run whose largest-size bus bandwidth is below <outlier> x the median.
Standard library only.
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys


def parse(path):
    r = {"file": os.path.basename(path), "run": "", "node": "", "result": "", "oob": "", "avg": None,
         "max_size": 0, "busbw_oop": None, "busbw_ip": None, "spcx": None, "gpus": None, "reason": ""}
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = re.search(r"RUN=(\S+)", line)
        if m and not r["run"]:
            r["run"] = m.group(1)
        m = re.search(r"NODE=(\S+)", line)
        if m and not r["node"]:
            r["node"] = m.group(1)
        m = re.search(r"preflight: (\S+) gpus=(\d+)", line)
        if m:
            r["node"] = r["node"] or m.group(1)
            r["gpus"] = int(m.group(2))
        m = re.search(r"RESULT=(\w+)(?: reason=(\S+))?", line)
        if m:
            r["result"] = m.group(1)
            r["reason"] = m.group(2) or r["reason"]
        m = re.search(r"SPCX_ASSIGNED=(\d+)", line)
        if m:
            r["spcx"] = int(m.group(1))
        m = re.search(r"Out of bounds values\s*:\s*(\d+)", line)
        if m:
            r["oob"] = int(m.group(1))
        m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", line)
        if m:
            r["avg"] = float(m.group(1))
        t = line.split()
        # nccl-tests row: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
        if len(t) >= 13 and t[0].isdigit() and t[1].isdigit():
            try:
                size = int(t[0])
                if size >= r["max_size"]:
                    r["max_size"] = size
                    r["busbw_oop"], r["busbw_ip"] = float(t[7]), float(t[11])
            except ValueError:
                pass
    return r


def verdict(r):
    if r["result"] in ("NOT_READY", "FAIL"):
        return r["result"] + (f" ({r['reason']})" if r["reason"] else "")
    if r["busbw_ip"] is None and r["result"] == "PASS":
        return "PASS"
    if r["busbw_ip"] is None:
        return "NO_DATA"
    if r["oob"] not in ("", 0):
        return "FAIL (wrong values)"
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--csv")
    ap.add_argument("--outlier", type=float, default=0.90)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.log_dir, "*.log")) + glob.glob(os.path.join(a.log_dir, "*.txt")))
    if not files:
        print("no .log or .txt files in", a.log_dir)
        return 1
    rows = [parse(f) for f in files]
    for r in rows:
        r["verdict"] = verdict(r)

    groups = {}
    for r in rows:
        key = re.sub(r"-?(dgx|node)?\d+$", "", r["run"] or r["file"]) or r["file"]
        groups.setdefault(key, []).append(r)

    print(f"{'file':38} {'node':10} {'maxsize':>12} {'busbw_ip':>9} {'avg_busbw':>9} {'spcx':>5}  verdict")
    for key in sorted(groups):
        vals = [r["busbw_ip"] for r in groups[key] if r["busbw_ip"] is not None and r["verdict"] == "OK"]
        med = statistics.median(vals) if len(vals) >= 3 else None
        for r in sorted(groups[key], key=lambda x: x["node"] or x["file"]):
            flag = ""
            if med and r["busbw_ip"] is not None and r["busbw_ip"] < a.outlier * med:
                flag = f"  <-- OUTLIER (<{a.outlier:.0%} of median {med:.1f})"
                r["verdict"] = "OUTLIER"
            bw = "" if r["busbw_ip"] is None else f"{r['busbw_ip']:.1f}"
            av = "" if r["avg"] is None else f"{r['avg']:.1f}"
            sp = "" if r["spcx"] is None else str(r["spcx"])
            print(f"{r['file'][:38]:38} {r['node'][:10]:10} {r['max_size']:>12} {bw:>9} {av:>9} {sp:>5}  {r['verdict']}{flag}")
        if med:
            print(f"  [{key}] median busbw at largest size: {med:.1f} GB/s over {len(vals)} runs")
        print()

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "run", "node", "gpus", "max_size_bytes", "busbw_oop", "busbw_ip", "avg_busbw", "oob", "spcx", "verdict"])
            for r in rows:
                w.writerow([r["file"], r["run"], r["node"], r["gpus"], r["max_size"], r["busbw_oop"], r["busbw_ip"],
                            r["avg"], r["oob"], r["spcx"], r["verdict"]])
        print("wrote", a.csv)
    bad = [r for r in rows if r["verdict"] not in ("OK", "PASS")]
    print(f"{len(rows)} logs, {len(rows) - len(bad)} ok, {len(bad)} need attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
