#!/usr/bin/env python
"""
Pipelined / chunked Boltz-2 runner — keeps GPUs busy during MSA generation.

Plain `boltz predict <dir> --devices 2` front-loads ALL MSA generation (network/CPU, GPUs idle) and
only then folds everything, so no output appears until every MSA is done. This runner instead splits
the inputs into small chunks and drives **one asynchronous single-GPU Boltz stream per GPU**. Because
the streams are independent, while one stream is folding on its GPU the other is typically in its
MSA-fetch phase — so the MSA (I/O-bound) and folding (GPU-bound) phases of the two streams overlap,
GPU idle time drops, and predictions are written continuously (one chunk at a time). Resumable:
pairs whose output already exists are skipped.

Output is written into the same layout Boltz produces (boltz_results_<chunk>/predictions/<name>/...);
`tools/boltz_to_af3_summary.py` globs recursively, so it consumes this output unchanged.

Usage:
  python tools/boltz_pipeline_run.py --inputs <dir> --out <dir> --cache <boltz_cache> \
      [--gpus 0,1] [--chunk 16] [--diffusion-samples 5] [--max-parallel-samples 1] [--extra "..."]

Notes: uses only the `boltz` CLI (must be on PATH / in the active env). With --use-msa-server (default),
two streams query the public ColabFold server concurrently — fine for moderate loads; for very large
jobs consider precomputed MSAs. Pass --no-msa-server if your FASTAs carry MSA paths in their headers.
"""
import argparse, glob, os, queue, subprocess, sys, threading, time

def already_done(out, name):
    return bool(glob.glob(os.path.join(out, "**", "predictions", name, f"confidence_{name}_model_0.json"),
                          recursive=True))

def worker(gpu, q, args, log_lock, counters):
    while True:
        try:
            idx, chunk = q.get_nowait()
        except queue.Empty:
            return
        cdir = os.path.join(args.out, "_chunks", f"chunk_{idx:04d}_gpu{gpu}")
        os.makedirs(cdir, exist_ok=True)
        for fp in chunk:                       # symlink the fastas into the chunk dir
            dst = os.path.join(cdir, os.path.basename(fp))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(fp), dst)
        cmd = ["boltz", "predict", cdir, "--devices", "1",
               "--diffusion_samples", str(args.diffusion_samples),
               "--max_parallel_samples", str(args.max_parallel_samples),
               "--out_dir", args.out, "--cache", args.cache]
        if args.use_msa_server:
            cmd.append("--use_msa_server")
        if args.extra:
            cmd += args.extra.split()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), BOLTZ_CACHE=args.cache)
        t0 = time.time()
        with log_lock:
            print(f"[gpu{gpu}] chunk {idx} ({len(chunk)} pairs) START", flush=True)
        lf = os.path.join(args.out, "_chunks", f"chunk_{idx:04d}_gpu{gpu}.log")
        with open(lf, "w") as fo:
            rc = subprocess.run(cmd, env=env, stdout=fo, stderr=subprocess.STDOUT).returncode
        with log_lock:
            counters["done"] += len(chunk)
            print(f"[gpu{gpu}] chunk {idx} DONE rc={rc} ({time.time()-t0:.0f}s)  "
                  f"[{counters['done']}/{counters['total']} pairs dispatched]", flush=True)
        q.task_done()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="dir of per-complex .fasta files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpus", default="0,1", help="comma list of GPU ids, one stream each")
    ap.add_argument("--chunk", type=int, default=16, help="pairs per chunk (smaller = finer overlap/output)")
    ap.add_argument("--diffusion-samples", type=int, default=5)
    ap.add_argument("--max-parallel-samples", type=int, default=1)
    ap.add_argument("--no-msa-server", dest="use_msa_server", action="store_false",
                    help="do NOT call the MSA server (FASTAs must supply their own MSA)")
    ap.add_argument("--extra", default="", help="extra args passed through to `boltz predict`")
    a = ap.parse_args()

    fastas = sorted(glob.glob(os.path.join(a.inputs, "*.fasta")))
    pending = [f for f in fastas if not already_done(a.out, os.path.splitext(os.path.basename(f))[0])]
    print(f"{len(fastas)} inputs | {len(fastas)-len(pending)} already done | {len(pending)} to run "
          f"| chunk={a.chunk} | gpus={a.gpus}", flush=True)
    if not pending:
        print("nothing to do."); return
    chunks = [pending[i:i+a.chunk] for i in range(0, len(pending), a.chunk)]
    q = queue.Queue()
    for i, c in enumerate(chunks):
        q.put((i, c))
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    counters = {"done": 0, "total": len(pending)}
    lock = threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, q, a, lock, counters), daemon=True) for g in gpus]
    for t in threads: t.start()
    for t in threads: t.join()
    ndone = sum(1 for f in fastas if already_done(a.out, os.path.splitext(os.path.basename(f))[0]))
    print(f"PIPELINE_DONE {ndone}/{len(fastas)} complexes have output  ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
