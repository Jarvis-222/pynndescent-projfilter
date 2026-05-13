#!/usr/bin/env python3
"""Sweep max_candidates on GIST100K. For each value, run vanilla and
filter (p_tau=0.95, m=16) and report construction recall, search recall,
QPS and dist-comp savings.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pynndescent

# Reuse helpers from bench_knng.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_knng import (
    DATASETS,
    load_fvecs,
    load_gt_bin,
    get_or_compute_search_gt,
    recall_at_k,
)


def one_run(data, n_neighbors, mc, seed, use_filter, m, p_tau,
            queries, search_gt, tree_init):
    t0 = time.perf_counter()
    idx = pynndescent.NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=mc,
        random_state=seed,
        verbose=False,
        tree_init=tree_init,
        use_projection_filter=use_filter,
        num_projections=m,
        filter_confidence=p_tau,
    )
    build_t = time.perf_counter() - t0
    cons_recall = recall_at_k(idx.neighbor_graph[0], gt_global, n_neighbors)
    tq = time.perf_counter()
    pred_q, _ = idx.query(queries, k=n_neighbors)
    qt = time.perf_counter() - tq
    search_recall = recall_at_k(pred_q, search_gt, n_neighbors)
    qps = queries.shape[0] / qt
    return {
        "build_t": build_t,
        "cons_recall": cons_recall,
        "search_recall": search_recall,
        "qps": qps,
        "dist_comps": idx.n_dist_comps,
        "filter_skips": idx.n_filter_skips,
    }


def main():
    global gt_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gist100k")
    ap.add_argument("--mcs", default="10,20,30,40,50,60",
                    help="comma-separated max_candidates values")
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--p-tau", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tree-init", action="store_true",
                    help="Disable RP-tree init (random init only)")
    args = ap.parse_args()
    mc_values = [int(x) for x in args.mcs.split(",")]
    tree_init = not args.no_tree_init

    cfg = DATASETS[args.dataset]
    print(f"Sweep on {args.dataset}, max_candidates in {mc_values}, "
          f"p_tau={args.p_tau}, m={args.m}, tree_init={tree_init}")

    print("Loading data + GT...", flush=True)
    data = load_fvecs(cfg["base"], n_limit=cfg["n_subset"])
    gt_global = load_gt_bin(cfg["gt"])
    queries = load_fvecs(cfg["queries"])
    search_gt = get_or_compute_search_gt(
        cfg["search_gt"], queries, data, args.n_neighbors
    )
    print(f"  data={data.shape} gt={gt_global.shape} queries={queries.shape}")

    # JIT warmup — exercise both branches once
    print("Warming up Numba JIT...", flush=True)
    rng = np.random.default_rng(42)
    warm = rng.standard_normal((200, 16)).astype(np.float32)
    _ = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                              random_state=0, tree_init=tree_init,
                              use_projection_filter=False)
    _ = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                              random_state=0, tree_init=tree_init,
                              use_projection_filter=True,
                              num_projections=args.m,
                              filter_confidence=args.p_tau)
    print("  done")

    rows = []
    for mc in mc_values:
        print(f"\n--- max_candidates = {mc} ---", flush=True)
        v = one_run(data, args.n_neighbors, mc, args.seed, False,
                    args.m, args.p_tau, queries, search_gt, tree_init)
        print(f"  vanilla:  build={v['build_t']:5.2f}s  "
              f"cons={v['cons_recall']:.4f}  "
              f"search={v['search_recall']:.4f}  "
              f"qps={v['qps']:4.0f}  "
              f"dist_comps={v['dist_comps']:,}")
        f = one_run(data, args.n_neighbors, mc, args.seed, True,
                    args.m, args.p_tau, queries, search_gt, tree_init)
        print(f"  filter:   build={f['build_t']:5.2f}s  "
              f"cons={f['cons_recall']:.4f}  "
              f"search={f['search_recall']:.4f}  "
              f"qps={f['qps']:4.0f}  "
              f"dist_comps={f['dist_comps']:,}  "
              f"skips={f['filter_skips']:,}")
        rows.append({"mc": mc, "v": v, "f": f})

    # Final summary table
    print("\n" + "=" * 100)
    print(f"SUMMARY  (dataset={args.dataset}, p_tau={args.p_tau}, "
          f"m={args.m}, tree_init={tree_init})")
    print("=" * 100)
    hdr = (
        f"{'mc':>4} | "
        f"{'cons_v':>7} {'cons_f':>7} {'Δcons':>7} | "
        f"{'srch_v':>7} {'srch_f':>7} {'Δsrch':>7} | "
        f"{'dcomp_v':>11} {'dcomp_f':>11} {'Δ%':>6} | "
        f"{'qps_v':>5} {'qps_f':>5} {'qpsx':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        v, f = r["v"], r["f"]
        d_cons = f["cons_recall"] - v["cons_recall"]
        d_srch = f["search_recall"] - v["search_recall"]
        d_dc = 100.0 * (v["dist_comps"] - f["dist_comps"]) / max(v["dist_comps"], 1)
        qpsx = f["qps"] / max(v["qps"], 1)
        print(
            f"{r['mc']:>4} | "
            f"{v['cons_recall']:>7.4f} {f['cons_recall']:>7.4f} {d_cons:>+7.4f} | "
            f"{v['search_recall']:>7.4f} {f['search_recall']:>7.4f} {d_srch:>+7.4f} | "
            f"{v['dist_comps']:>11,} {f['dist_comps']:>11,} {d_dc:>+5.1f}% | "
            f"{v['qps']:>5.0f} {f['qps']:>5.0f} {qpsx:>4.2f}x"
        )


if __name__ == "__main__":
    main()
