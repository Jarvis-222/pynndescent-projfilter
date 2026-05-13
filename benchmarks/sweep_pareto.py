#!/usr/bin/env python3
"""Pareto comparison: vanilla (varying max_candidates) vs filter (varying
p_tau at fixed mc).

Produces two curves on (construction_cost, recall) axes. The headline
question this answers: does the projection filter dominate the
'just-lower-max_candidates' baseline?

Loops are separated so vanilla runs do NOT pay the (irrelevant) filter
overhead and we do not run redundant configurations.
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


def one_run(data, gt, n_neighbors, mc, seed, use_filter, m, p_tau,
            queries, search_gt, tree_init):
    """Build + query + recall, returns dict of metrics."""
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
    cons_recall = recall_at_k(idx.neighbor_graph[0], gt, n_neighbors)
    tq = time.perf_counter()
    pred_q, _ = idx.query(queries, k=n_neighbors)
    query_t = time.perf_counter() - tq
    search_recall = recall_at_k(pred_q, search_gt, n_neighbors)
    qps = queries.shape[0] / query_t
    return {
        "build_t":       build_t,
        "cons_recall":   cons_recall,
        "search_recall": search_recall,
        "qps":           qps,
        "dist_comps":    idx.n_dist_comps,
        "filter_skips":  idx.n_filter_skips,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gist1m",
                    help="dataset name from bench_knng.DATASETS")
    ap.add_argument("--mcs", default="10,20,30,40,50,60",
                    help="comma-separated mc values for vanilla sweep")
    ap.add_argument("--filter-mc", type=int, default=60,
                    help="fixed mc value for filter sweep (default: 60)")
    ap.add_argument("--p-taus", default="0.30,0.50,0.70,0.80,0.95,0.99",
                    help="comma-separated p_tau values for filter sweep")
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tree-init", action="store_true",
                    help="Disable RP-tree init (random init only)")
    args = ap.parse_args()

    mc_values = [int(x) for x in args.mcs.split(",")]
    p_tau_values = [float(x) for x in args.p_taus.split(",")]
    tree_init = not args.no_tree_init

    cfg = DATASETS[args.dataset]
    print(f"Pareto sweep on {args.dataset}")
    print(f"  Vanilla:  mc in {mc_values}")
    print(f"  Filter:   mc={args.filter_mc}, p_tau in {p_tau_values}, m={args.m}")
    print(f"  Common:   tree_init={tree_init}, seed={args.seed}, "
          f"n_neighbors={args.n_neighbors}")

    print("\nLoading data + GT...", flush=True)
    data = load_fvecs(cfg["base"], n_limit=cfg["n_subset"])
    gt = load_gt_bin(cfg["gt"])
    if not (cfg.get("queries") and cfg.get("search_gt")):
        raise SystemExit(
            f"Dataset '{args.dataset}' missing queries/search_gt config. "
            f"Add them to DATASETS in bench_knng.py first."
        )
    queries = load_fvecs(cfg["queries"])
    search_gt = get_or_compute_search_gt(
        cfg["search_gt"], queries, data, args.n_neighbors
    )
    print(f"  data={data.shape}  gt={gt.shape}  "
          f"queries={queries.shape}  search_gt={search_gt.shape}")

    # JIT warmup -- both code paths once each
    print("\nWarming up Numba JIT...", flush=True)
    t = time.perf_counter()
    rng = np.random.default_rng(42)
    warm = rng.standard_normal((200, 16)).astype(np.float32)
    _ = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                              random_state=0, tree_init=tree_init,
                              use_projection_filter=False)
    _ = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                              random_state=0, tree_init=tree_init,
                              use_projection_filter=True,
                              num_projections=args.m,
                              filter_confidence=0.95)
    print(f"  done ({time.perf_counter()-t:.1f}s)")

    # === VANILLA SWEEP (varying mc, no filter) ===
    print("\n" + "=" * 80)
    print("VANILLA SWEEP  (max_candidates varying; no projection filter)")
    print("=" * 80)
    vanilla_rows = []
    for mc in mc_values:
        print(f"  mc={mc:>3} ...", flush=True, end="")
        r = one_run(
            data=data, gt=gt, n_neighbors=args.n_neighbors,
            mc=mc, seed=args.seed, use_filter=False,
            m=args.m, p_tau=0.95,    # value irrelevant when use_filter=False
            queries=queries, search_gt=search_gt, tree_init=tree_init,
        )
        print(f"  build={r['build_t']:6.2f}s  "
              f"cons={r['cons_recall']:.4f}  "
              f"search={r['search_recall']:.4f}  "
              f"qps={r['qps']:5.0f}  "
              f"dist_comps={r['dist_comps']:>13,}")
        vanilla_rows.append({"mc": mc, **r})

    # === FILTER SWEEP (fixed mc, varying p_tau) ===
    print("\n" + "=" * 80)
    print(f"FILTER SWEEP  (mc={args.filter_mc} fixed; p_tau varying)")
    print("=" * 80)
    filter_rows = []
    for pt in p_tau_values:
        print(f"  p_tau={pt:.2f} ...", flush=True, end="")
        r = one_run(
            data=data, gt=gt, n_neighbors=args.n_neighbors,
            mc=args.filter_mc, seed=args.seed, use_filter=True,
            m=args.m, p_tau=pt,
            queries=queries, search_gt=search_gt, tree_init=tree_init,
        )
        print(f"  build={r['build_t']:6.2f}s  "
              f"cons={r['cons_recall']:.4f}  "
              f"search={r['search_recall']:.4f}  "
              f"qps={r['qps']:5.0f}  "
              f"dist_comps={r['dist_comps']:>13,}  "
              f"skips={r['filter_skips']:>12,}")
        filter_rows.append({"p_tau": pt, **r})

    # === COMBINED OUTPUT (table-shaped, copy-paste into a plot) ===
    print("\n" + "=" * 90)
    print(f"VANILLA CURVE   (dataset={args.dataset}, tree_init={tree_init})")
    print("=" * 90)
    hdr = (f"{'mc':>4} | {'build':>8} | {'dist_comps':>13} | "
           f"{'cons_rec':>9} | {'srch_rec':>9} | {'qps':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in vanilla_rows:
        print(f"{r['mc']:>4} | "
              f"{r['build_t']:>7.2f}s | "
              f"{r['dist_comps']:>13,} | "
              f"{r['cons_recall']:>9.4f} | "
              f"{r['search_recall']:>9.4f} | "
              f"{r['qps']:>5.0f}")

    print("\n" + "=" * 90)
    print(f"FILTER CURVE    (mc={args.filter_mc}, m={args.m}, varying p_tau)")
    print("=" * 90)
    hdr = (f"{'p_tau':>6} | {'build':>8} | {'dist_comps':>13} | "
           f"{'cons_rec':>9} | {'srch_rec':>9} | {'qps':>5} | {'skip%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in filter_rows:
        total = r["dist_comps"] + r["filter_skips"]
        skip_pct = 100.0 * r["filter_skips"] / max(total, 1)
        print(f"{r['p_tau']:>6.2f} | "
              f"{r['build_t']:>7.2f}s | "
              f"{r['dist_comps']:>13,} | "
              f"{r['cons_recall']:>9.4f} | "
              f"{r['search_recall']:>9.4f} | "
              f"{r['qps']:>5.0f} | "
              f"{skip_pct:>5.1f}%")

    # Sanity: vanilla at mc=filter_mc should match filter run with use_filter=False.
    # Print a quick "noise floor" check from vanilla rows if filter_mc was swept.
    matching = [r for r in vanilla_rows if r["mc"] == args.filter_mc]
    if matching:
        print(f"\nVanilla baseline at mc={args.filter_mc}: "
              f"build={matching[0]['build_t']:.2f}s  "
              f"cons={matching[0]['cons_recall']:.4f}  "
              f"search={matching[0]['search_recall']:.4f}")
        print("(This is the apples-to-apples vs the filter curve above.)")


if __name__ == "__main__":
    main()
