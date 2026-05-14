#!/usr/bin/env python3
"""Filter vs ef sweep: search-recall behavior at varying query budgets.

For each filter configuration (vanilla + several p_tau values), build the
graph once, then query it at multiple epsilon values (PyNNDescent's
search-budget knob; analogous to ef in HNSW). This characterizes how
filter-induced construction-recall reductions affect search recall as
a function of query-time exploration budget.

Headline question this answers: does low construction recall hurt search
at low ef (myopic search) but get compensated at high ef (patient search)?
Or is search recall preserved across the entire ef range?

Single JIT warmup + per-graph search warmup so timings reflect compute,
not compilation.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gist1m",
                    help="dataset name from bench_knng.DATASETS")
    ap.add_argument("--mc", type=int, default=40,
                    help="max_candidates (matches C++ paper default of 40)")
    ap.add_argument("--p-taus", default="0.60,0.80,0.95,0.99",
                    help="comma-separated p_tau values for filter graphs")
    ap.add_argument("--epsilons", default="0.0,0.05,0.1,0.2,0.3,0.4,0.5",
                    help="comma-separated epsilon (search budget) values")
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--m", type=int, default=16, help="num projections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tree-init", action="store_true",
                    help="Disable RP-tree init (random init only)")
    args = ap.parse_args()

    p_tau_values = [float(x) for x in args.p_taus.split(",")]
    epsilon_values = [float(x) for x in args.epsilons.split(",")]
    tree_init = not args.no_tree_init

    cfg = DATASETS[args.dataset]
    print(f"Filter vs ef sweep on {args.dataset}")
    print(f"  Config:    mc={args.mc} m={args.m} k={args.n_neighbors} "
          f"seed={args.seed} tree_init={tree_init}")
    print(f"  Builds:    vanilla + filter p_tau in {p_tau_values}")
    print(f"  Queries:   epsilon in {epsilon_values}")

    print("\nLoading data + GT + queries...", flush=True)
    data = load_fvecs(cfg["base"], n_limit=cfg["n_subset"])
    gt = load_gt_bin(cfg["gt"])
    if not (cfg.get("queries") and cfg.get("search_gt")):
        raise SystemExit(
            f"Dataset '{args.dataset}' missing queries/search_gt config."
        )
    queries = load_fvecs(cfg["queries"])
    search_gt = get_or_compute_search_gt(
        cfg["search_gt"], queries, data, args.n_neighbors
    )
    print(f"  data={data.shape}  queries={queries.shape}  "
          f"search_gt={search_gt.shape}")

    # JIT warmup: exercise BUILD hot paths once
    print("\nWarming up Numba JIT (build paths)...", flush=True)
    t = time.perf_counter()
    rng = np.random.default_rng(42)
    warm = rng.standard_normal((200, 16)).astype(np.float32)
    _ = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                              random_state=0, tree_init=tree_init,
                              use_projection_filter=False)
    warm_idx = pynndescent.NNDescent(warm, n_neighbors=5, max_candidates=10,
                                     random_state=0, tree_init=tree_init,
                                     use_projection_filter=True,
                                     num_projections=args.m,
                                     filter_confidence=0.95)
    # Exercise SEARCH hot path: PyNNDescent lazy-compiles its search
    # function on first call, then caches.
    warm_q = rng.standard_normal((10, 16)).astype(np.float32)
    _ = warm_idx.query(warm_q, k=3, epsilon=0.1)
    print(f"  done ({time.perf_counter()-t:.1f}s)")

    # === Build all graphs and measure search at each epsilon ===
    configs = [("vanilla", False, 0.95)]
    for pt in p_tau_values:
        configs.append((f"filter pt={pt}", True, pt))

    results = []
    for name, use_filter, p_tau in configs:
        print(f"\n--- {name} ---", flush=True)
        t0 = time.perf_counter()
        idx = pynndescent.NNDescent(
            data,
            n_neighbors=args.n_neighbors,
            max_candidates=args.mc,
            random_state=args.seed,
            verbose=False,
            tree_init=tree_init,
            use_projection_filter=use_filter,
            num_projections=args.m,
            filter_confidence=p_tau,
        )
        build_t = time.perf_counter() - t0
        cons_recall = recall_at_k(idx.neighbor_graph[0], gt, args.n_neighbors)
        print(f"  build: {build_t:6.2f}s  cons_recall={cons_recall:.4f}  "
              f"dist_comps={idx.n_dist_comps:>13,}  "
              f"filter_skips={idx.n_filter_skips:>12,}")

        # Per-graph search warmup: first query compiles the search function
        # specifically for THIS graph's structure. Doing it once with a tiny
        # batch lets subsequent measurements reflect real query cost.
        _ = idx.query(queries[:10], k=args.n_neighbors, epsilon=0.1)

        eps_results = []
        for eps in epsilon_values:
            t1 = time.perf_counter()
            pred, _ = idx.query(queries, k=args.n_neighbors, epsilon=eps)
            qtime = time.perf_counter() - t1
            srec = recall_at_k(pred, search_gt, args.n_neighbors)
            qps = queries.shape[0] / qtime
            eps_results.append({
                "eps": eps, "search_recall": srec, "qps": qps, "qtime": qtime,
            })
            print(f"    eps={eps:.2f}: srec={srec:.4f}  "
                  f"qps={qps:>6.1f}  qtime={qtime:5.2f}s")

        results.append({
            "name": name,
            "build_t": build_t,
            "cons_recall": cons_recall,
            "dist_comps": idx.n_dist_comps,
            "filter_skips": idx.n_filter_skips,
            "eps_results": eps_results,
        })

    # === SUMMARY: 2D tables (config x epsilon) ===
    print("\n" + "=" * 110)
    print(f"SUMMARY  (dataset={args.dataset}, mc={args.mc}, m={args.m}, "
          f"k={args.n_neighbors}, tree_init={tree_init})")
    print("=" * 110)

    eps_header = "  ".join(f"ε={e:>4.2f}" for e in epsilon_values)

    print(f"\nSearch recall@{args.n_neighbors}:")
    print(f"{'config':<18} {'cons':>6} {'build':>7}   {eps_header}")
    print("-" * (18 + 7 + 9 + 9 * len(epsilon_values)))
    for r in results:
        eps_str = "  ".join(f"{er['search_recall']:>6.4f}"
                            for er in r["eps_results"])
        print(f"{r['name']:<18} {r['cons_recall']:>6.4f} "
              f"{r['build_t']:>6.1f}s   {eps_str}")

    print(f"\nQPS (queries/second):")
    print(f"{'config':<18} {'cons':>6} {'build':>7}   {eps_header}")
    print("-" * (18 + 7 + 9 + 9 * len(epsilon_values)))
    for r in results:
        eps_str = "  ".join(f"{int(er['qps']):>6}"
                            for er in r["eps_results"])
        print(f"{r['name']:<18} {r['cons_recall']:>6.4f} "
              f"{r['build_t']:>6.1f}s   {eps_str}")

    # === CSV-friendly dump (long format, easy to plot) ===
    print(f"\n--- CSV (long format, for plotting) ---")
    print("config,cons_recall,build_t,dist_comps,filter_skips,epsilon,"
          "search_recall,qps,qtime")
    for r in results:
        for er in r["eps_results"]:
            print(f"{r['name']},{r['cons_recall']:.4f},{r['build_t']:.2f},"
                  f"{r['dist_comps']},{r['filter_skips']},"
                  f"{er['eps']:.2f},{er['search_recall']:.4f},"
                  f"{er['qps']:.1f},{er['qtime']:.3f}")


if __name__ == "__main__":
    main()
