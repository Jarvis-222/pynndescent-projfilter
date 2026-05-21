#!/usr/bin/env python3
"""Grid sweep: search recall at multiple (p_tau, epsilon, k) configurations.

For each filter config (vanilla + several p_tau values), build the graph
once, run prepare() once for the search graph, then query at multiple
epsilons. From each query result (taken at k=max(eval_ks)) we compute
recall at every k in eval_ks.

External 3-phase wall-clock timing:
  t_build   = time around NNDescent(...) ctor   (init + iter combined)
  t_prepare = time around idx.prepare()         (search-graph diversify)
  t_query   = time around idx.query(...)        (one per epsilon)

Dist-comp savings decomposed into:
  direct_save     = filter_skips
  trajectory_save = vanilla_dist_comps - filter_dist_comps - filter_skips
                    (pairs the modified graph never even generated)

Skips construction-recall measurement entirely (no construction GT load).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pynndescent

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_knng import (
    DATASETS,
    DATA_DIR,
    load_fvecs,
    get_or_compute_search_gt,
    recall_at_k,
)

# --- Sweep-only dataset extensions ---
# These datasets don't have pre-computed construction ground truth.
# sweep_recall_grid never reads `gt` (it only measures search recall),
# so leaving it None is safe. search_gt is computed on the fly via
# get_or_compute_search_gt (FAISS-style brute force + caching) on first
# run, then reloaded from the .ivecs cache for subsequent runs.
DATASETS = dict(DATASETS)  # copy so we don't mutate bench_knng's dict
DATASETS["gist500k"] = {
    "base":      DATA_DIR / "gist" / "gist_base.fvecs",
    "gt":        None,
    "n_subset":  500_000,
    "queries":   DATA_DIR / "gist" / "gist_query.fvecs",
    "search_gt": DATA_DIR / "gist500k_search_gt_for_std_query.ivecs",
}


class _Tee:
    """Stdout wrapper: writes to multiple streams (terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _make_results_path(args, tree_init):
    """Pick a descriptive filename under <repo>/results/ for this run."""
    init_tag = "tree" if tree_init else "rand"
    ts = time.strftime("%Y%m%dT%H%M")
    fname = (
        f"sweep_recall_grid_{args.dataset}"
        f"_mc{args.mc}_m{args.m}_{init_tag}_{ts}.txt"
    )
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir / fname


def build_and_prepare(data, n_neighbors, mc, seed, use_filter, m, p_tau,
                       tree_init):
    """Build + eagerly prepare. Returns (idx, t_build, t_prepare)."""
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
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    idx.prepare()
    t_prepare = time.perf_counter() - t1
    return idx, t_build, t_prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gist1m")
    ap.add_argument("--n-neighbors", type=int, default=10,
                    help="graph k (constructed graph stores this many neighbors per vertex)")
    ap.add_argument("--mc", type=int, default=40,
                    help="max_candidates (matches C++ paper default of 40)")
    ap.add_argument(
        "--p-taus",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99",
        help="comma-separated filter confidences (vanilla added automatically); "
             "default sweeps 0.10..0.90 in 0.10 steps then 0.95, 0.99",
    )
    ap.add_argument("--epsilons", default="0.0,0.05,0.1,0.2,0.3,0.5",
                    help="comma-separated search-budget values for query() calls")
    ap.add_argument("--eval-ks", default="10,20,50,100",
                    help="comma-separated k values for recall@k evaluation")
    ap.add_argument("--m", type=int, default=16, help="num projections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tree-init", action="store_true",
                    help="Disable RP-tree init (random init only)")
    args = ap.parse_args()

    p_tau_values = [float(x) for x in args.p_taus.split(",")]
    epsilon_values = [float(x) for x in args.epsilons.split(",")]
    eval_ks = sorted(set(int(k) for k in args.eval_ks.split(",")))
    max_eval_k = max(eval_ks)
    tree_init = not args.no_tree_init

    # Tee stdout to a result file so the run is captured to disk while still
    # showing live progress in the terminal. File path is committable so the
    # results can be pushed back from uni and pulled on the laptop.
    log_path = _make_results_path(args, tree_init)
    _log_file = open(log_path, "w")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_file)
    print(f"# results being written to: {log_path}")

    cfg = DATASETS[args.dataset]
    print(f"Grid sweep on {args.dataset}")
    print(f"  graph k (n_neighbors)= {args.n_neighbors}, mc={args.mc}, "
          f"m={args.m}, tree_init={tree_init}, seed={args.seed}")
    print(f"  filter p_tau values:  {p_tau_values} (+ vanilla)")
    print(f"  query epsilons:       {epsilon_values}")
    print(f"  recall@k evaluation:  {eval_ks}  (query at k={max_eval_k})")

    print("\nLoading data + queries + search GT...", flush=True)
    data = load_fvecs(cfg["base"], n_limit=cfg["n_subset"])
    if not (cfg.get("queries") and cfg.get("search_gt")):
        raise SystemExit(
            f"Dataset '{args.dataset}' missing queries/search_gt config."
        )
    queries = load_fvecs(cfg["queries"])
    search_gt = get_or_compute_search_gt(
        cfg["search_gt"], queries, data, max_eval_k
    )
    if search_gt.shape[1] < max_eval_k:
        raise SystemExit(
            f"search_gt has only {search_gt.shape[1]} cols, need {max_eval_k}."
        )
    print(f"  data={data.shape}  queries={queries.shape}  "
          f"search_gt={search_gt.shape}")

    # JIT warmup: build (vanilla + filter) and one query, all on tiny data
    print("\nWarming up Numba JIT...", flush=True)
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
    warm_idx.prepare()
    _ = warm_idx.query(rng.standard_normal((10, 16)).astype(np.float32),
                       k=3, epsilon=0.1)
    print(f"  done ({time.perf_counter()-t:.1f}s)")

    # === Build phase: one build per config ===
    configs = [("vanilla", False, 0.95)]
    for pt in p_tau_values:
        configs.append((f"pt={pt}", True, pt))

    rows = []
    for name, use_filter, p_tau in configs:
        print(f"\n--- {name} ---", flush=True)
        idx, t_build, t_prep = build_and_prepare(
            data, args.n_neighbors, args.mc, args.seed,
            use_filter, args.m, p_tau, tree_init,
        )
        print(f"  t_build={t_build:6.2f}s  t_prepare={t_prep:5.2f}s  "
              f"dist_comps={idx.n_dist_comps:>13,}  "
              f"filter_skips={idx.n_filter_skips:>13,}")

        # Query at each epsilon, derive recall@k for every k in eval_ks
        per_eps = []
        for eps in epsilon_values:
            tq = time.perf_counter()
            pred, _ = idx.query(queries, k=max_eval_k, epsilon=eps)
            qtime = time.perf_counter() - tq
            qps = queries.shape[0] / qtime
            recalls = {k: recall_at_k(pred, search_gt, k) for k in eval_ks}
            per_eps.append({
                "eps": eps, "qtime": qtime, "qps": qps, "recalls": recalls,
            })
            recall_str = "  ".join(
                f"sr@{k}={recalls[k]:.4f}" for k in eval_ks
            )
            print(f"    eps={eps:.2f}: {recall_str}  qps={qps:>6.1f}  "
                  f"qtime={qtime:5.2f}s")

        rows.append({
            "name": name, "use_filter": use_filter, "p_tau": p_tau,
            "t_build": t_build, "t_prepare": t_prep,
            "t_phase_b": idx.t_phase_b,
            "t_phase_a": max(t_build - idx.t_phase_b, 0.0),
            "dist_comps": idx.n_dist_comps,
            "filter_skips": idx.n_filter_skips,
            "per_eps": per_eps,
        })

    # === Output tables ===
    print("\n" + "=" * 100)
    print(f"TIME BREAKDOWN  (dataset={args.dataset}, mc={args.mc}, m={args.m}, "
          f"tree_init={tree_init})")
    print("=" * 100)
    print(f"{'config':<12}  {'t_build':>10}  {'phaseA':>9}  {'phaseB':>9}  "
          f"{'t_prepare':>10}  {'t_total':>10}  {'dist_comps':>14}  {'skip%':>6}")
    print("-" * 100)
    vanilla_dc = rows[0]["dist_comps"]
    for r in rows:
        total = r["t_build"] + r["t_prepare"]
        total_attempted = max(r["dist_comps"] + r["filter_skips"], 1)
        skip_pct = 100.0 * r["filter_skips"] / total_attempted
        print(f"{r['name']:<12}  {r['t_build']:>9.2f}s  {r['t_phase_a']:>8.2f}s  "
              f"{r['t_phase_b']:>8.2f}s  {r['t_prepare']:>9.2f}s"
              f"  {total:>9.2f}s  {r['dist_comps']:>14,}  {skip_pct:>5.1f}%")

    # Dist-comp savings decomposition
    print("\n" + "=" * 100)
    print(f"DIST-COMP SAVINGS DECOMPOSITION  (vanilla baseline = {vanilla_dc:,})")
    print("=" * 100)
    print(f"{'config':<12}  {'direct':>14}  {'trajectory':>14}  "
          f"{'total':>14}  {'total%':>7}")
    print("-" * 70)
    for r in rows:
        if not r["use_filter"]:
            continue
        direct = r["filter_skips"]
        total = vanilla_dc - r["dist_comps"]
        traj = total - direct
        pct = 100.0 * total / max(vanilla_dc, 1)
        print(f"{r['name']:<12}  {direct:>14,}  {traj:>14,}  "
              f"{total:>14,}  {pct:>6.1f}%")

    # Recall tables (one per k)
    for k in eval_ks:
        print("\n" + "=" * 100)
        print(f"SEARCH RECALL @{k}  (rows: config, cols: epsilon)")
        print("=" * 100)
        eps_hdr = "  ".join(f"{'ε='+str(e):>7}" for e in epsilon_values)
        print(f"{'config':<12}  {eps_hdr}")
        print("-" * (12 + 2 + 9 * len(epsilon_values)))
        for r in rows:
            vals = "  ".join(f"{er['recalls'][k]:>7.4f}" for er in r["per_eps"])
            print(f"{r['name']:<12}  {vals}")

    # QPS table
    print("\n" + "=" * 100)
    print(f"QPS  (rows: config, cols: epsilon)")
    print("=" * 100)
    eps_hdr = "  ".join(f"{'ε='+str(e):>8}" for e in epsilon_values)
    print(f"{'config':<12}  {eps_hdr}")
    print("-" * (12 + 2 + 10 * len(epsilon_values)))
    for r in rows:
        vals = "  ".join(f"{int(er['qps']):>8}" for er in r["per_eps"])
        print(f"{r['name']:<12}  {vals}")

    # CSV long-format dump
    print(f"\n--- CSV (long format, one row per config x epsilon x k) ---")
    print("config,p_tau,t_build,t_phase_a,t_phase_b,t_prepare,"
          "dist_comps,filter_skips,epsilon,qps,qtime,k,search_recall")
    for r in rows:
        for er in r["per_eps"]:
            for k in eval_ks:
                print(f"{r['name']},{r['p_tau']:.2f},{r['t_build']:.3f},"
                      f"{r['t_phase_a']:.3f},{r['t_phase_b']:.3f},"
                      f"{r['t_prepare']:.3f},{r['dist_comps']},"
                      f"{r['filter_skips']},{er['eps']:.2f},"
                      f"{er['qps']:.2f},{er['qtime']:.4f},"
                      f"{k},{er['recalls'][k]:.4f}")

    # Restore stdout and close the log file
    sys.stdout = _orig_stdout
    _log_file.close()
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
