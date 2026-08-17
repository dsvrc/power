"""Is a PTDF (or equivalent injection->flow sensitivity) reachable on this env?

    python probe_ptdf.py

No training, no logging -- builds the env once, tries every route grid2op /
lightsim2grid offer, and reports which works plus the resulting zone-to-zone
coupling matrix.  Run before rebuilding the basis on PTDF.
"""
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import grid2op                                              # noqa: E402
from grid2op.Action import PlayableAction                    # noqa: E402
from utils import G2OP_ENV_DIR                               # noqa: E402

ENV = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023")


def main():
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        print("backend: LightSimBackend")
    except ImportError:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
        print("backend: PandaPowerBackend")

    env = grid2op.make(ENV, action_class=PlayableAction, backend=backend)
    obs = env.reset()
    print(f"grid: n_sub={env.n_sub} n_line={env.n_line} n_gen={env.n_gen} "
          f"n_load={env.n_load}")

    ptdf = None
    route = None

    # 1) lightsim2grid's own PTDF
    try:
        gridmodel = env.backend._grid
        ptdf = np.asarray(gridmodel.get_ptdf())
        route = "lightsim2grid gridmodel.get_ptdf()"
    except Exception as e:                                   # noqa: BLE001
        print(f"  get_ptdf() unavailable: {type(e).__name__}: {e}")

    # 2) backend-level accessor, name varies by version
    if ptdf is None:
        for attr in ("get_ptdf", "get_PTDF", "ptdf"):
            try:
                cand = getattr(env.backend, attr)
                ptdf = np.asarray(cand() if callable(cand) else cand)
                route = f"backend.{attr}"
                break
            except Exception:                                # noqa: BLE001
                continue

    # 3) fall back to a DC PTDF built from the susceptance matrix
    if ptdf is None:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            lor = env.line_or_to_subid
            lex = env.line_ex_to_subid
            nl, ns = env.n_line, env.n_sub
            b = np.ones(nl)                 # unit susceptance: topology-only PTDF
            A = sp.lil_matrix((nl, ns))
            for k in range(nl):
                A[k, lor[k]] = 1.0
                A[k, lex[k]] = -1.0
            A = A.tocsr()
            Bf = sp.diags(b) @ A
            Bbus = (A.T @ Bf).tocsc()
            slack = 0
            keep = [i for i in range(ns) if i != slack]
            Bred = Bbus[keep, :][:, keep]
            H = np.zeros((nl, ns))
            H[:, keep] = Bf[:, keep].toarray() @ np.linalg.inv(Bred.toarray())
            ptdf = H
            route = "DC PTDF from topology (unit susceptance fallback)"
        except Exception as e:                               # noqa: BLE001
            print(f"  DC fallback failed: {type(e).__name__}: {e}")

    if ptdf is None:
        print("\nNO PTDF ROUTE AVAILABLE -- report this and we use a different basis.")
        return

    print(f"\nPTDF via: {route}")
    print(f"  shape={ptdf.shape}  finite={np.isfinite(ptdf).all()}  "
          f"|max|={np.abs(ptdf).max():.4g}  mean|.|={np.abs(ptdf).mean():.4g}")

    # --- zone-to-zone coupling implied by that PTDF ----------------------
    from benchmarl.environments.G2OpPowerGrid.utils import ZONES_DICT
    zones = [f"Zone{i}" for i in range(11)]
    gen_bus = env.gen_to_subid
    W = np.zeros((11, 11))
    for a, zi in enumerate(zones):
        lines_i = np.asarray(ZONES_DICT[zi]["line_in_zone_idx"], dtype=int)
        if len(lines_i) == 0:
            continue
        for b, zj in enumerate(zones):
            gi = np.asarray(ZONES_DICT[zj]["gen_inside_idx"], dtype=int)
            gj = np.intersect1d(gi, np.where(env.gen_renewable)[0])
            if a == b or len(gj) == 0:
                continue
            buses = gen_bus[gj]
            buses = buses[buses < ptdf.shape[1]]
            if not len(buses):
                continue
            W[a, b] = float(np.mean(np.abs(ptdf[np.ix_(lines_i, buses)])))

    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print("\nzone-to-zone coupling W[i,j] = mean |PTDF| from j's curtailable gens")
    print("onto i's own lines  (row = agent i):")
    print(W)

    off = W[~np.eye(11, dtype=bool)]
    nz = off[off > 0]
    print(f"\noff-diagonal: nonzero={len(nz)}/{off.size}  "
          f"min={nz.min():.4g} max={nz.max():.4g}  "
          f"ratio max/min={nz.max()/max(nz.min(), 1e-12):.1f}")
    print(f"spread std/mean = {nz.std()/nz.mean():.4f}")
    print("\nA large max/min ratio is the point: the geometric halo/distant basis")
    print("gave every pair in a bucket the SAME weight, which is what flattened")
    print("fit_gain. If this ratio is >> 1, PTDF weighting is the fix.")


if __name__ == "__main__":
    main()
