"""#1 sanity check: pairwise error-set Jaccard across 7 actions.

Per setup, build error sets E_a = {i : action a is wrong on example i} for
a in {gB, gZ, gR, S0..S3}. Report J(E_a, E_b) = |inter|/|union| as a 7x7
matrix and compare:
  - mean J among samples (S_i, S_j)
  - mean J between samples and det grafts (S_i, gZ/gR)
  - mean J among det grafts (gZ, gR)

If det grafts are MORE disjoint from samples than samples are from each
other, the diversity story holds.
"""
import numpy as np

from _fig_common import SETUPS, load_rows

ACTIONS = ["gB", "gZ", "gR", "S0", "S1", "S2", "S3"]
DET = ["gZ", "gR"]
SAMP = ["S0", "S1", "S2", "S3"]


def err_set(rows, key):
    return {r["id"] for r in rows if r[key] != r["gold"]}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    print(f"{'Setup':<16} {'mean J(S,S)':>11} {'mean J(S,det)':>14} "
          f"{'J(gZ,gR)':>10} {'J(S,det)/J(S,S)':>17}")
    for name, cfg in SETUPS.items():
        rows = load_rows(cfg)
        E = {a: err_set(rows, a) for a in ACTIONS}

        # Pairwise within samples
        ss = [jaccard(E[i], E[j]) for i in SAMP for j in SAMP if i < j]
        # Sample x det
        sd = [jaccard(E[i], E[d]) for i in SAMP for d in DET]
        # Det x det
        dd = jaccard(E["gZ"], E["gR"])

        m_ss = np.mean(ss)
        m_sd = np.mean(sd)
        ratio = m_sd / m_ss if m_ss > 0 else float("nan")
        print(f"{name:<16} {m_ss:>11.4f} {m_sd:>14.4f} "
              f"{dd:>10.4f} {ratio:>17.3f}")

    # Detailed 7x7 for first setup
    print()
    name, cfg = next(iter(SETUPS.items()))
    rows = load_rows(cfg)
    E = {a: err_set(rows, a) for a in ACTIONS}
    print(f"Detail: {name} (J matrix, |E_a| in diag)")
    print(" " * 6 + " ".join(f"{a:>6}" for a in ACTIONS))
    for a in ACTIONS:
        row = []
        for b in ACTIONS:
            if a == b:
                row.append(f"{len(E[a]):>6d}")
            else:
                row.append(f"{jaccard(E[a], E[b]):>6.3f}")
        print(f"{a:>6} " + " ".join(row))


if __name__ == "__main__":
    main()
