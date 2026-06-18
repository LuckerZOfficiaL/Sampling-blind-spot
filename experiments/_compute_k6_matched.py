"""Budget-matched k=6 comparison: pure 6S MV vs (3S + 3det) MV.

For each setup:
  - pure 6S MV over seeds 42-47 (single number)
  - 3S + 3det MV averaged over all C(6,3)=20 subsets of the 6 sample seeds,
    plus min/max across subsets

Both use the paper's 50-tie-break-seed averaging via mv_acc().
"""
from itertools import combinations
import numpy as np

from _fig_common import SETUPS, mv_acc
from fig1_budget_curve import load_rows6

DET_KEYS = ["gB", "gZ", "gR"]
S6_KEYS = [f"S{i}" for i in range(6)]


def main():
    print(f"{'Setup':<16} {'pure 6S':>9} {'3S+3det mean':>13} "
          f"{'[min, max]':>16} {'Δ mean':>8} {'#wins/20':>9}")
    for name, cfg in SETUPS.items():
        rows = load_rows6(name, cfg)
        pure6 = mv_acc(rows, S6_KEYS)

        accs = []
        for sub in combinations(S6_KEYS, 3):
            accs.append(mv_acc(rows, list(sub) + DET_KEYS))
        accs = np.array(accs)
        mean_mix = accs.mean()
        wins = int((accs > pure6).sum())
        print(f"{name:<16} {pure6:>9.4f} {mean_mix:>13.4f} "
              f"  [{accs.min():.4f}, {accs.max():.4f}] "
              f"{mean_mix - pure6:>+8.4f} {wins:>5}/20")


if __name__ == "__main__":
    main()
