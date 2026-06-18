"""H1: disagreement-routing with greedy fallback.

Policy: if k samples agree (unanimity or >=3/4 plurality), return that
answer; else fall back to greedy g_B. Compare against pure MV_{4S}.
"""
from collections import Counter

from _fig_common import SETUPS, load_rows, mv_acc

S_KEYS = ["S0", "S1", "S2", "S3"]


def routed_acc(rows, min_votes):
    """Return (accuracy, fraction_routed_to_gB)."""
    correct = 0
    routed = 0
    for r in rows:
        votes = [r[k] for k in S_KEYS if r[k] is not None]
        chosen = None
        if votes:
            top_ans, top_cnt = Counter(votes).most_common(1)[0]
            if top_cnt >= min_votes:
                chosen = top_ans
        if chosen is None:
            chosen = r["gB"]
            routed += 1
        if chosen == r["gold"]:
            correct += 1
    n = len(rows)
    return correct / n, routed / n


def gB_acc(rows):
    return sum(1 for r in rows if r["gB"] == r["gold"]) / len(rows)


def main():
    print(f"{'Setup':<16} {'gB':>7} {'MV4S':>7} "
          f"{'unan':>7} {'Δ_un':>7} {'%gB_un':>7} "
          f"{'>=3/4':>7} {'Δ_3':>7} {'%gB_3':>7}")
    for name, cfg in SETUPS.items():
        rows = load_rows(cfg)
        a_gb = gB_acc(rows)
        a_mv = mv_acc(rows, S_KEYS)
        a_un, f_un = routed_acc(rows, 4)
        a_3, f_3 = routed_acc(rows, 3)
        print(f"{name:<16} {a_gb:>7.4f} {a_mv:>7.4f} "
              f"{a_un:>7.4f} {a_un - a_mv:>+7.4f} {f_un:>7.2%} "
              f"{a_3:>7.4f} {a_3 - a_mv:>+7.4f} {f_3:>7.2%}")


if __name__ == "__main__":
    main()
