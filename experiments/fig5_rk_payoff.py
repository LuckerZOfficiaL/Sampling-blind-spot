"""Figure 5: deterministic recovery R_k on the pass@6=0 slice.

For each setup, grouped bars show R_1, R_3, R_6, R_8 as fractions of
the pass@6=0 stratum (the examples where no sampling seed in
{42,...,47} reaches gold). Numbers come from /tmp/compute_R6.py and
match Table~\\ref{tab:Rk} in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _fig_common import FIG_DIR, set_paper_style

set_paper_style()

SETUPS = ["Qwen-3B / GSM8K", "Qwen-3B / MATH", "Qwen-3B / MMLU-Pro",
          "Llama-3B / GSM8K", "Llama-3B / MATH", "Llama-3B / MMLU-Pro",
          "Llama-8B / GSM8K", "Llama-8B / MATH", "Llama-8B / MMLU-Pro",
          "Nemo-12B / GSM8K", "Nemo-12B / MATH", "Nemo-12B / MMLU-Pro"]
PASS_AT_6 = {"Qwen-3B / GSM8K": 58, "Qwen-3B / MATH": 293, "Qwen-3B / MMLU-Pro": 323,
             "Llama-3B / GSM8K": 83, "Llama-3B / MATH": 287, "Llama-3B / MMLU-Pro": 336,
             "Llama-8B / GSM8K": 51, "Llama-8B / MATH": 291, "Llama-8B / MMLU-Pro": 252,
             "Nemo-12B / GSM8K": 54, "Nemo-12B / MATH": 435, "Nemo-12B / MMLU-Pro": 322}
R_COUNTS = {
    "Qwen-3B / GSM8K":     {1: 0,  3: 5,  6: 8,  8: 10},
    "Qwen-3B / MATH":      {1: 10, 3: 23, 6: 39, 8: 48},
    "Qwen-3B / MMLU-Pro":  {1: 14, 3: 43, 6: 81, 8: 100},
    "Llama-3B / GSM8K":    {1: 2,  3: 12, 6: 19, 8: 21},
    "Llama-3B / MATH":     {1: 11, 3: 24, 6: 42, 8: 50},
    "Llama-3B / MMLU-Pro": {1: 30, 3: 66, 6: 96, 8: 114},
    "Llama-8B / GSM8K":    {1: 3,  3: 7,  6: 10, 8: 11},
    "Llama-8B / MATH":     {1: 9,  3: 22, 6: 30, 8: 36},
    "Llama-8B / MMLU-Pro": {1: 19, 3: 46, 6: 72, 8: 80},
    "Nemo-12B / GSM8K":    {1: 1,  3: 8,  6: 11, 8: 13},
    "Nemo-12B / MATH":     {1: 16, 3: 39, 6: 64, 8: 75},
    "Nemo-12B / MMLU-Pro": {1: 11, 3: 55, 6: 93, 8: 109},
}
K_VALUES = [1, 3, 6, 8]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 2.6))

    x = np.arange(len(SETUPS))
    w = 0.2
    palette = ["#cccccc", "#9ec5fe", "#e76f51", "#7a3f2a"]

    for i, k in enumerate(K_VALUES):
        vals = np.array([R_COUNTS[s][k] / PASS_AT_6[s] * 100 for s in SETUPS])
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w,
                      color=palette[i], edgecolor="white",
                      label=rf"$R_{{{k}}}$")
        for xi, v, raw in zip(x + (i - 1.5) * w, vals,
                              [R_COUNTS[s][k] for s in SETUPS]):
            ax.text(xi, v + 0.4, f"{raw}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(SETUPS, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(r"\% of pass@6$=$0 recovered", fontsize=8)
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.0,
              handletextpad=0.5, columnspacing=1.0)

    fig.tight_layout()
    out = FIG_DIR / "fig5_rk_payoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[fig5] wrote {out}")


if __name__ == "__main__":
    main()
