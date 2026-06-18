"""Figure 4: per-layer hidden-state divergence at graft position.

Inputs: attention_analysis/qwen3b_gsm8k_{random,zero}_last_L26/example_*/hidden_trajectory.npz
Outputs: tex/figures/fig4_hidden_divergence.pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os

from _fig_common import FIG_DIR, set_paper_style

set_paper_style()

ATTN_ROOT = Path(os.environ.get("ATTENTION_ANALYSIS_DIR") or "attention_analysis")
GRAFT_LAYER = 26


def load_run(name, drop_outliers=False):
    run = ATTN_ROOT / name
    cos_all, l2_all = [], []
    for d in sorted(run.glob("example_*")):
        ht = np.load(d / "hidden_trajectory.npz")
        cos_all.append(ht["cos_sim"])
        l2_all.append(ht["l2_dist"])
    cos = np.stack(cos_all)
    l2 = np.stack(l2_all)
    if drop_outliers:
        med = np.median(l2[:, GRAFT_LAYER:], axis=0)
        bad = np.any(l2[:, GRAFT_LAYER:] > 3 * med[None, :], axis=1)
        cos, l2 = cos[~bad], l2[~bad]
    return cos, l2


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cos_R, l2_R = load_run("qwen3b_gsm8k_random_last_L26_n100", drop_outliers=True)
    cos_Z, l2_Z = load_run("qwen3b_gsm8k_zero_last_L26_n100", drop_outliers=True)

    n_layers = cos_R.shape[1]
    layers = np.arange(n_layers)
    keep = layers >= GRAFT_LAYER  # cos=1, L2=0 below graft layer; uninformative

    fig, (ax_l2, ax_cos) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True)

    # ── L_2 panel
    for label, l2, color, marker in [
        (r"$g_R$ (Random)", l2_R, "C0", "o"),
        (r"$g_Z$ (Zero)",   l2_Z, "C3", "s"),
    ]:
        mu  = l2[:, keep].mean(0)
        std = l2[:, keep].std(0)
        ax_l2.plot(layers[keep], mu, marker=marker, color=color, lw=1.5, ms=4, label=label)
        ax_l2.fill_between(layers[keep], mu - std, mu + std, color=color, alpha=0.18, lw=0)

    ax_l2.axvline(GRAFT_LAYER, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_l2.set_xlabel("Layer $\\ell$", fontsize=9)
    ax_l2.set_ylabel(r"$L_2$ Distance", fontsize=9)
    ax_l2.spines["top"].set_visible(False)
    ax_l2.spines["right"].set_visible(False)
    ax_l2.tick_params(labelsize=8)
    ax_l2.legend(fontsize=8, frameon=False, loc="upper left")
    ax_l2.grid(axis="y", alpha=0.25)

    # ── cosine panel
    for label, cos, color, marker in [
        (r"$g_R$ (Random)", cos_R, "C0", "o"),
        (r"$g_Z$ (Zero)",   cos_Z, "C3", "s"),
    ]:
        mu  = cos[:, keep].mean(0)
        std = cos[:, keep].std(0)
        ax_cos.plot(layers[keep], mu, marker=marker, color=color, lw=1.5, ms=4, label=label)
        ax_cos.fill_between(layers[keep], mu - std, mu + std, color=color, alpha=0.18, lw=0)

    ax_cos.axhline(0, color="gray", lw=0.6, alpha=0.6)
    ax_cos.axvline(GRAFT_LAYER, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_cos.set_xlabel("Layer $\\ell$", fontsize=9)
    ax_cos.set_ylabel("Cosine Similarity", fontsize=9)
    ax_cos.set_ylim(-0.1, 1.0)
    ax_cos.spines["top"].set_visible(False)
    ax_cos.spines["right"].set_visible(False)
    ax_cos.tick_params(labelsize=8)
    ax_cos.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "fig4_hidden_divergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[fig4] wrote {out}")


if __name__ == "__main__":
    main()
