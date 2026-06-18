"""Shared loading and majority-vote utilities for figure scripts."""
import json
import os
import re
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = Path(os.environ.get("FIG_DIR") or (REPO / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_paper_style():
    """Render figures with LaTeX so fonts match the paper body."""
    import matplotlib as mpl
    # Workaround: the system luatex is broken (zlib mismatch) and silently
    # returns an empty path from kpsewhich, which matplotlib then trusts.
    # Force matplotlib to use the cmdline kpsewhich instead.
    import matplotlib.dviread as _dvi
    class _NoLuatex(Exception):
        pass
    _dvi._LuatexKpsewhich = lambda: (_ for _ in ()).throw(FileNotFoundError("disabled"))
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })


def normalize_gsm8k(ans):
    if ans is None:
        return None
    s = str(ans).strip().replace(",", "").replace("$", "").strip()
    s = re.sub(r"[^\d\-\.eE]", "", s)
    if s in ("", "-", ".", "-."):
        return None
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.10g}"
    except ValueError:
        return None


def normalize_math(ans):
    if ans is None:
        return None
    s = re.sub(r"\s+", "", str(ans).strip()).rstrip(".")
    return s if s else None


def _seeds(s_list):
    return [REPO / f"results_sampling/{s_list['bench']}_baseline_{s_list['m']}_sampling_seed{s}_1000/baseline_{s_list['mslug']}/predictions.json"
            for s in (42, 43, 44, 45)]


SETUPS = {
    "Qwen / GSM8K": {
        "norm": normalize_gsm8k,
        "S": _seeds({"bench": "gsm8k", "m": "Qwen2.5-3B-Instruct", "mslug": "Qwen2_5_3B_Instruct"}),
        "gB": REPO / "results_conf/gsm8k_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json",
        "gZ": REPO / "results_conf/gsm8k_ac_zero_last_L26_Qwen2.5-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_conf/gsm8k_ac_random_last_L26_Qwen2.5-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
    },
    "Qwen / MATH": {
        "norm": normalize_math,
        "S": _seeds({"bench": "math", "m": "Qwen2.5-3B-Instruct", "mslug": "Qwen2_5_3B_Instruct"}),
        "gB": REPO / "results_conf/math_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json",
        "gZ": REPO / "results_conf/math_ac_zero_last_L26_Qwen2.5-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_conf/math_ac_random_last_L26_Qwen2.5-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
    },
    "Llama / GSM8K": {
        "norm": normalize_gsm8k,
        "S": _seeds({"bench": "gsm8k", "m": "Llama-3.2-3B-Instruct", "mslug": "Llama_3_2_3B_Instruct"}),
        "gB": REPO / "results_conf/gsm8k_baseline_Llama-3.2-3B-Instruct_greedy_1000/baseline_Llama_3_2_3B_Instruct/predictions.json",
        "gZ": REPO / "results_conf/gsm8k_ac_zero_last_L26_Llama-3.2-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_conf/gsm8k_ac_random_last_L26_Llama-3.2-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
    },
    "Llama / MATH": {
        "norm": normalize_math,
        "S": _seeds({"bench": "math", "m": "Llama-3.2-3B-Instruct", "mslug": "Llama_3_2_3B_Instruct"}),
        "gB": REPO / "results_conf/math_baseline_Llama-3.2-3B-Instruct_greedy_1000/baseline_Llama_3_2_3B_Instruct/predictions.json",
        "gZ": REPO / "results_conf/math_ac_zero_last_L26_Llama-3.2-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_conf/math_ac_random_last_L26_Llama-3.2-3B-Instruct_greedy_1000/ac_trained_projection/predictions.json",
    },
    "Llama-8B / GSM8K": {
        "norm": normalize_gsm8k,
        "S": [REPO / f"results_sampling/gsm8k_baseline_Llama-3.1-8B-Instruct_sampling_seed{s}_1000/gsm8k_baseline_Llama-3.1-8B-Instruct_sampling_seed{s}_1000/baseline_Llama_3_1_8B_Instruct/predictions.json" for s in (42, 43, 44, 45, 46, 47)],
        "gB": REPO / "results_greedy_decoding/gsm8k_baselines_Llama-3.1-8B-Instruct_last_L26_Llama-3.1-8B-Instruct_1000/baseline_Llama_3_1_8B_Instruct/predictions.json",
        "gZ": REPO / "results_greedy_decoding/gsm8k_ac_zero_last_L26_Llama-3.1-8B-Instruct_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_greedy_decoding/gsm8k_ac_random_last_L26_Llama-3.1-8B-Instruct_1000/ac_trained_projection/predictions.json",
    },
    "Llama-8B / MATH": {
        "norm": normalize_math,
        "S": [REPO / f"results_sampling/math_baseline_Llama-3.1-8B-Instruct_sampling_seed{s}_1000/math_baseline_Llama-3.1-8B-Instruct_sampling_seed{s}_1000/baseline_Llama_3_1_8B_Instruct/predictions.json" for s in (42, 43, 44, 45, 46, 47)],
        "gB": REPO / "results_greedy_decoding/math_baseline_llama8B_1000/baseline_Llama_3_1_8B_Instruct/predictions.json",
        "gZ": REPO / "results_greedy_decoding/math_ac_zero_llama8B_1000/ac_trained_projection/predictions.json",
        "gR": REPO / "results_greedy_decoding/math_ac_random_llama8B_1000/ac_trained_projection/predictions.json",
    },
}


def load_rows(cfg):
    norm = cfg["norm"]

    def gx(p):
        return norm(p.get("extracted_answer"))

    s_runs = [{p["id"]: p for p in json.load(open(str(pp)))} for pp in cfg["S"]]
    gB = {p["id"]: p for p in json.load(open(str(cfg["gB"])))}
    gZ = {p["id"]: p for p in json.load(open(str(cfg["gZ"])))}
    gR = {p["id"]: p for p in json.load(open(str(cfg["gR"])))}
    common = sorted(set.intersection(*(set(r.keys()) for r in s_runs + [gB, gZ, gR])))
    rows = []
    for ex_id in common:
        exp = gB[ex_id].get("expected")
        if exp is None:
            continue
        gold = norm(exp)
        if gold is None:
            continue
        rows.append({
            "id": ex_id, "gold": gold,
            "gB": gx(gB[ex_id]), "gZ": gx(gZ[ex_id]), "gR": gx(gR[ex_id]),
            "S0": gx(s_runs[0][ex_id]), "S1": gx(s_runs[1][ex_id]),
            "S2": gx(s_runs[2][ex_id]), "S3": gx(s_runs[3][ex_id]),
        })
    return rows


def mv_correct_one(answers, gold, rng):
    valid = [a for a in answers if a is not None]
    if not valid:
        return False
    cnt = Counter(valid)
    top = cnt.most_common()
    max_count = top[0][1]
    winners = [a for a, c in top if c == max_count]
    chosen = rng.choice(winners) if len(winners) > 1 else winners[0]
    return chosen == gold


def mv_acc(rows, keys, n_seeds=50):
    accs = np.zeros(n_seeds)
    for s in range(n_seeds):
        rng = random.Random(s)
        c = sum(1 for r in rows if mv_correct_one([r[k] for k in keys], r["gold"], rng))
        accs[s] = c / len(rows)
    return float(accs.mean())


def mv_correct_per_example_majority(rows, keys, n_seeds=50, threshold=0.5):
    """Per-example: True if MV is correct in >= threshold * n_seeds tie-break runs."""
    out = []
    for r in rows:
        c = 0
        for s in range(n_seeds):
            rng = random.Random(s)
            if mv_correct_one([r[k] for k in keys], r["gold"], rng):
                c += 1
        out.append(c >= threshold * n_seeds)
    return out


def avg_subset_acc(rows, keys, k):
    """Mean MV accuracy averaged over all C(|keys|, k) k-subsets."""
    subs = list(combinations(keys, k))
    return float(np.mean([mv_acc(rows, list(sub)) for sub in subs]))
