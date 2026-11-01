#!/usr/bin/env python3
"""Plot hasil training continuous_qlearning dari episode_log.csv.

Pakai (dari folder ini, di luar Webots):
    python plot_training.py                 # episode_log.csv -> training_plot.png
    python plot_training.py --best          # episode_log_best.csv -> training_plot_best.png
    python plot_training.py --window 30     # jendela rata-rata bergerak (bawaan 50)

Enam panel: reward, success rate + epsilon, langkah keputusan episode sukses,
komposisi hasil per blok episode, langkah di area bahaya, dan jumlah state.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COL = {"goal": "#2e9e4f", "collision": "#d64545", "timeout": "#e0a030", "stuck": "#8a5fc0"}


def moving(x, w):
    """Rata-rata bergerak; di awal memakai data yang ada (panjang keluaran = masukan)."""
    x = np.asarray(x, float)
    out = np.empty(len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load(path):
    # errors="ignore" dan penyaring baris: CSV bisa berisi sisa/padding kosong bila
    # ditimpa saat file masih dipegang proses Webots lain.
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get("episode") or "").strip().isdigit()
                and all(r.get(k) not in (None, "") for k in
                        ("epsilon", "decisions", "reward", "outcome", "hazard_steps", "states"))]
    if not rows:
        raise SystemExit(f"{path} kosong")
    return {
        "ep": np.array([int(r["episode"]) for r in rows]),
        "eps": np.array([float(r["epsilon"]) for r in rows]),
        "dec": np.array([int(r["decisions"]) for r in rows]),
        "rew": np.array([float(r["reward"]) for r in rows]),
        "out": np.array([r["outcome"] for r in rows]),
        "haz": np.array([int(r["hazard_steps"]) for r in rows]),
        "states": np.array([int(r["states"]) for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true", help="pakai episode_log_best.csv")
    ap.add_argument("--window", type=int, default=50)
    a = ap.parse_args()
    suffix = "_best" if a.best else ""
    src = HERE / f"episode_log{suffix}.csv"
    if not src.exists():
        raise SystemExit(f"tidak ada {src.name} (jalankan training dulu)")
    d = load(src)
    ep, W = d["ep"], a.window
    goal = d["out"] == "goal"

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"continuous_qlearning: {len(ep)} episode, {int(goal.sum())} sukses "
                 f"({goal.mean() * 100:.0f}%), {d['states'][-1]} state", fontsize=13)

    # 1 reward
    x = ax[0, 0]
    x.scatter(ep, d["rew"], s=6, c=[COL[o] for o in d["out"]], alpha=.5)
    x.plot(ep, moving(d["rew"], W), "k", lw=1.8, label=f"rata-rata {W}")
    x.set(title="Reward per episode", xlabel="episode", ylabel="reward")
    x.legend(loc="lower right")

    # 2 success rate + epsilon
    x = ax[0, 1]
    x.plot(ep, moving(goal.astype(float), W), color="#2e9e4f", lw=2, label=f"success rate ({W})")
    x.set(title="Success rate dan epsilon", xlabel="episode", ylabel="success rate", ylim=(-0.02, 1.02))
    x2 = x.twinx()
    x2.plot(ep, d["eps"], color="#4477cc", lw=1.3, ls="--", label="epsilon")
    x2.set_ylabel("epsilon", color="#4477cc")
    x2.set_ylim(0, 1.02)
    h1, l1 = x.get_legend_handles_labels()
    h2, l2 = x2.get_legend_handles_labels()
    x.legend(h1 + h2, l1 + l2, loc="center right")

    # 3 langkah keputusan episode sukses
    x = ax[0, 2]
    if goal.any():
        x.scatter(ep[goal], d["dec"][goal], s=8, c=COL["goal"], alpha=.6)
        x.plot(ep[goal], moving(d["dec"][goal], max(5, W // 5)), "k", lw=1.6, label="rata-rata")
        x.legend()
    else:
        x.text(.5, .5, "belum ada episode sukses", ha="center", va="center", transform=x.transAxes)
    x.set(title="Langkah keputusan (episode sukses)", xlabel="episode", ylabel="decisions")

    # 4 komposisi hasil per blok
    x = ax[1, 0]
    nb = max(1, len(ep) // W)
    edges = np.linspace(0, len(ep), nb + 1).astype(int)
    bottom = np.zeros(nb)
    centers = [(ep[edges[i]] + ep[edges[i + 1] - 1]) / 2 for i in range(nb)]
    width = (ep[-1] - ep[0] + 1) / nb * 0.9
    for name in ("goal", "collision", "timeout", "stuck"):
        frac = np.array([(d["out"][edges[i]:edges[i + 1]] == name).mean() for i in range(nb)])
        x.bar(centers, frac, width, bottom=bottom, color=COL[name], label=name)
        bottom += frac
    x.set(title=f"Komposisi hasil per blok (~{W} episode)", xlabel="episode", ylabel="proporsi", ylim=(0, 1))
    x.legend(loc="upper right", fontsize=8)

    # 5 langkah di area bahaya
    x = ax[1, 1]
    x.scatter(ep, d["haz"], s=6, c="#e8731a", alpha=.4)
    x.plot(ep, moving(d["haz"], W), "k", lw=1.6, label=f"rata-rata {W}")
    x.set(title="Langkah di area bahaya per episode", xlabel="episode", ylabel="hazard_steps")
    x.legend()

    # 6 jumlah state
    x = ax[1, 2]
    x.plot(ep, d["states"], color="#7a4fb5", lw=2)
    x.set(title="State unik yang pernah dikunjungi", xlabel="episode", ylabel="jumlah state")

    for row in ax:
        for x in row:
            x.grid(alpha=.25)
    fig.tight_layout()
    out = HERE / f"training_plot{suffix}.png"
    fig.savefig(out, dpi=110)
    print("ditulis:", out)


if __name__ == "__main__":
    main()
