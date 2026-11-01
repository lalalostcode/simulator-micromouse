#!/usr/bin/env python3
"""Heatmap lokasi tabrakan dari episode_log.csv (kolom end_x/end_y).

Pakai (dari folder ini, di luar Webots):
    python plot_collision_heatmap.py                 # semua episode
    python plot_collision_heatmap.py --last 500      # hanya 500 episode terakhir
    python plot_collision_heatmap.py --bins 30       # kehalusan kotak (bawaan 24 per sumbu)

Keluaran collision_heatmap.png: tiga panel dengan skala warna yang sama, yaitu semua
episode, paruh awal, dan paruh akhir, ditambah titik goal (hijau) dan macet (ungu).
Untuk menjalankan bagian lain dari analisis lihat plot_endpoints.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_endpoints import load
from visualize_path import draw_map

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "episode_log.csv"))
    ap.add_argument("--layout", default=None)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--out", default=str(HERE / "collision_heatmap.png"))
    a = ap.parse_args()

    lay_path = Path(a.layout) if a.layout else (HERE / "layout.json" if (HERE / "layout.json").exists() else HERE / "eval_result.json")
    lay = json.loads(lay_path.read_text(encoding="utf-8"))
    A = lay.get("arena", 2.4)
    d = load(a.csv)
    if a.last:
        keep = d["ep"] > d["ep"].max() - a.last
        d = {k: v[keep] for k, v in d.items()}
    n = len(d["ep"])
    half = n // 2
    col = d["out"] == "collision"
    idx = np.arange(n)
    edges = np.linspace(0, A, a.bins + 1)

    def hist(sel):
        h, _, _ = np.histogram2d(d["x"][sel], d["y"][sel], bins=[edges, edges])
        return h

    panels = [("Semua episode", col), (f"Paruh awal (ep {d['ep'][0]}-{d['ep'][max(half - 1, 0)]})", col & (idx < half)),
              (f"Paruh akhir (ep {d['ep'][half] if half < n else d['ep'][-1]}-{d['ep'][-1]})", col & (idx >= half))]
    hs = [hist(sel) for _, sel in panels]
    vmax = max(h.max() for h in hs) or 1

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6))
    fig.suptitle(f"Heatmap lokasi tabrakan: {int(col.sum())} tabrakan dari {n} episode "
                 f"(skala warna sama di ketiga panel)", fontsize=12)
    im = None
    for k, ((title, sel), h) in enumerate(zip(panels, hs)):
        draw_map(ax[k], lay)
        hm = np.ma.masked_where(h.T == 0, h.T)
        im = ax[k].imshow(hm, origin="lower", extent=(0, A, 0, A), cmap="inferno_r", vmin=1, vmax=vmax,
                          alpha=.85, interpolation="nearest", zorder=5)
        g = d["out"] == "goal"
        s = d["out"] == "stuck"
        part = (idx < half) if k == 1 else ((idx >= half) if k == 2 else np.ones(n, bool))
        ax[k].scatter(d["x"][g & part], d["y"][g & part], s=10, c="#1ac233", edgecolors="k", linewidths=.3, zorder=6)
        ax[k].scatter(d["x"][s & part], d["y"][s & part], s=10, c="#8a5fc0", edgecolors="none", alpha=.7, zorder=6)
        ax[k].set_title(f"{title}: {int(sel.sum())} tabrakan")
    fig.colorbar(im, ax=ax, fraction=.015, pad=.01, label="jumlah tabrakan per kotak")
    fig.savefig(a.out, dpi=105, bbox_inches="tight")
    print("ditulis:", a.out)

    # ringkasan: kotak paling rawan
    h = hs[0]
    top = np.dstack(np.unravel_index(np.argsort(h.ravel())[::-1][:5], h.shape))[0]
    print("5 kotak paling rawan (pusat kotak x, y -> jumlah tabrakan):")
    for i, j in top:
        print(f"  ({(edges[i] + edges[i + 1]) / 2:.2f}, {(edges[j] + edges[j + 1]) / 2:.2f}) -> {int(h[i, j])}")


if __name__ == "__main__":
    main()
