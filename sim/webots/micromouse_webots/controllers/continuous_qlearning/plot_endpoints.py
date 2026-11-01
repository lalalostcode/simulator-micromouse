#!/usr/bin/env python3
"""Peta posisi akhir episode: di mana robot menabrak dan di mana ia sampai target.

Butuh episode_log.csv dengan kolom end_x/end_y (dibuat controller versi terbaru)
dan layout.json (tata letak run itu). Pakai, dari folder ini dan di luar Webots:
    python plot_endpoints.py                  # seluruh training
    python plot_endpoints.py --last 300       # hanya 300 episode terakhir
    python plot_endpoints.py --from 400       # mulai episode 400
    python plot_endpoints.py --edge 0.40      # lebar "pinggir" (m dari dinding luar)

Keluaran: endpoints_plot.png (tiga panel: titik akhir per hasil, kepadatan tabrakan,
dan persentase tabrakan di pinggir vs tengah untuk paruh pertama dan kedua) serta
ringkasan angka di konsol.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_path import draw_map

HERE = Path(__file__).resolve().parent
COL = {"goal": "#2e9e4f", "collision": "#d64545", "timeout": "#e0a030", "stuck": "#8a5fc0"}


def load(path):
    txt = Path(path).read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    need = ("episode", "outcome", "hazard_steps", "end_x", "end_y")
    rows = [r for r in csv.DictReader(io.StringIO(txt))
            if (r.get("episode") or "").strip().isdigit() and all(r.get(k) not in (None, "") for k in need)]
    if not rows:
        raise SystemExit("Tidak ada baris dengan end_x/end_y. Jalankan training dengan controller terbaru.")
    return {
        "ep": np.array([int(r["episode"]) for r in rows]),
        "out": np.array([r["outcome"] for r in rows]),
        "haz": np.array([int(r["hazard_steps"]) for r in rows]),
        "x": np.array([float(r["end_x"]) for r in rows]),
        "y": np.array([float(r["end_y"]) for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "episode_log.csv"))
    ap.add_argument("--layout", default=None, help="layout.json (bawaan) atau eval_result.json")
    ap.add_argument("--last", type=int, default=0, help="hanya N episode terakhir")
    ap.add_argument("--from", dest="start", type=int, default=0, help="mulai dari episode ini")
    ap.add_argument("--edge", type=float, default=0.40, help="lebar pita pinggir (m)")
    ap.add_argument("--out", default=str(HERE / "endpoints_plot.png"))
    a = ap.parse_args()

    lay_path = Path(a.layout) if a.layout else (HERE / "layout.json" if (HERE / "layout.json").exists() else HERE / "eval_result.json")
    lay = json.loads(lay_path.read_text(encoding="utf-8"))
    d = load(a.csv)
    keep = d["ep"] >= a.start
    if a.last:
        keep &= d["ep"] > d["ep"].max() - a.last
    d = {k: v[keep] for k, v in d.items()}
    n = len(d["ep"])
    A = lay.get("arena", 2.4)
    wall_dist = np.minimum.reduce([d["x"], d["y"], A - d["x"], A - d["y"]])
    edge = wall_dist < a.edge
    half = n // 2
    phase = np.arange(n) < half

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5))
    fig.suptitle(f"Posisi akhir episode: {n} episode (ep {d['ep'].min()}-{d['ep'].max()}), "
                 f"layout dari {lay_path.name}", fontsize=12)

    # 1 titik akhir menurut hasil
    draw_map(ax[0], lay)
    for name in ("collision", "stuck", "timeout", "goal"):
        m = d["out"] == name
        ax[0].scatter(d["x"][m], d["y"][m], s=14, c=COL[name], alpha=.55, label=f"{name} ({m.sum()})", edgecolors="none")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Titik akhir tiap episode")

    # 2 kepadatan tabrakan
    draw_map(ax[1], lay)
    m = d["out"] == "collision"
    if m.any():
        hb = ax[1].hexbin(d["x"][m], d["y"][m], gridsize=24, extent=(0, A, 0, A), cmap="Reds", mincnt=1, alpha=.85)
        fig.colorbar(hb, ax=ax[1], fraction=.046, label="jumlah tabrakan")
    ax[1].set_title("Kepadatan lokasi tabrakan")

    # 3 tabrakan pinggir vs tengah, per paruh
    ax[2].set_title(f"Lokasi tabrakan: pinggir (< {a.edge:g} m dari dinding) vs tengah")
    labels, edge_pct, inner_pct, counts = [], [], [], []
    for name, ph in ((f"paruh 1 (ep {d['ep'][0]}-{d['ep'][max(half - 1, 0)]})", phase),
                     (f"paruh 2 (ep {d['ep'][half] if half < n else d['ep'][-1]}-{d['ep'][-1]})", ~phase)):
        c = (d["out"] == "collision") & ph
        k = int(c.sum())
        e = float(edge[c].mean() * 100) if k else 0.0
        labels.append(name)
        edge_pct.append(e)
        inner_pct.append(100 - e if k else 0.0)
        counts.append(k)
    xs = np.arange(2)
    ax[2].bar(xs - .18, edge_pct, .36, color="#4477cc", label="di pinggir")
    ax[2].bar(xs + .18, inner_pct, .36, color="#d64545", label="di tengah")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f"{l}\n{k} tabrakan" for l, k in zip(labels, counts)], fontsize=9)
    ax[2].set_ylabel("% dari tabrakan")
    ax[2].set_ylim(0, 100)
    ax[2].legend()
    ax[2].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(a.out, dpi=105)
    print("ditulis:", a.out)

    # ringkasan angka
    print(f"\nEpisode dianalisis: {n}. Pita pinggir = < {a.edge:g} m dari dinding luar.")
    for name, ph in (("paruh 1", phase), ("paruh 2", ~phase), ("semua", np.ones(n, bool))):
        sel = ph
        c = sel & (d["out"] == "collision")
        g = sel & (d["out"] == "goal")
        t = sel & np.isin(d["out"], ("timeout", "stuck"))
        k = int(c.sum())
        print(f"  {name:8s}: goal {int(g.sum()):4d} | tabrakan {k:4d} "
              f"(pinggir {int((c & edge).sum()):4d} = {((c & edge).sum() / k * 100 if k else 0):5.1f}%, "
              f"tengah {int((c & ~edge).sum()):4d}) | timeout/macet {int(t.sum()):3d} | "
              f"tabrakan sambil menyentuh area bahaya: {int((c & (d['haz'] > 0)).sum())}")


if __name__ == "__main__":
    main()
