#!/usr/bin/env python3
"""Gambar jalur robot (evaluasi greedy) di atas peta, dari eval_result.json.

Pakai (dari folder ini, di luar Webots):
    python visualize_path.py                # eval_result.json -> eval_paths.png
    python visualize_path.py --best         # eval_result_best.json -> eval_paths_best.png
    python visualize_path.py --gif          # tambah animasi eval_path.gif (evaluasi pertama)
    python visualize_path.py --episode 2    # pilih evaluasi ke-N untuk gif (mulai 1)

Peta: rintangan (lingkaran/persegi/segitiga), area bahaya (oranye), start
(biru), dan target (hijau). Garis jalur diwarnai menurut hasil evaluasi:
hijau = sampai target, merah = tabrakan, kuning = timeout, ungu = macet.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D

HERE = Path(__file__).resolve().parent
COL = {"goal": "#2e9e4f", "collision": "#d64545", "timeout": "#e0a030", "stuck": "#8a5fc0"}
SHAPE_COL = {"cyl": "#595966", "box": "#8c6640", "tri": "#734d8c"}


def draw_map(ax, d):
    A = d.get("arena", 2.4)
    ax.set_xlim(-0.03, A + 0.03)
    ax.set_ylim(-0.03, A + 0.03)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), A, A, facecolor="#e6e6df", edgecolor="#a33", lw=2))
    for hx, hy, hr in d["hazards"]:
        ax.add_patch(Circle((hx, hy), hr, facecolor="#f27319", alpha=.55, edgecolor="#c25000"))
    sx, sy = d["start"]
    ax.add_patch(Circle((sx, sy), 0.18, facecolor="#4d80f2", alpha=.45))
    for o in d["obstacles"]:
        c = SHAPE_COL[o["shape"]]
        if o["shape"] == "cyl":
            ax.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor=c))
        elif o["shape"] == "box":
            ax.add_patch(Rectangle((o["x"] - o["sx"] / 2, o["y"] - o["sy"] / 2), o["sx"], o["sy"],
                                   facecolor=c,
                                   transform=Affine2D().rotate_around(o["x"], o["y"], o["yaw"]) + ax.transData))
        else:
            ax.add_patch(Polygon(o["pts"], facecolor=c))
    tx, ty = d["target"]
    ax.add_patch(Circle((tx, ty), d["target_r"], facecolor="#1ac233", alpha=.8, edgecolor="#0a7a1e"))
    ax.plot(sx, sy, "ko", ms=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true", help="pakai eval_result_best.json")
    ap.add_argument("--gif", action="store_true", help="buat animasi jalur")
    ap.add_argument("--episode", type=int, default=1, help="evaluasi ke-N untuk gif")
    a = ap.parse_args()
    suffix = "_best" if a.best else ""
    src = HERE / f"eval_result{suffix}.json"
    if not src.exists():
        raise SystemExit(f"tidak ada {src.name} (jalankan training/evaluasi dulu)")
    d = json.loads(src.read_text(encoding="utf-8"))
    eps = d["episodes"]

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_map(ax, d)
    for i, e in enumerate(eps, 1):
        p = e["path"]
        if not p:
            continue
        xs, ys = zip(*p)
        ax.plot(xs, ys, color=COL[e["outcome"]], lw=1.6, alpha=.8,
                label=(f"#{i} {e['heading']:g}° {e['outcome']}" if "heading" in e else f"#{i} {e['outcome']}")
                      + f" ({e['decisions']} langkah, bahaya {e['hazard_steps']})")
        ax.plot(xs[-1], ys[-1], "x", color=COL[e["outcome"]], ms=8, mew=2)
    ax.set_title(f"Jalur evaluasi greedy: {d['success']}/{d['n']} sampai target"
                 + (" (arah awal kompas: " + ", ".join(f"{h:g}°" for h in d["headings"]) + ")" if "headings" in d else ""),
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=6.5, framealpha=.9, ncol=2 if len(eps) > 6 else 1)
    out = HERE / f"eval_paths{suffix}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("ditulis:", out)

    if a.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        e = eps[max(0, min(a.episode, len(eps)) - 1)]
        p = e["path"]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        draw_map(ax, d)
        (trail,) = ax.plot([], [], color=COL[e["outcome"]], lw=2)
        (dot,) = ax.plot([], [], "o", color="k", ms=8)
        title = ax.set_title("")

        def upd(i):
            xs, ys = zip(*p[:i + 1])
            trail.set_data(xs, ys)
            dot.set_data([xs[-1]], [ys[-1]])
            title.set_text(f"langkah {i + 1}/{len(p)}  ({e['outcome']})")
            return trail, dot, title

        step = max(1, len(p) // 150)           # batasi ~150 frame
        frames = list(range(0, len(p), step)) + [len(p) - 1]
        ani = FuncAnimation(fig, upd, frames=frames, blit=False)
        out = HERE / f"eval_path{suffix}.gif"
        ani.save(out, writer=PillowWriter(fps=15))
        print("ditulis:", out)


if __name__ == "__main__":
    main()
