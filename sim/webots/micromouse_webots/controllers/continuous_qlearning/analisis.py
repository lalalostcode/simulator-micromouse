#!/usr/bin/env python3
"""Analisis hasil continuous_qlearning: satu grafik = satu berkas, semua di hasil_analisis/.

Pakai (dari folder ini, di luar Webots):
    python analisis.py                  # episode_log.csv + eval_result.json -> hasil_analisis/*.png
    python analisis.py --best           # file *_best -> hasil_analisis/*_best.png  (yang di-push)
    python analisis.py --best --gif     # tambah hasil_analisis/13_animasi_jalur_evaluasi_best.gif
    python analisis.py --last 500       # hanya 500 episode terakhir
    python analisis.py --from 400       # mulai dari episode 400
    python analisis.py --outdir X       # folder keluaran lain (bawaan hasil_analisis)

Berkas yang dibuat (tiap grafik terpisah, awalan nomor):
    01_reward              05_langkah_area_bahaya      09_tabrakan_pinggir_tengah
    02_success_rate_epsilon 06_jumlah_state            10_heatmap_tabrakan_semua
    03_langkah_episode_sukses 07_jalur_evaluasi        11_heatmap_tabrakan_paruh_awal
    04_komposisi_hasil     08_titik_akhir              12_heatmap_tabrakan_paruh_akhir
    ringkasan.txt (angka-angka ringkasan), 13_animasi_jalur_evaluasi.gif (bila --gif)
Tiap nama berakhiran _best bila memakai --best. Heatmap 10-12 memakai skala warna sama.
Peta (rintangan, area bahaya, start, target) dibaca dari eval_result*.json.
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
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D

HERE = Path(__file__).resolve().parent
COL = {"goal": "#2e9e4f", "collision": "#d64545", "timeout": "#e0a030", "stuck": "#8a5fc0"}
SHAPE_COL = {"cyl": "#595966", "box": "#8c6640", "tri": "#734d8c"}
OUTCOMES = ("goal", "collision", "stuck", "timeout")


# ---------------------------------------------------------------- data
def moving(x, w):
    """Rata-rata bergerak (panjang keluaran = masukan; di awal memakai data yang ada)."""
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    lo = np.maximum(0, np.arange(len(x)) + 1 - w)
    return (c[np.arange(len(x)) + 1] - c[lo]) / (np.arange(len(x)) + 1 - lo)


def load_log(path):
    # errors="ignore" dan penyaring baris: CSV bisa berisi padding kosong bila ditimpa
    # saat file masih dipegang proses Webots lain.
    txt = Path(path).read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    need = ("epsilon", "decisions", "reward", "outcome", "hazard_steps", "states")
    rows = [r for r in csv.DictReader(io.StringIO(txt))
            if (r.get("episode") or "").strip().isdigit() and all(r.get(k) not in (None, "") for k in need)]
    if not rows:
        raise SystemExit(f"{path} tidak berisi baris episode yang valid")
    d = {
        "ep": np.array([int(r["episode"]) for r in rows]),
        "eps": np.array([float(r["epsilon"]) for r in rows]),
        "dec": np.array([int(r["decisions"]) for r in rows]),
        "rew": np.array([float(r["reward"]) for r in rows]),
        "out": np.array([r["outcome"] for r in rows]),
        "haz": np.array([int(r["hazard_steps"]) for r in rows]),
        "states": np.array([int(r["states"]) for r in rows]),
    }
    has_xy = all((r.get("end_x") or "") != "" and (r.get("end_y") or "") != "" for r in rows)
    d["x"] = np.array([float(r["end_x"]) for r in rows]) if has_xy else None
    d["y"] = np.array([float(r["end_y"]) for r in rows]) if has_xy else None
    return d


def draw_map(ax, lay):
    A = lay.get("arena", 2.4)
    ax.set_xlim(-0.03, A + 0.03)
    ax.set_ylim(-0.03, A + 0.03)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), A, A, facecolor="#e6e6df", edgecolor="#a33", lw=2))
    for hx, hy, hr in lay["hazards"]:
        ax.add_patch(Circle((hx, hy), hr, facecolor="#f27319", alpha=.55, edgecolor="#c25000"))
    sx, sy = lay["start"]
    ax.add_patch(Circle((sx, sy), 0.18, facecolor="#4d80f2", alpha=.45))
    for o in lay["obstacles"]:
        c = SHAPE_COL[o["shape"]]
        if o["shape"] == "cyl":
            ax.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor=c))
        elif o["shape"] == "box":
            ax.add_patch(Rectangle((o["x"] - o["sx"] / 2, o["y"] - o["sy"] / 2), o["sx"], o["sy"], facecolor=c,
                                   transform=Affine2D().rotate_around(o["x"], o["y"], o["yaw"]) + ax.transData))
        else:
            ax.add_patch(Polygon(o["pts"], facecolor=c))
    tx, ty = lay["target"]
    ax.add_patch(Circle((tx, ty), lay["target_r"], facecolor="#1ac233", alpha=.8, edgecolor="#0a7a1e"))
    ax.plot(sx, sy, "ko", ms=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---------------------------------------------------------------- panel
def panel_success(ax, d, W):
    goal = d["out"] == "goal"
    ax.plot(d["ep"], moving(goal.astype(float), W), color=COL["goal"], lw=2, label=f"success rate ({W} ep)")
    ax.set(title="Success rate dan epsilon", xlabel="episode", ylabel="success rate", ylim=(-0.02, 1.02))
    a2 = ax.twinx()
    a2.plot(d["ep"], d["eps"], color="#4477cc", lw=1.3, ls="--", label="epsilon")
    a2.set_ylabel("epsilon", color="#4477cc")
    a2.set_ylim(0, 1.02)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax.grid(alpha=.25)


def panel_composition(ax, d, W):
    n = len(d["ep"])
    nb = max(1, n // W)
    edges = np.linspace(0, n, nb + 1).astype(int)
    centers = [(d["ep"][edges[i]] + d["ep"][edges[i + 1] - 1]) / 2 for i in range(nb)]
    width = (d["ep"][-1] - d["ep"][0] + 1) / nb * 0.9
    bottom = np.zeros(nb)
    for name in OUTCOMES:
        frac = np.array([(d["out"][edges[i]:edges[i + 1]] == name).mean() for i in range(nb)])
        ax.bar(centers, frac, width, bottom=bottom, color=COL[name], label=name)
        bottom += frac
    ax.set(title=f"Komposisi hasil per blok (~{W} episode)", xlabel="episode", ylabel="proporsi", ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)


def panel_reward(ax, d, W):
    ax.scatter(d["ep"], d["rew"], s=5, c=[COL.get(o, "#888") for o in d["out"]], alpha=.4)
    ax.plot(d["ep"], moving(d["rew"], W), "k", lw=1.8, label=f"rata-rata {W}")
    ax.set(title="Reward per episode", xlabel="episode", ylabel="reward")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.25)


def panel_steps_goal(ax, d, W):
    g = d["out"] == "goal"
    if g.any():
        ax.scatter(d["ep"][g], d["dec"][g], s=8, c=COL["goal"], alpha=.6)
        ax.plot(d["ep"][g], moving(d["dec"][g], max(5, W // 5)), "k", lw=1.6, label="rata-rata")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "belum ada episode sukses", ha="center", va="center", transform=ax.transAxes)
    ax.set(title="Langkah keputusan (episode sukses)", xlabel="episode", ylabel="decisions")
    ax.grid(alpha=.25)


def panel_hazard(ax, d, W):
    ax.scatter(d["ep"], d["haz"], s=6, c="#e8731a", alpha=.4)
    ax.plot(d["ep"], moving(d["haz"], W), "k", lw=1.6, label=f"rata-rata {W}")
    ax.set(title="Langkah di area bahaya per episode", xlabel="episode", ylabel="hazard_steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)


def panel_states(ax, d):
    ax.plot(d["ep"], d["states"], color="#7a4fb5", lw=2)
    ax.set(title="State unik yang pernah dikunjungi", xlabel="episode", ylabel="jumlah state")
    ax.grid(alpha=.25)


def panel_paths(ax, lay):
    draw_map(ax, lay)
    eps = lay["episodes"]
    for e in eps:
        p = e["path"]
        if not p:
            continue
        xs, ys = zip(*p)
        ax.plot(xs, ys, color=COL[e["outcome"]], lw=1.5, alpha=.75)
        ax.plot(xs[-1], ys[-1], "x", color=COL[e["outcome"]], ms=7, mew=2)
    ax.set_title(f"Jalur evaluasi greedy: {lay['success']}/{lay['n']} sampai target")


def panel_endpoints(ax, d, lay):
    draw_map(ax, lay)
    if d["x"] is None:
        ax.text(.5, .5, "log tanpa kolom end_x/end_y", transform=ax.transAxes, ha="center")
        return
    for name in ("collision", "stuck", "timeout", "goal"):
        m = d["out"] == name
        ax.scatter(d["x"][m], d["y"][m], s=14, c=COL[name], alpha=.55, edgecolors="none", zorder=5,
                   label=f"{name} ({int(m.sum())})")
    ax.legend(loc="upper left", fontsize=8, framealpha=.9)
    ax.set_title("Titik akhir tiap episode")


def _wall_dist(d, lay):
    A = lay.get("arena", 2.4)
    return np.minimum.reduce([d["x"], d["y"], A - d["x"], A - d["y"]])


def panel_edge_bar(ax, d, lay, edge):
    ax.set_title(f"Lokasi tabrakan: pinggir (< {edge:g} m dari dinding) vs tengah")
    if d["x"] is None:
        ax.text(.5, .5, "log tanpa kolom end_x/end_y", transform=ax.transAxes, ha="center")
        return
    n = len(d["ep"])
    half = n // 2
    wd = _wall_dist(d, lay)
    labels, e_pct, i_pct = [], [], []
    for name, sel in ((f"paruh 1 (ep {d['ep'][0]}-{d['ep'][max(half - 1, 0)]})", np.arange(n) < half),
                      (f"paruh 2 (ep {d['ep'][half] if half < n else d['ep'][-1]}-{d['ep'][-1]})", np.arange(n) >= half)):
        c = sel & (d["out"] == "collision")
        k = int(c.sum())
        pe = float((wd[c] < edge).mean() * 100) if k else 0.0
        labels.append(f"{name}\n{k} tabrakan")
        e_pct.append(pe)
        i_pct.append(100 - pe if k else 0.0)
    xs = np.arange(2)
    ax.bar(xs - .18, e_pct, .36, color="#4477cc", label="di pinggir")
    ax.bar(xs + .18, i_pct, .36, color="#d64545", label="di tengah")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% dari tabrakan")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=.25)


def heat_hist(d, lay, bins, sel):
    A = lay.get("arena", 2.4)
    edges = np.linspace(0, A, bins + 1)
    h, _, _ = np.histogram2d(d["x"][sel], d["y"][sel], bins=[edges, edges])
    return h


def panel_heat(ax, d, lay, h, vmax, title, part):
    draw_map(ax, lay)
    A = lay.get("arena", 2.4)
    im = ax.imshow(np.ma.masked_where(h.T == 0, h.T), origin="lower", extent=(0, A, 0, A), cmap="inferno_r",
                   vmin=1, vmax=vmax, alpha=.85, interpolation="nearest", zorder=5)
    for name, size in (("goal", 10), ("stuck", 9)):
        m = (d["out"] == name) & part
        ax.scatter(d["x"][m], d["y"][m], s=size, c=COL[name], edgecolors="k" if name == "goal" else "none",
                   linewidths=.3, alpha=.8, zorder=6)
    ax.set_title(title)
    return im


def summary_lines(d, lay, W, edge):
    """Ringkasan angka sebagai daftar baris teks."""
    n = len(d["ep"])
    cnt = {k: int((d["out"] == k).sum()) for k in OUTCOMES}
    m = moving((d["out"] == "goal").astype(float), W)
    k = int(np.argmax(m))
    L = ["RINGKASAN TRAINING",
         f"  episode            : {n} (ep {d['ep'][0]}-{d['ep'][-1]})",
         f"  hasil              : goal {cnt['goal']} ({cnt['goal'] / n * 100:.0f}%), tabrakan {cnt['collision']}, "
         f"macet {cnt['stuck']}, timeout {cnt['timeout']}",
         f"  success rate maks  : {m[k]:.2f} (jendela {W}) di episode {d['ep'][k]}",
         f"  epsilon akhir      : {d['eps'][-1]:.2f}    state dikunjungi: {d['states'][-1]}",
         f"  langkah di bahaya  : rata-rata {d['haz'].mean():.1f} per episode",
         ""]
    if d["x"] is not None and cnt["collision"]:
        wd = _wall_dist(d, lay)
        half = n // 2
        L.append(f"TABRAKAN: pinggir (< {edge:g} m dari dinding) vs tengah")
        for name, sel in (("paruh 1", np.arange(n) < half), ("paruh 2", np.arange(n) >= half)):
            c = sel & (d["out"] == "collision")
            kk = int(c.sum())
            pi = int((c & (wd < edge)).sum())
            L.append(f"  {name}: {kk} tabrakan, pinggir {pi} ({pi / kk * 100 if kk else 0:.0f}%), tengah {kk - pi}")
        L.append("")
    L.append("EVALUASI GREEDY (epsilon 0)")
    L.append(f"  sampai target      : {lay['success']}/{lay['n']}")
    by = lay.get("by_heading")
    if by:
        for h, outs in by.items():
            L.append(f"  arah {float(h):5.1f} deg    : {sum(o == 'goal' for o in outs)}/{len(outs)} goal")
    dec = [e["decisions"] for e in lay["episodes"] if e["outcome"] == "goal"]
    if dec:
        L.append(f"  langkah (goal)     : {min(dec)}-{max(dec)} keputusan (~{max(dec) * 0.128:.0f} s)")
    L.append(f"  langkah di bahaya  : {sum(e['hazard_steps'] for e in lay['episodes'])} (total semua uji)")
    return L


# ---------------------------------------------------------------- gif
def make_gif(lay, which, out):
    from matplotlib.animation import FuncAnimation, PillowWriter
    eps = lay["episodes"]
    e = eps[max(0, min(which, len(eps)) - 1)]
    p = e["path"]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw_map(ax, lay)
    (trail,) = ax.plot([], [], color=COL[e["outcome"]], lw=2)
    (dot,) = ax.plot([], [], "o", color="k", ms=8)
    title = ax.set_title("")

    def upd(i):
        xs, ys = zip(*p[:i + 1])
        trail.set_data(xs, ys)
        dot.set_data([xs[-1]], [ys[-1]])
        title.set_text(f"langkah {i + 1}/{len(p)}  ({e['outcome']})")
        return trail, dot, title

    step = max(1, len(p) // 150)
    frames = list(range(0, len(p), step)) + [len(p) - 1]
    FuncAnimation(fig, upd, frames=frames, blit=False).save(out, writer=PillowWriter(fps=15))
    plt.close(fig)
    print("ditulis:", out)


# ---------------------------------------------------------------- main
def save_one(outdir, name, sfx, figsize, draw):
    """Satu grafik = satu figure = satu berkas."""
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.tight_layout()
    path = outdir / f"{name}{sfx}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("ditulis:", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", action="store_true", help="pakai file *_best, nama keluaran berakhiran _best")
    ap.add_argument("--gif", action="store_true", help="tambah animasi jalur evaluasi")
    ap.add_argument("--episode", type=int, default=1, help="uji evaluasi ke-N untuk gif (mulai 1)")
    ap.add_argument("--last", type=int, default=0, help="hanya N episode terakhir")
    ap.add_argument("--from", dest="start", type=int, default=0, help="mulai dari episode ini")
    ap.add_argument("--window", type=int, default=50, help="jendela rata-rata bergerak")
    ap.add_argument("--bins", type=int, default=24, help="kehalusan heatmap (kotak per sumbu)")
    ap.add_argument("--edge", type=float, default=0.40, help="lebar pita pinggir (m)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--eval", default=None, help="berkas eval_result*.json (sekaligus sumber peta)")
    ap.add_argument("--outdir", default=None, help="folder keluaran (bawaan hasil_analisis)")
    a = ap.parse_args()
    sfx = "_best" if a.best else ""
    csv_path = Path(a.csv) if a.csv else HERE / f"episode_log{sfx}.csv"
    eval_path = Path(a.eval) if a.eval else HERE / f"eval_result{sfx}.json"
    outdir = Path(a.outdir) if a.outdir else HERE / "hasil_analisis"
    for p in (csv_path, eval_path):
        if not p.exists():
            raise SystemExit(f"tidak ada {p.name} (jalankan training dulu)")
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_log(csv_path)
    keep = d["ep"] >= a.start
    if a.last:
        keep &= d["ep"] > d["ep"].max() - a.last
    d = {k: (v[keep] if v is not None else None) for k, v in d.items()}
    lay = json.loads(eval_path.read_text(encoding="utf-8"))
    W = a.window
    wide, square = (9, 5.5), (7.5, 7.5)

    # ---- grafik training (tiap satu berkas)
    save_one(outdir, "01_reward", sfx, wide, lambda ax: panel_reward(ax, d, W))
    save_one(outdir, "02_success_rate_epsilon", sfx, wide, lambda ax: panel_success(ax, d, W))
    save_one(outdir, "03_langkah_episode_sukses", sfx, wide, lambda ax: panel_steps_goal(ax, d, W))
    save_one(outdir, "04_komposisi_hasil", sfx, wide, lambda ax: panel_composition(ax, d, W))
    save_one(outdir, "05_langkah_area_bahaya", sfx, wide, lambda ax: panel_hazard(ax, d, W))
    save_one(outdir, "06_jumlah_state", sfx, wide, lambda ax: panel_states(ax, d))

    # ---- grafik berbasis peta (tiap satu berkas)
    save_one(outdir, "07_jalur_evaluasi", sfx, square, lambda ax: panel_paths(ax, lay))
    save_one(outdir, "08_titik_akhir", sfx, square, lambda ax: panel_endpoints(ax, d, lay))
    save_one(outdir, "09_tabrakan_pinggir_tengah", sfx, wide, lambda ax: panel_edge_bar(ax, d, lay, a.edge))
    if d["x"] is not None:
        n = len(d["ep"])
        half = n // 2
        idx = np.arange(n)
        col = d["out"] == "collision"
        parts = [("10_heatmap_tabrakan_semua", "Heatmap tabrakan: semua episode", np.ones(n, bool)),
                 ("11_heatmap_tabrakan_paruh_awal", f"Heatmap tabrakan: paruh awal (ep {d['ep'][0]}-{d['ep'][max(half - 1, 0)]})", idx < half),
                 ("12_heatmap_tabrakan_paruh_akhir", f"Heatmap tabrakan: paruh akhir (ep {d['ep'][half] if half < n else d['ep'][-1]}-{d['ep'][-1]})", idx >= half)]
        hs = [heat_hist(d, lay, a.bins, col & part) for _, _, part in parts]
        vmax = max(1, max(h.max() for h in hs))          # skala warna sama di ketiga heatmap

        def heat_draw(title, part, h):
            def draw(ax):
                im = panel_heat(ax, d, lay, h, vmax, f"{title}: {int((col & part).sum())} tabrakan", part)
                ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.02, label="tabrakan per kotak (skala sama)")
            return draw

        for (name, title, part), h in zip(parts, hs):
            save_one(outdir, name, sfx, square, heat_draw(title, part, h))
    else:
        print("log tanpa kolom end_x/end_y: grafik 08-12 dilewati/tidak lengkap")

    # ---- ringkasan angka
    txt = "\n".join(summary_lines(d, lay, W, a.edge)) + "\n"
    (outdir / f"ringkasan{sfx}.txt").write_text(txt, encoding="utf-8")
    print("ditulis:", outdir / f"ringkasan{sfx}.txt")

    if a.gif:
        make_gif(lay, a.episode, outdir / f"13_animasi_jalur_evaluasi{sfx}.gif")


if __name__ == "__main__":
    main()
