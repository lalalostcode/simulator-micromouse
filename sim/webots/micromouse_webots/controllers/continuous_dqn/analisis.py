#!/usr/bin/env python3
"""Analisis hasil continuous_dqn: satu grafik = satu berkas, semua di hasil_analisis/.

Memakai ulang fungsi gambar dari ../continuous_qlearning/analisis.py (peta, reward, success rate, dst)
dan menambah grafik khusus DQN (Q rata-rata, pengaman/pemecah, porsi aksi, porsi keputusan greedy).

Pakai (dari folder ini, di luar Webots):
    python analisis.py                  # episode_log.csv + eval_result.json -> hasil_analisis/*.png
    python analisis.py --best           # file *_best -> hasil_analisis/*_best.png
    python analisis.py --last 500       # hanya 500 episode terakhir
    python analisis.py --from 400       # mulai dari episode 400
    python analisis.py --gif --episode 3   # animasi uji evaluasi ke-3
    python analisis.py --outdir X       # folder keluaran lain

Berkas yang dibuat:
    01_reward                 05_langkah_area_bahaya      09_tabrakan_pinggir_tengah   13_porsi_aksi
    02_success_rate_epsilon   06_q_rata_rata              10_heatmap_tabrakan          ringkasan.txt
    03_langkah_episode_sukses 07_jalur_evaluasi           11_animasi_jalur_evaluasi.gif (--gif)
    04_komposisi_hasil        08_titik_akhir
Hanya bila lapisan buatan aktif (log punya pengaman/pemecah/putaran > 0, mode PURE_DQN=False):
    12_pengaman_pemecah, 14_porsi_greedy  (di mode murni dilewati: 12 selalu 0, 14 = 1 - epsilon).
Tiap nama berakhiran _best bila memakai --best.
"""
import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "continuous_qlearning"))
import analisis as qa                                    # noqa: E402  (fungsi gambar bersama)
from analisis import plt, np as _np, moving, save_one    # noqa: E402,F401

Y_MIN, Y_MAX = -15.0, 8.0     # batas target DQN (dqn_agent.py), skala imbalan x0,05
AKSI = ["0 lurus cepat", "1 lurus pelan", "2 kiri landai", "3 kanan landai", "4 kiri tajam",
        "5 kanan tajam", "6 kiri tajam pelan", "7 kanan tajam pelan", "8 putar kiri", "9 putar kanan"]
CMAP = plt.get_cmap("tab10")


def load_dqn(path):
    """Kolom dasar (seperti Q-learning) + kolom DQN: greedy, aksi_0..9, pengaman, pemecah, q_rata."""
    txt = Path(path).read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    need = ("epsilon", "decisions", "reward", "outcome", "hazard_steps", "states")
    rows = [r for r in csv.DictReader(io.StringIO(txt))
            if (r.get("episode") or "").strip().isdigit() and all(r.get(k) not in (None, "") for k in need)]
    if not rows:
        raise SystemExit(f"{path} tidak berisi baris episode yang valid")

    def col(name, cast=float, default=0.0):
        return np.array([cast(r[name]) if r.get(name) not in (None, "") else default for r in rows])

    d = {"ep": col("episode", int), "eps": col("epsilon"), "dec": col("decisions", int), "rew": col("reward"),
         "out": np.array([r["outcome"] for r in rows]), "haz": col("hazard_steps", int),
         "states": col("states", int), "x": col("end_x"), "y": col("end_y"),
         "head": col("start_heading"), "greedy": col("greedy"),
         "pengaman": col("pengaman"), "pemecah": col("pemecah"), "putaran": col("putaran"),
         "q": col("q_rata") if "q_rata" in rows[0] else None}
    if "end_x" not in rows[0]:
        d["x"] = d["y"] = None
    d["aksi"] = np.stack([col(f"aksi_{i}") for i in range(10)], 1)
    return d


def has_layers(d):
    """True bila lapisan buatan pernah aktif (mode PURE_DQN=False)."""
    return float(d["pengaman"].sum() + d["pemecah"].sum() + d["putaran"].sum()) > 0


def panel_q(ax, d, W):
    if d["q"] is None:
        ax.text(.5, .5, "log tanpa kolom q_rata (run lama)", ha="center", va="center", transform=ax.transAxes)
        return
    ax.scatter(d["ep"], d["q"], s=5, c="#7a4fb5", alpha=.3)
    ax.plot(d["ep"], moving(d["q"], W), "k", lw=1.8, label=f"rata-rata {W}")
    ax.axhline(Y_MAX, color="#d64545", ls="--", lw=1, label=f"batas target {Y_MAX:g}")
    ax.axhline(0, color="#999", lw=.8)
    ax.set(title="Q rata-rata aksi yang dijalankan (skala imbalan x0,05)", xlabel="episode", ylabel="Q")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)


def panel_intervensi(ax, d, W):
    ax.plot(d["ep"], moving(d["pengaman"], W), color="#2d6fd6", lw=1.8, label="pengaman (aksi diganti refleks)")
    ax.plot(d["ep"], moving(d["pemecah"], W), color="#d6772d", lw=1.8, label="pemecah (aktivasi per episode)")
    ax.plot(d["ep"], moving(d["putaran"], W), color="#7a4fb5", lw=1.8, label="pemutus orbit (telusur mundur)")
    ax.set(title="Intervensi pengaman, pemecah, dan pemutus orbit per episode", xlabel="episode", ylabel="jumlah")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)


def panel_aksi(ax, d, W):
    n = len(d["ep"])
    nb = max(1, n // W)
    edges = np.linspace(0, n, nb + 1).astype(int)
    centers = [(d["ep"][edges[i]] + d["ep"][edges[i + 1] - 1]) / 2 for i in range(nb)]
    width = (d["ep"][-1] - d["ep"][0] + 1) / nb * 0.9
    bottom = np.zeros(nb)
    for k in range(10):
        tot = np.array([d["aksi"][edges[i]:edges[i + 1]].sum() for i in range(nb)])
        part = np.array([d["aksi"][edges[i]:edges[i + 1], k].sum() for i in range(nb)])
        frac = np.divide(part, tot, out=np.zeros(nb), where=tot > 0)
        ax.bar(centers, frac, width, bottom=bottom, color=CMAP(k), label=AKSI[k])
        bottom += frac
    ax.set(title=f"Porsi aksi yang dijalankan per blok (~{W} episode)", xlabel="episode", ylabel="porsi", ylim=(0, 1))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=7)


def panel_greedy(ax, d, W):
    share = d["greedy"] / np.maximum(d["dec"], 1)
    ax.scatter(d["ep"], share, s=5, c="#2e9e4f", alpha=.3)
    ax.plot(d["ep"], moving(share, W), "k", lw=1.8, label=f"rata-rata {W}")
    ax.plot(d["ep"], 1 - d["eps"], color="#4477cc", ls="--", lw=1.2, label="1 - epsilon")
    ax.set(title="Porsi keputusan greedy (tidak diganti pengaman/pemecah)", xlabel="episode",
           ylabel="greedy / keputusan", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)


def summary(d, lay, W, edge):
    n = len(d["ep"])
    cnt = {k: int((d["out"] == k).sum()) for k in qa.OUTCOMES}
    m = moving((d["out"] == "goal").astype(float), W)
    k = int(np.argmax(m[W - 1:])) + W - 1 if n >= W else int(np.argmax(m))     # hanya jendela penuh
    L = ["RINGKASAN TRAINING",
         f"  episode            : {n} (ep {d['ep'][0]}-{d['ep'][-1]})",
         f"  hasil              : goal {cnt['goal']} ({cnt['goal'] / n * 100:.0f}%), tabrakan {cnt['collision']}, "
         f"macet {cnt['stuck']}, timeout {cnt['timeout']}",
         f"  success rate maks  : {m[k]:.2f} (jendela {W}) di episode {d['ep'][k]}",
         f"  epsilon akhir      : {d['eps'][-1]:.2f}",
         f"  langkah di bahaya  : rata-rata {d['haz'].mean():.1f} per episode"]
    if d["x"] is not None and cnt["collision"]:
        wd = qa._wall_dist(d, lay)
        c = d["out"] == "collision"
        kk = int(c.sum())
        pi = int((wd[c] < edge).sum())
        L += ["", f"TABRAKAN: pinggir (< {edge:g} m dari dinding) vs tengah",
              f"  {kk} tabrakan: pinggir {pi} ({pi / kk * 100:.0f}%), tengah {kk - pi} ({(kk - pi) / kk * 100:.0f}%)"]
    L += ["", "EVALUASI GREEDY (epsilon 0)", f"  sampai target      : {lay['success']}/{lay['n']}"]
    for h, outs in (lay.get("by_heading") or {}).items():
        L.append(f"  arah {float(h):5.1f} deg    : {sum(o == 'goal' for o in outs)}/{len(outs)} goal")
    dec = [e["decisions"] for e in lay["episodes"] if e["outcome"] == "goal"]
    if dec:
        L.append(f"  langkah (goal)     : {min(dec)}-{max(dec)} keputusan (~{max(dec) * 0.128:.0f} s)")
    tail = slice(max(0, n - 100), n)
    cnt = Counter(d["out"][tail])
    g = (d["out"][tail] == "goal")
    L += ["", "100 EPISODE TERAKHIR",
          "  hasil              : " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in qa.OUTCOMES)]
    if g.any():
        dec = d["dec"][tail][g]
        L.append(f"  keputusan per goal : rata-rata {dec.mean():.0f} (min {dec.min()}, maks {dec.max()})")
    if has_layers(d):
        L.append(f"  pengaman / pemecah : {d['pengaman'][tail].mean():.0f} / {d['pemecah'][tail].mean():.1f} per episode; "
                 f"pemutus orbit {d['putaran'][tail].mean():.2f} per episode")
        L.append(f"  porsi greedy       : {(d['greedy'][tail] / np.maximum(d['dec'][tail], 1)).mean() * 100:.0f}% keputusan")
    else:
        L.append("  mode               : DQN murni (tanpa pengaman, pemecah, pemutus orbit)")
    if d["q"] is not None:
        q = d["q"][tail]
        L.append(f"  Q rata-rata        : {q.mean():.2f} (batas atas target {Y_MAX:g}; menempel di batas = tanda optimis)")
    tot = d["aksi"][tail].sum(0)
    L.append("  aksi dijalankan    : " + ", ".join(f"{i}={tot[i] / max(tot.sum(), 1) * 100:.0f}%" for i in range(10)))
    bad = d["out"][tail] != "goal"
    if d["x"] is not None and bad.any():
        c = Counter((str(o), round(float(x), 1), round(float(y), 1))
                    for o, x, y in zip(d["out"][tail][bad], d["x"][tail][bad], d["y"][tail][bad]))
        L.append("  titik gagal terbanyak (jenis, x, y): " + "; ".join(f"{k} x{v}" for k, v in c.most_common(5)))
    L += ["", "EVALUASI PER ARAH (detail uji)"]
    for e in lay["episodes"]:
        L.append(f"  arah {e['heading']:5.1f}: {e['outcome']:9s} {e['decisions']:4d} keputusan, hazard {e['hazard_steps']}")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", action="store_true", help="pakai file *_best, nama keluaran berakhiran _best")
    ap.add_argument("--gif", action="store_true", help="tambah animasi jalur evaluasi")
    ap.add_argument("--episode", type=int, default=1, help="uji evaluasi ke-N untuk gif (mulai 1)")
    ap.add_argument("--last", type=int, default=0, help="hanya N episode terakhir")
    ap.add_argument("--from", dest="start", type=int, default=0, help="mulai dari episode ini")
    ap.add_argument("--window", type=int, default=50, help="jendela rata-rata bergerak")
    ap.add_argument("--bins", type=int, default=24, help="kehalusan heatmap")
    ap.add_argument("--edge", type=float, default=0.40, help="lebar pita pinggir (m)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    sfx = "_best" if a.best else ""
    csv_path = Path(a.csv) if a.csv else HERE / f"episode_log{sfx}.csv"
    eval_path = Path(a.eval) if a.eval else HERE / f"eval_result{sfx}.json"
    outdir = Path(a.outdir) if a.outdir else HERE / "hasil_analisis"
    for p in (csv_path, eval_path):
        if not p.exists():
            raise SystemExit(f"tidak ada {p.name} (jalankan training dulu)")
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_dqn(csv_path)
    keep = d["ep"] >= a.start
    if a.last:
        keep &= d["ep"] > d["ep"].max() - a.last
    d = {k: (v[keep] if v is not None else None) for k, v in d.items()}
    lay = json.loads(eval_path.read_text(encoding="utf-8"))
    W = a.window
    wide, square = (9, 5.5), (7.5, 7.5)

    save_one(outdir, "01_reward", sfx, wide, lambda ax: qa.panel_reward(ax, d, W))
    save_one(outdir, "02_success_rate_epsilon", sfx, wide, lambda ax: qa.panel_success(ax, d, W))
    save_one(outdir, "03_langkah_episode_sukses", sfx, wide, lambda ax: qa.panel_steps_goal(ax, d, W))
    save_one(outdir, "04_komposisi_hasil", sfx, wide, lambda ax: qa.panel_composition(ax, d, W))
    save_one(outdir, "05_langkah_area_bahaya", sfx, wide, lambda ax: qa.panel_hazard(ax, d, W))
    save_one(outdir, "06_q_rata_rata", sfx, wide, lambda ax: panel_q(ax, d, W))
    save_one(outdir, "07_jalur_evaluasi", sfx, square, lambda ax: qa.panel_paths(ax, lay))
    if d["x"] is not None:
        save_one(outdir, "08_titik_akhir", sfx, square, lambda ax: qa.panel_endpoints(ax, d, lay))
        save_one(outdir, "09_tabrakan_pinggir_tengah", sfx, wide, lambda ax: qa.panel_edge_bar(ax, d, lay, a.edge))
        col = d["out"] == "collision"
        h = qa.heat_hist(d, lay, a.bins, col)

        def draw_heat(ax):
            im = qa.panel_heat(ax, d, lay, h, max(1, h.max()), f"Heatmap tabrakan: {int(col.sum())} tabrakan",
                               np.ones(len(d["ep"]), bool))
            ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.02, label="tabrakan per kotak")

        save_one(outdir, "10_heatmap_tabrakan", sfx, square, draw_heat)
    save_one(outdir, "13_porsi_aksi", sfx, (11, 5.5), lambda ax: panel_aksi(ax, d, W))
    if has_layers(d):
        save_one(outdir, "12_pengaman_pemecah", sfx, wide, lambda ax: panel_intervensi(ax, d, W))
        save_one(outdir, "14_porsi_greedy", sfx, wide, lambda ax: panel_greedy(ax, d, W))
    else:
        print("mode DQN murni: grafik 12 (pengaman/pemecah) dan 14 (porsi greedy) dilewati")

    txt = "\n".join(summary(d, lay, W, a.edge)) + "\n"
    (outdir / f"ringkasan{sfx}.txt").write_text(txt, encoding="utf-8")
    print("ditulis:", outdir / f"ringkasan{sfx}.txt")
    if a.gif:
        qa.make_gif(lay, a.episode, outdir / f"11_animasi_jalur_evaluasi{sfx}.gif")


if __name__ == "__main__":
    main()
