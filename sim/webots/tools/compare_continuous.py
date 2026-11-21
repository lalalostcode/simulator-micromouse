"""Buat diagram perbandingan Q-learning dan DQN dari data eksperimen."""

import argparse
import csv
import json
from pathlib import Path
import sys

from run_continuous_experiments import ALGORITHMS, CONTROLLERS, MODES

sys.path.insert(0, str(CONTROLLERS / "continuous_qlearning"))
import analisis as qa  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

BLOCK_SIZE = 250
NAMES = {"qlearning": "Q-learning tabular", "dqn": "DQN murni"}
OUTCOME_NAMES = {"goal": "Sampai target", "collision": "Tabrakan", "stuck": "Macet", "timeout": "Waktu habis"}
MODE_NAMES = {"decay": "Decay", "constant": "Konstan 0,10", "route": "Waktu rute"}


def block_counts(data, block_count):
    index = (data["ep"] - 1) // BLOCK_SIZE
    if not qa.np.isin(data["out"], qa.OUTCOMES).all() or not qa.np.all(qa.np.diff(data["ep"]) > 0):
        raise ValueError("Episode harus urut dan hasilnya harus valid.")
    return qa.np.array([qa.np.bincount(index[data["out"] == name], minlength=block_count)
                        for name in qa.OUTCOMES])


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    qa.plt.close(fig)


def composition(data, out):
    size = max(int((values["ep"][-1] - 1) // BLOCK_SIZE + 1) for values in data.values())
    positions = qa.np.arange(size)
    fig, axes = qa.plt.subplots(2, 1, sharex=True, figsize=(9.4, 5.0), layout="constrained")
    for ax, (algorithm, values) in zip(axes, data.items()):
        counts = block_counts(values, size)
        totals = counts.sum(axis=0)
        occupied = totals > 0
        bottom = qa.np.zeros(occupied.sum())
        for kind, row in zip(qa.OUTCOMES, counts):
            fraction = row[occupied] / totals[occupied]
            ax.bar(positions[occupied], fraction, bottom=bottom, width=.82, color=qa.COL[kind],
                   label=OUTCOME_NAMES[kind])
            bottom += fraction
        for i in positions[occupied & (totals < BLOCK_SIZE)]:
            ax.text(i, 1.018, f"n={totals[i]}", ha="center", va="bottom", fontsize=9)
        ax.set_title(NAMES[algorithm], loc="left", fontsize=12, fontweight="bold")
        ax.set(ylim=(0, 1.13), ylabel="Proporsi episode", yticks=[0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.24), ncols=4, frameon=False)
    labels = [f"{i * BLOCK_SIZE + 1}–{(i + 1) * BLOCK_SIZE}" for i in positions]
    axes[-1].set_xticks(positions, labels, fontsize=9)
    axes[-1].set_xlabel("Nomor episode dalam blok")
    save(fig, out, "komposisi_hasil_per_blok")


def heatmaps(data, layouts, out):
    surfaces = {alg: qa.collision_density(data[alg], layouts[alg]) for alg in ALGORITHMS}
    vmax = max(1.0, *(density.max() for density, _ in surfaces.values()))
    fig, axes = qa.plt.subplots(1, 2, figsize=(9.4, 4.5), layout="constrained")
    image = None
    for ax, (algorithm, (density, count)) in zip(axes, surfaces.items()):
        drawn = qa.panel_collision_density(ax, layouts[algorithm], density, count, NAMES[algorithm], vmax)
        if drawn is not None:
            image = drawn
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=.82, label="Tabrakan per m²")
    save(fig, out, "kepadatan_tabrakan")


def transfer_paths(runs, out):
    variants = {"original": "Arena asli", "complex": "Arena kompleks"}
    files = [(alg, variant, runs / "transfer" / alg / variant / "eval_result.json")
             for alg in ALGORITHMS for variant in variants]
    if not all(path.is_file() for _, _, path in files):
        return
    fig, axes = qa.plt.subplots(2, 2, figsize=(8.7, 7.6), layout="constrained")
    trials = []
    for ax, (algorithm, variant, path) in zip(axes.flat, files):
        data = json.loads(path.read_text(encoding="utf-8"))
        qa.panel_paths(ax, data)
        ax.set_title(f"{NAMES[algorithm]} · {variants[variant]}\n{data['success']}/{data['n']} sampai target")
        trials.extend({"algorithm": algorithm, "arena": variant, "checkpoint": data["checkpoint"]["episode"],
                       **episode} for episode in data["episodes"])
    save(fig, out, "lintasan_transfer")
    columns = ("algorithm", "arena", "checkpoint", "heading", "repeat", "outcome", "decisions", "sim_time", "path_m",
               "hazard_steps", "reward")
    with (out / "evaluasi_transfer.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)


def training_curves(runs, out):
    grouped = {alg: {mode: runs / "train" / alg / mode for mode in MODES} for alg in ALGORITHMS}
    completed = {alg: folders for alg, folders in grouped.items()
                 if all((folder / "eval_result.json").is_file() for folder in folders.values())}
    if not completed:
        return
    fig, axes = qa.plt.subplots(len(completed), 1, figsize=(9.4, 2.5 * len(completed)), sharex=True,
                                squeeze=False, layout="constrained")
    for ax, (algorithm, folders) in zip(axes.flat, completed.items()):
        for mode, folder in folders.items():
            data = qa.load_log(folder / "episode_log.csv")
            ax.plot(data["ep"][49:], qa.moving(data["out"] == "goal", 50)[49:], label=MODE_NAMES[mode], lw=1.5)
        ax.set(title=NAMES[algorithm], ylim=(-.02, 1.02), ylabel="Sampai target")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    axes[0, 0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1, ncols=3)
    axes[-1, 0].set_xlabel("Episode")
    save(fig, out, "strategi_epsilon")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folders = {alg: CONTROLLERS / f"continuous_{alg}" for alg in ALGORITHMS}
    data = {key: qa.load_log(folder / "episode_log_best.csv") for key, folder in folders.items()}
    layouts = {key: json.loads((folder / "eval_result_best.json").read_text(encoding="utf-8"))
               for key, folder in folders.items()}
    composition(data, args.out)
    heatmaps(data, layouts, args.out)
    if args.runs:
        transfer_paths(args.runs, args.out)
        training_curves(args.runs, args.out)


if __name__ == "__main__":
    main()
