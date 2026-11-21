"""Jalankan uji checkpoint dan eksperimen epsilon Tugas 9 di Webots."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from make_continuous_world import WORLDS

ALGORITHMS = {
    "qlearning": ("micromouse_continuous", "qtable_state_best.pkl"),
    "dqn": ("micromouse_continuous_dqn", "dqn_model_best.pt"),
}
MODES = ("decay", "constant", "route")
CONTROLLERS = WORLDS.parent / "controllers"


def complete(folder, episode_count, epsilon_mode, checkpoint_hash):
    result_file = folder / "eval_result.json"
    if not result_file.is_file():
        return False
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if len(result["episodes"]) != 10 or result["epsilon"] != 0 or result["seed"] != 7:
        return False
    if episode_count is None:
        return result["checkpoint"]["sha256"] == checkpoint_hash
    best = folder / "eval_result_rate_tertinggi.json"
    selected = json.loads(best.read_text(encoding="utf-8")) if best.is_file() else result
    with (folder / "episode_log.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    train = selected["train"]
    return (len(rows) == episode_count == train["episodes"] and train["epsilon"]["mode"] == epsilon_mode
            and len(selected["episodes"]) == 10 and selected["checkpoint"]["episode"] == train["best_episode"])


def run_job(job, args, webots, root):
    number, (phase, algorithm, variant) = job
    stem, archive = ALGORITHMS[algorithm]
    model = CONTROLLERS / f"continuous_{algorithm}" / archive
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    is_transfer = phase == "transfer"
    episodes = None if is_transfer else (3 if phase == "smoke" else args.episodes)
    mode = "decay" if is_transfer else variant
    world = WORLDS / f"{stem}{'_complex' if variant == 'complex' else ''}.wbt"
    folder = root / phase / algorithm / variant
    name = f"{phase}/{algorithm}/{variant}"
    if complete(folder, episodes, mode, checksum):
        print(f"{name}: hasil lengkap tersedia", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "episode_log.csv").exists() or (folder / "eval_result.json").exists():
        raise ValueError(f"{name}: ada hasil belum lengkap di {folder}. Pilih direktori keluaran baru.")
    env = {key: value for key, value in os.environ.items() if not key.startswith("MM_")}
    env.update({"MM_SEED": "7", "MM_PURE": "1", "MM_HUD": "0", "MM_START_HEADING": "0,90",
                "MM_EARLY_STOP": "0", "MM_OUT": str(folder), "MM_EPS_MODE": mode, "MM_EPSILON": "0.10",
                "MM_EVAL_ONLY": str(int(is_transfer)), "MM_MODEL": str(model), "MM_WORLD": world.name,
                "MICROMOUSE_QUIT": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "PATH": os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", "")))})
    if episodes is not None:
        env["MM_EPISODES"] = str(episodes)
    cmd = [str(webots), "--batch", "--mode=fast", "--minimize", "--no-rendering", "--stdout", "--stderr",
           f"--port={args.port + number}", str(world)]
    print(f"{name}: mulai", flush=True)
    log = folder / "webots.log"
    with log.open("wb") as stream:
        proc = subprocess.run(cmd, cwd=webots.parent, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              timeout=240 if phase != "train" else None)
    if proc.returncode or not complete(folder, episodes, mode, checksum):
        raise RuntimeError(f"{name}: Webots selesai dengan kode {proc.returncode}. Periksa {log}")
    if hashlib.sha256(model.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"{name}: checksum checkpoint arsip berubah")
    print(f"{name}: selesai", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webots", required=True, type=Path, help="path executable Webots R2025a")
    parser.add_argument("--out", required=True, type=Path, help="direktori induk hasil eksperimen")
    parser.add_argument("--phase", choices=("smoke", "transfer", "train", "all"), default="all")
    parser.add_argument("--algorithm", choices=("all", *ALGORITHMS), default="all")
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--port", type=int, default=18000, help="nomor port Webots pertama")
    args = parser.parse_args()
    webots, root = args.webots.resolve(), args.out.resolve()
    if not webots.is_file() or args.episodes < 1 or args.jobs < 1:
        parser.error("Executable Webots harus tersedia, dan jumlah episode serta job harus positif.")
    algorithms = tuple(ALGORITHMS) if args.algorithm == "all" else (args.algorithm,)
    phases = ("smoke", "transfer", "train") if args.phase == "all" else (args.phase,)
    modes = MODES if args.mode == "all" else (args.mode,)
    for phase in phases:
        variants = ("original", "complex") if phase == "transfer" else modes
        jobs = list(enumerate((phase, algorithm, variant) for algorithm in algorithms for variant in variants))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_job, job, args, webots, root) for job in jobs]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
