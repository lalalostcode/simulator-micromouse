"""Q-learning tabular murni: robot menemukan dan mencapai target di arena kontinu berisi rintangan.

Arena dan posisi start dibaca dari world. Arah awal training dapat diacak.

Dunia fisik kontinu (kecepatan roda riil, fisika Webots 16 ms), tetapi kunci
Q-table adalah tuple bilangan bulat hasil bin dari sensor (lihat arena_env.py).
Mode bawaan (arena_env.TARGET_KNOWN = False): robot TIDAK tahu letak target dan
area bahaya; state = (IR depan-kiri, IR depan-kanan, IR diagonal kiri, IR diagonal
kanan, sel posisi x, sel posisi y, arah hadap), imbalan menuju target hanya +100
saat masuk area target, plus bonus kecil menjelajah sel baru. Dengan
TARGET_KNOWN = True state memuat arah/jarak ke target dan zona bahaya.
5 aksi (pasangan kecepatan roda).

Rumus update (sama dengan proyek micromouse):
  Q[s][a] += alpha * (r + gamma * max Q[s'] - Q[s][a])      (tanpa max di state akhir)

Training berhenti bila success rate WINDOW episode terakhir >= CONVERGED, atau
tidak membaik selama PATIENCE episode (dihitung hanya setelah epsilon mencapai
minimum), atau EPISODES habis. Q-table dengan
success rate WINDOW-episode tertinggi disimpan ke qtable_state.pkl (bukan
Q-table terakhir) dan itulah yang dievaluasi greedy (epsilon 0) dari beberapa arah
awal (EVAL_HEADINGS, tiap arah EVAL_REPEATS kali). Timeout diberi penalti
(R_TIMEOUT) dan dianggap akhir episode.

Variabel lingkungan (semua opsional):
  MM_EPISODES   batas episode            (6000; 3000 bila TARGET_KNOWN)
  MM_SEED       seed agen                (7)
  MM_PATIENCE   episode tanpa perbaikan  (800; 400 bila TARGET_KNOWN)
  MM_START_HEADING  "a,b": arah awal robot ACAK per episode training, seragam di
                antara a dan b derajat arah kompas (0 = utara, 90 = timur);
                "world" = pakai arah robot di world. Bawaan: "0,90"
                (START_HEADING_DEFAULT). Evaluasi memakai lima arah tetap, dua ulangan per arah.
  MM_EVAL_ONLY  1: muat checkpoint dan jalankan evaluasi greedy
  MM_MODEL, MM_OUT: path checkpoint dan folder keluaran
  MM_EPS_MODE, MM_EPSILON, MM_EARLY_STOP: jadwal eksplorasi dan anggaran training
  MICROMOUSE_QUIT 1: tutup Webots setelah selesai
Keluaran di folder ini: episode_log.csv, qtable_state.pkl, eval_result.json (berisi juga peta untuk analisis.py).
"""
import csv
import math
import os
import pickle
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from arena_env import ACTIONS, ACTION_NAMES, DECISION_STEPS, DT, SETTLE_STEPS, TARGET_KNOWN, TRACK, WHEEL_R, ArenaEnv, discretize
from experiments import EpsilonSchedule, checkpoint_metadata, evaluate_policy, output_dir

HERE = Path(__file__).resolve().parent
ALPHA, GAMMA = 0.3, 0.95
# Mode tidak tahu target butuh eksplorasi lebih lama (imbalan jarang): epsilon turun lebih pelan.
EPS_START, EPS_DECAY, EPS_MIN = 1.0, (0.997 if TARGET_KNOWN else 0.9985), 0.05
WINDOW, CONVERGED = 50, 0.95

EPISODES = int(os.environ.get("MM_EPISODES", "3000" if TARGET_KNOWN else "6000"))
SEED = int(os.environ.get("MM_SEED", "7"))
PATIENCE = int(os.environ.get("MM_PATIENCE", "400" if TARGET_KNOWN else "800"))
EVAL_ONLY = os.environ.get("MM_EVAL_ONLY") == "1"
# Arah awal robot ACAK tiap episode training, seragam antara dua sudut (derajat kompas:
# 0 = utara, 90 = timur). Ubah angka di bawah ini bila menjalankan dari aplikasi Webots.
# None = pakai arah robot di world. Variabel lingkungan MM_START_HEADING menimpanya
# ("a,b", atau "world" untuk mematikan).
START_HEADING_DEFAULT = (0.0, 90.0)
_sh = os.environ.get("MM_START_HEADING", "").strip().lower()
if not _sh:
    START_HEADING = START_HEADING_DEFAULT
elif _sh in ("world", "off", "none"):
    START_HEADING = None
else:
    START_HEADING = tuple(sorted(float(v) for v in _sh.split(",")))


class QAgent:
    def __init__(self, seed):
        self.q = {}
        self.rng = random.Random(seed)

    def row(self, s):
        if s not in self.q:
            self.q[s] = np.zeros(len(ACTIONS))
        return self.q[s]

    def act(self, s, eps, *, learn=True):
        if self.rng.random() < eps:
            return self.rng.randrange(len(ACTIONS))
        r = self.row(s) if learn else self.q.get(s)
        if r is None:
            return self.rng.randrange(len(ACTIONS))
        best = np.flatnonzero(r == r.max())          # seri: acak, bukan selalu indeks 0
        return int(self.rng.choice(list(best)))

    def update(self, s, a, r, s2, done):
        target = r if done else r + GAMMA * self.row(s2).max()
        self.row(s)[a] += ALPHA * (target - self.row(s)[a])


def run_episode(env, agent, eps, learn, record_path=False, start_yaw=None):
    env.record_stuck = learn
    s = discretize(env.reset(start_yaw))
    total = 0.0
    path = [env.true_pose()[:2]] if record_path else []
    while True:
        a = agent.act(s, eps, learn=learn)
        raw2, r, outcome = env.step(a)
        s2 = discretize(raw2)
        if learn:
            agent.update(s, a, r, s2, outcome is not None)
        total += r
        if record_path:
            x, y, _ = env.true_pose()
            path.append((x, y))
        s = s2
        if outcome:
            return outcome, env.n_decisions, total, path


class QTableUnpickler(pickle.Unpickler):
    """Checkpoint Q-table menyimpan tuple, daftar, dan angka."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Tipe checkpoint tidak didukung: {module}.{name}")


def evaluate(env, agent, dest, model, episode, train=None):
    evaluate_policy(env, agent, run_episode, dest, seed=SEED,
                    checkpoint=checkpoint_metadata(model, episode), settle_time=SETTLE_STEPS * DT,
                    fields={"train": train})


def main():
    t_start = time.perf_counter()
    env = ArenaEnv()
    out = output_dir(HERE)
    agent = QAgent(SEED)
    print(f"continuous_qlearning: seed={SEED} episodes<={EPISODES} layout=FIXED target_known={TARGET_KNOWN} start_heading={START_HEADING if START_HEADING else 'world'} "
          f"aksi={list(ACTION_NAMES)}")

    if EVAL_ONLY:
        model = Path(os.getenv("MM_MODEL") or HERE / "qtable_state_best.pkl")
        with model.open("rb") as f:
            checkpoint = QTableUnpickler(f).load()
        agent.q = {key: np.asarray(values) for key, values in checkpoint["Q"].items()}
        if any(len(row) != len(ACTIONS) for row in agent.q.values()):
            raise ValueError("Jumlah aksi checkpoint Q-table berbeda dari controller.")
        evaluate(env, agent, out / "eval_result.json", model, checkpoint.get("episode"))
        return env

    if EPISODES < 1:
        raise ValueError("MM_EPISODES harus positif.")
    schedule = EpsilonSchedule(env, ACTIONS, WHEEL_R, TRACK, DECISION_STEPS * DT, (EPS_START, EPS_DECAY, EPS_MIN))
    early_stop = os.getenv("MM_EARLY_STOP", "1") == "1"
    model = out / "qtable_state.pkl"
    log = (out / "episode_log.csv").open("w", newline="", encoding="utf-8")
    w = csv.writer(log)
    w.writerow(["episode", "epsilon", "decisions", "reward", "outcome", "hazard_steps", "sim_time", "states", "start_heading", "end_x", "end_y", "route_seconds", "route_decisions"])
    heading_rng = random.Random(SEED + 1)     # terpisah dari rng agen
    window = deque(maxlen=WINDOW)
    best_rate, since_best, stop, rate = -1.0, 0, "max_episodes", 0.0
    best_q, best_ep = None, 0
    for ep in range(1, EPISODES + 1):
        start_yaw, heading = None, 90.0 - math.degrees(env.layout["yaw"])    # arah kompas
        if START_HEADING:
            heading = heading_rng.uniform(*START_HEADING)
            start_yaw = math.radians(90.0 - heading)
        eps, route_time, route_steps = schedule.for_episode(ep, math.radians(90 - heading))
        outcome, steps, total, _ = run_episode(env, agent, eps, True, start_yaw=start_yaw)
        end_x, end_y, _ = env.true_pose()          # posisi robot saat episode berakhir
        window.append(outcome == "goal")
        rate = sum(window) / len(window)
        w.writerow([ep, f"{eps:.9f}", steps, f"{total:.2f}", outcome, env.hazard_steps,
                    f"{env.sim_time():.1f}", len(agent.q), f"{heading:.1f}", f"{end_x:.3f}", f"{end_y:.3f}", route_time, route_steps])
        log.flush()
        print(f"EP {ep:4d} eps={eps:.2f} {outcome:9s} decisions={steps:3d} "
              f"R={total:7.1f} hazard={env.hazard_steps:3d} rate{len(window)}={rate:.2f} states={len(agent.q)}")
        if len(window) == WINDOW:
            if rate > best_rate:
                best_rate, since_best = rate, 0
                best_q, best_ep = {k: v.copy() for k, v in agent.q.items()}, ep
                with model.open("wb") as f:
                    pickle.dump({"Q": {k: v.tolist() for k, v in best_q.items()},
                                 "rate": rate, "episode": ep}, f)
            elif schedule.mode != "decay" or max(EPS_MIN, eps * EPS_DECAY) <= EPS_MIN + 1e-12:
                # Patience baru dihitung setelah epsilon mencapai minimum: selama masih
                # eksplorasi, success rate yang belum naik belum berarti training macet.
                since_best += 1
            if early_stop and rate >= CONVERGED:
                stop = "converged"
                break
            if early_stop and since_best >= PATIENCE:
                stop = "patience"
                break
    log.close()
    print(f"TRAINING selesai: {stop}, episode={ep}, rate_akhir={rate:.2f}, states={len(agent.q)}")
    if best_q is not None:
        agent.q = best_q
        print(f"Q-table terbaik: episode {best_ep}, success rate {best_rate:.2f} (disimpan di qtable_state.pkl)")
    else:      # belum pernah ada jendela penuh (training sangat singkat): simpan yang terakhir
        best_ep = ep
        with model.open("wb") as f:
            pickle.dump({"Q": {k: v.tolist() for k, v in agent.q.items()},
                         "rate": rate, "episode": ep}, f)
    evaluate(env, agent, out / "eval_result.json", model, best_ep,
             {"stop": stop, "episodes": ep, "best_episode": best_ep, "epsilon": schedule.metadata(),
              "wall_seconds": round(time.perf_counter() - t_start)})
    return env


if __name__ == "__main__":
    env = main()
    if os.environ.get("MICROMOUSE_QUIT") == "1":
        env.robot.simulationQuit(0)
    else:
        while env.robot.step(16) != -1:    # robot diam, simulasi tetap hidup
            pass
