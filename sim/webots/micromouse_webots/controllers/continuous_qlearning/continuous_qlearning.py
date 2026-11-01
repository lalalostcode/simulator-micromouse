"""Q-learning tabular murni: robot menemukan dan mencapai target di arena kontinu berisi rintangan.

Tata letak arena TETAP (target dan 5 rintangan dipasang di world, lihat
tools/make_continuous_world.py); robot selalu start dari pose yang sama.

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
                (START_HEADING_DEFAULT). Evaluasi akhir selalu memakai arah robot di world.
  MM_EVAL_ONLY  1: tanpa training, muat qtable_state_best.pkl lalu evaluasi
  MICROMOUSE_QUIT 1: tutup Webots setelah selesai
Keluaran di folder ini: episode_log.csv, qtable_state.pkl, eval_result.json.
"""
import csv
import json
import math
import os
import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np

from arena_env import ACTIONS, ACTION_NAMES, TARGET_KNOWN, ArenaEnv, discretize

HERE = Path(__file__).resolve().parent
ALPHA, GAMMA = 0.3, 0.95
# Mode tidak tahu target butuh eksplorasi lebih lama (imbalan jarang): epsilon turun lebih pelan.
EPS_START, EPS_DECAY, EPS_MIN = 1.0, (0.997 if TARGET_KNOWN else 0.9985), 0.05
WINDOW, CONVERGED = 50, 0.95
# Evaluasi akhir: arah awal (derajat kompas, 0 = utara, 90 = timur) dan ulangan per arah.
EVAL_HEADINGS = (0.0, 22.5, 45.0, 67.5, 90.0)
EVAL_REPEATS = 2

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

    def act(self, s, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(len(ACTIONS))
        r = self.row(s)
        best = np.flatnonzero(r == r.max())          # seri: acak, bukan selalu indeks 0
        return int(self.rng.choice(list(best)))

    def update(self, s, a, r, s2, done):
        target = r if done else r + GAMMA * self.row(s2).max()
        self.row(s)[a] += ALPHA * (target - self.row(s)[a])


def run_episode(env, agent, eps, learn, record_path=False, start_yaw=None):
    s = discretize(env.reset(start_yaw))
    total, path = 0.0, []
    while True:
        a = agent.act(s, eps)
        raw2, r, outcome = env.step(a)
        s2 = discretize(raw2)
        if learn:
            agent.update(s, a, r, s2, outcome is not None)
        total += r
        if record_path:
            x, y, _ = env.true_pose()
            path.append((round(x, 4), round(y, 4)))
        s = s2
        if outcome:
            return outcome, env.n_decisions, total, path


def evaluate(env, agent):
    """Evaluasi greedy (epsilon 0) dari beberapa arah awal, EVAL_REPEATS kali tiap arah."""
    results = []
    k = 0
    n = len(EVAL_HEADINGS) * EVAL_REPEATS
    for h in EVAL_HEADINGS:
        for _ in range(EVAL_REPEATS):
            k += 1
            outcome, steps, total, path = run_episode(
                env, agent, 0.0, False, True, start_yaw=math.radians(90.0 - h))
            results.append({"heading": h, "outcome": outcome, "decisions": steps,
                            "reward": round(total, 2), "sim_time": round(steps * 0.128, 1),
                            "hazard_steps": env.hazard_steps, "path": path})
            print(f"EVAL {k}/{n}: arah={h:5.1f} {outcome:9s} decisions={steps} R={total:.1f} "
                  f"hazard_steps={env.hazard_steps}")
    ok = sum(r["outcome"] == "goal" for r in results)
    by_heading = {}
    for r in results:
        by_heading.setdefault(str(r["heading"]), []).append(r["outcome"])
    print("EVAL ringkasan per arah awal (kompas, 0 = utara, 90 = timur):")
    for h, outs in by_heading.items():
        print(f"  {float(h):5.1f} deg: {sum(o == 'goal' for o in outs)}/{len(outs)} goal  {outs}")
    print(f"EVAL success: {ok}/{n}")
    lay = env.layout
    (HERE / "eval_result.json").write_text(json.dumps({
        "success": ok, "n": n, "arena": env.arena, "target": list(lay["target"]), "start": list(env.start),
        "target_r": env.target_r, "obstacles": lay["obstacles"], "hazards": lay["hazards"],
        "headings": list(EVAL_HEADINGS), "repeats": EVAL_REPEATS, "by_heading": by_heading,
        "episodes": results}), encoding="utf-8")


def main():
    env = ArenaEnv()
    agent = QAgent(SEED)
    # Salinan tata letak yang dipakai run ini, untuk plot_endpoints.py (tidak bergantung eval_result.json).
    lay = env.layout
    (HERE / "layout.json").write_text(json.dumps({
        "arena": env.arena, "start": list(env.start), "target": list(lay["target"]),
        "target_r": env.target_r, "obstacles": lay["obstacles"], "hazards": lay["hazards"]}), encoding="utf-8")
    print(f"continuous_qlearning: seed={SEED} episodes<={EPISODES} layout=FIXED target_known={TARGET_KNOWN} start_heading={START_HEADING if START_HEADING else 'world'} "
          f"aksi={list(ACTION_NAMES)}")

    if EVAL_ONLY:
        with open(HERE / "qtable_state_best.pkl", "rb") as f:
            agent.q = {k: np.array(v) for k, v in pickle.load(f)["Q"].items()}
        evaluate(env, agent)
        return env

    log = open(HERE / "episode_log.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(log)
    w.writerow(["episode", "epsilon", "decisions", "reward", "outcome", "hazard_steps", "sim_time", "states", "start_heading", "end_x", "end_y"])
    eps = EPS_START
    heading_rng = random.Random(SEED + 1)     # terpisah dari rng agen
    window = deque(maxlen=WINDOW)
    best_rate, since_best, stop, rate = -1.0, 0, "max_episodes", 0.0
    best_q, best_ep = None, 0
    for ep in range(1, EPISODES + 1):
        start_yaw, heading = None, 90.0 - math.degrees(env.layout["yaw"])    # arah kompas
        if START_HEADING:
            heading = heading_rng.uniform(*START_HEADING)
            start_yaw = math.radians(90.0 - heading)
        outcome, steps, total, _ = run_episode(env, agent, eps, True, start_yaw=start_yaw)
        end_x, end_y, _ = env.true_pose()          # posisi robot saat episode berakhir
        window.append(outcome == "goal")
        rate = sum(window) / len(window)
        w.writerow([ep, f"{eps:.4f}", steps, f"{total:.2f}", outcome, env.hazard_steps,
                    f"{env.sim_time():.1f}", len(agent.q), f"{heading:.1f}", f"{end_x:.3f}", f"{end_y:.3f}"])
        log.flush()
        print(f"EP {ep:4d} eps={eps:.2f} {outcome:9s} decisions={steps:3d} "
              f"R={total:7.1f} hazard={env.hazard_steps:3d} rate{len(window)}={rate:.2f} states={len(agent.q)}")
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if len(window) == WINDOW:
            if rate > best_rate:
                best_rate, since_best = rate, 0
                best_q, best_ep = {k: v.copy() for k, v in agent.q.items()}, ep
                with open(HERE / "qtable_state.pkl", "wb") as f:
                    pickle.dump({"Q": {k: v.tolist() for k, v in best_q.items()},
                                 "rate": rate, "episode": ep}, f)
            elif eps <= EPS_MIN + 1e-12:
                # Patience baru dihitung setelah epsilon mencapai minimum: selama masih
                # eksplorasi, success rate yang belum naik belum berarti training macet.
                since_best += 1
            if rate >= CONVERGED:
                stop = "converged"
                break
            if since_best >= PATIENCE:
                stop = "patience"
                break
    log.close()
    print(f"TRAINING selesai: {stop}, episode={ep}, rate_akhir={rate:.2f}, states={len(agent.q)}")
    if best_q is not None:
        agent.q = best_q
        print(f"Q-table terbaik: episode {best_ep}, success rate {best_rate:.2f} (disimpan di qtable_state.pkl)")
    else:      # belum pernah ada jendela penuh (training sangat singkat): simpan yang terakhir
        with open(HERE / "qtable_state.pkl", "wb") as f:
            pickle.dump({"Q": {k: v.tolist() for k, v in agent.q.items()},
                         "rate": rate, "episode": ep}, f)
    evaluate(env, agent)
    return env


if __name__ == "__main__":
    env = main()
    if os.environ.get("MICROMOUSE_QUIT") == "1":
        env.robot.simulationQuit(0)
    else:
        while env.robot.step(16) != -1:    # robot diam, simulasi tetap hidup
            pass
