"""DQN: robot menemukan dan mencapai target di arena kontinu (world yang sama dengan Q-learning).

Beda dengan continuous_qlearning: state TIDAK di-bin. Jaringan menerima 13 angka riil
(4 IR berskala akar kuadrat, posisi x/y odometri, sin dan cos arah hadap, waktu terpakai, 4 selisih IR
terhadap keputusan sebelumnya) dan mengeluarkan satu nilai Q untuk tiap aksi diskret. Lingkungan, reward, deteksi macet/tabrakan, dan evaluasi sama dengan
Q-learning (arena_env.py dipakai bersama dari folder continuous_qlearning).

10 aksi (pasangan kecepatan roda, rad/s), lihat ACTIONS di bawah. Aksi 0-7 tanpa roda mundur dan dipilih
DQN; aksi 8 dan 9 (putar di tempat, satu roda mundur) HANYA dijalankan lapisan pengaman untuk menghindar.
Robot tidak tahu letak target (arena_env.TARGET_KNOWN = False).
Pose awal: posisi TETAP (0,18 ; 0,18) m; arah hadap ACAK seragam 0-90 derajat kompas tiap episode training
(0 = utara, 90 = timur), sama dengan Q-learning. Evaluasi dari 5 arah: 0, 22,5, 45, 67,5, 90.

Training berhenti (sama dengan Q-learning) bila success rate WINDOW episode terakhir >= CONVERGED, atau
tidak membaik selama PATIENCE episode (dihitung setelah epsilon minimum), atau EPISODES habis.
EARLY_STOP=False mematikan dua syarat pertama; STOP_AT_EPS_MIN=True berhenti begitu epsilon minimum. Bobot dengan
success rate tertinggi disimpan ke dqn_model.pt (pembanding); evaluasi greedy (epsilon 0) memakai
JARINGAN TERAKHIR (dqn_model_last.pt), bukan yang rate-nya tertinggi.

Variabel lingkungan (opsional, hanya bila Webots dijalankan dari terminal):
  MM_EPISODES, MM_SEED, MM_PATIENCE, MM_EVAL_ONLY=1 (muat dqn_model_best.pt), MICROMOUSE_QUIT=1
Keluaran di folder ini: episode_log.csv, dqn_model.pt, eval_result.json, stuck_hotspots.json (titik macet dan titik tabrakan beserta jumlahnya).
Catatan: kolom "states" di episode_log.csv berisi jumlah transisi di replay buffer; "greedy" =
jumlah keputusan greedy (bukan acak) yang tidak diganti pengaman; aksi_0..aksi_9 = berapa kali tiap
aksi DIJALANKAN; "pengaman" = berapa kali lapisan pengaman (SAFETY_LAYER) mengganti aksi pilihan DQN; "pemecah" =
berapa kali pemecah loop (ESCAPE_LAYER) memulai manuver keluar dari jebakan.
"""
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "continuous_qlearning"))
import arena_env                                  # noqa: E402
from arena_env import ArenaEnv, TARGET_KNOWN      # noqa: E402
from dqn_agent import DQNAgent                    # noqa: E402

# Daftar aksi DQN menimpa daftar 5 aksi milik Q-learning (arena_env membacanya saat step()).
ACTIONS = (
    (10.0, 10.0),    # 0 lurus cepat
    (5.0, 5.0),      # 1 lurus pelan
    (5.0, 10.0),     # 2 busur kiri landai
    (10.0, 5.0),     # 3 busur kanan landai
    (0.0, 10.0),     # 4 belok kiri tajam
    (10.0, 0.0),     # 5 belok kanan tajam
    (0.0, 5.0),      # 6 belok kiri tajam, pelan
    (5.0, 0.0),      # 7 belok kanan tajam, pelan
    (-5.0, 5.0),     # 8 putar kiri di tempat (roda kiri mundur, kanan maju; ~14 derajat per keputusan)
    (5.0, -5.0),     # 9 putar kanan di tempat
)
ACTION_NAMES = ("lurus-cepat", "lurus-pelan", "kiri-landai", "kanan-landai", "kiri-tajam",
                "kanan-tajam", "kiri-tajam-pelan", "kanan-tajam-pelan", "putar-kiri",
                "putar-kanan")
arena_env.ACTIONS = ACTIONS
arena_env.ACTION_NAMES = ACTION_NAMES
# REWARD = SAMA DENGAN Q-LEARNING (apple to apple): goal +100, tabrakan -100, macet -30 (episode berakhir),
# timeout -20, -0,05 per keputusan, putar di tempat -0,10, area bahaya -3 per keputusan, bonus sel baru
# +0,3..+1,5. Tidak ada bonus waktu, penalti tabrakan dekat start, penalti yang membesar, atau penalti zona.
arena_env.SPIN_ACTIONS = (8, 9)                  # hanya putar di tempat yang kena R_SPIN (aksi 4-7 = busur/pivot, seperti 'maju-kiri/kanan' Q-learning)
R_STEP_DQN = -0.05
arena_env.R_STEP = R_STEP_DQN
R_GOAL_TIME_DQN = 0.0                            # bonus waktu goal dimatikan (Q-learning: imbalan goal tetap +100)
GOAL_TIME_REF_DQN = 400                          # patokan bonus waktu (tidak dipakai selama R_GOAL_TIME_DQN = 0)
arena_env.R_GOAL_TIME = R_GOAL_TIME_DQN
arena_env.GOAL_TIME_REF = GOAL_TIME_REF_DQN
R_COLLISION_NEAR_DQN = 0.0                       # penalti tabrakan tetap -100, tidak bergantung jarak ke start
arena_env.R_COLLISION_NEAR = R_COLLISION_NEAR_DQN
R_STUCK_DQN = -30.0
arena_env.R_STUCK = R_STUCK_DQN
arena_env.GROW_RATE = 1.0
arena_env.STUCK_GROW, arena_env.STUCK_GROW_CAP = 0.0, 100.0         # tanpa penalti yang membesar
arena_env.ZONE_STEP, arena_env.ZONE_CAP = 0.0, 1.0                  # tanpa penalti zona
arena_env.COLL_GROW, arena_env.COLL_GROW_CAP = 0.0, 50.0
arena_env.COLL_ZONE_STEP, arena_env.COLL_ZONE_CAP = 0.0, 1.0

N_FEATURES = 13
# MODE DQN MURNI (bawaan, untuk perbandingan apple to apple dengan Q-learning): tanpa pengaman, ingatan sisi, pemecah
# macet, dan pemutus orbit; DQN memilih SEMUA 10 aksi (termasuk putar di tempat, seperti Q-learning); fitur waktu
# terpakai dan selisih IR (yang tidak ada di state Q-learning) dimatikan (diisi 0). False = DQN + lapisan buatan.
# Dari terminal: MM_PURE=0 atau 1; dari GUI Webots ubah nilai bawaan di sini.
PURE_DQN = os.environ.get("MM_PURE", "1") == "1"
USE_TIME_FEATURE = not PURE_DQN
USE_IR_DELTA = not PURE_DQN
N_POLICY_ACTIONS = 10 if PURE_DQN else 8    # mode lapisan: aksi 0-7 dipilih DQN; 8 dan 9 (putar di tempat) hanya dari lapisan pengaman
N_STEP = int(os.environ.get("MM_NSTEP", "20"))             # target 20-langkah (~0,37 m): memutus penguatan bias maks-Q (hasil uji offline: TD 1-langkah jenuh ke batas atas)
GAMMA = float(os.environ.get("MM_GAMMA", "0.995"))          # horizon efektif ~200 keputusan (sebelumnya 0,99: ~100); bobot goal pada 220 keputusan 0,33 (sebelumnya 0,11)
EPS_START, EPS_DECAY, EPS_MIN = 1.0, 0.9985, 0.05     # epsilon mencapai 0,05 sekitar episode 2.000 (sama dengan Q-learning)
WINDOW, CONVERGED = 50, 0.95
EARLY_STOP = True           # sama dengan Q-learning: berhenti bila rate >= CONVERGED, atau patience habis (dihitung setelah epsilon minimum)
STOP_AT_EPS_MIN = False     # True = berhenti begitu epsilon mencapai EPS_MIN (tanpa menunggu konvergen)
EVAL_HEADINGS = (0.0, 22.5, 45.0, 67.5, 90.0)   # arah awal evaluasi (kompas: 0 = utara, 90 = timur), sama dengan Q-learning
EVAL_REPEATS = 2            # sama dengan Q-learning (greedy + fisika deterministik: ulangan umumnya identik)
IR_SCALE = 1200.0       # input IR = sqrt(ADC / IR_SCALE), dipotong ke [0, 1] (ADC >= 1200 ~ < 50 mm = 1,0)
IR_DELTA_GAIN = 3.0     # pengali selisih IR antar keputusan (skala sebanding dengan input lain)

# Lapisan pengaman (refleks hard-coded) di atas kebijakan DQN. False = DQN murni.
SAFETY_LAYER = not PURE_DQN
SH_FRONT = 100.0        # ADC FL/FR: rintangan di depan terdeteksi (lebih peka dari 150: bereaksi lebih awal)
SH_DIAG = 15.0          # ADC DL/DR (sinar 45 derajat, jangkauan sensor 25 cm): 15 -> sinar ~23,6 cm -> ~13,6 cm dari sisi badan
                        # (15 cm tepat tidak mungkin: sinar harus > 25 cm). Dulu 80 = ~10 cm dari sisi badan.
SH_DIAG_MEM = 80.0      # ambang ingatan sisi (tidak ikut dilonggarkan: ingatan hanya untuk rintangan yang benar-benar dekat)
A_KANAN_LANDAI, A_KIRI_LANDAI = 3, 2
A_LURUS_PELAN = 1
SH_BALANCE = 0.2        # DL dan DR dianggap seimbang bila selisihnya <= 20% dari yang lebih besar
A_KANAN_TAJAM, A_KIRI_TAJAM = 5, 4
A_PUTAR_KIRI, A_PUTAR_KANAN = 8, 9
# Pemecah loop (memori): terjebak = perpindahan odometri < ESC_DIST dalam ESC_WINDOW keputusan terakhir.
# Manuver: putar di tempat >= ESC_TURN_DEG, lalu maju lurus ESC_FORWARD keputusan (berhenti bila FL/FR terbaca).
# ESC_WINDOW harus < STUCK_WINDOW (20) milik lingkungan, kalau tidak episode sudah berakhir 'stuck' lebih dulu.
ESCAPE_LAYER = not PURE_DQN
ESC_WINDOW, ESC_DIST = 12, 0.05
ESC_TURN_DEG, ESC_TURN_MAX, ESC_FORWARD = 150.0, 20, 12
SH_CORNER_MIN = 3       # sudut cekung: minimal 3 dari 4 sensor IR >= ambang (4 = harus keempatnya)
# Ingatan sisi: DL/DR hanya melihat serong ke depan, jadi sudut rintangan yang baru lewat ke samping tak terbaca
# lagi padahal badan masih di dekatnya. Bila DL (DR) membaca rintangan dalam SIDE_MEM keputusan terakhir
# (~11 cm lintasan), belok ke kiri (kanan) dilarang: diganti lurus, atau belok menjauh bila depan terblokir.
SIDE_MEMORY = not PURE_DQN
SIDE_MEM = 6
LEFT_TURNS, RIGHT_TURNS = (2, 4, 6), (3, 5, 7)      # aksi 8/9 (putar di tempat, sudut cekung) tidak dibatasi

# Pemutus orbit + zona terlarang sementara: robot berputar >= 1 kali (yaw bersih >= 360 derajat dalam
# LOOP_WINDOW keputusan, tanpa perpindahan berarti) = terjebak orbit/kantong buntu. Aksi: menelusuri jejak
# (jalur bersih tanpa lingkaran) mundur LOOP_BACK m, lalu zona di sekitar jebakan dilarang selama
# TABU_STEPS keputusan (robot ditolak masuk). Titik jebakan juga dicatat sebagai titik macet (penalti zona).
LOOP_LAYER = not PURE_DQN
LOOP_WINDOW, LOOP_YAW, LOOP_DISP = 150, 2 * math.pi, 0.6
LOOP_ERASE_R, LOOP_ERASE_LAG = 0.12, 20      # lingkaran dihapus dari jejak bila kembali ke titik >= LAG keputusan lalu
LOOP_BACK, LOOP_WP, LOOP_BACK_MAX = 0.8, 0.15, 100    # panjang telusur mundur (m), jarak antar titik tuju (m), maks keputusan
TABU_MIN, TABU_MAX, TABU_STEPS = 0.25, 0.45, 300      # radius zona terlarang (m), lama berlaku (keputusan)
# Saat TRAINING dengan epsilon <= LOOP_PEN_EPS (gerak acak awal sering memutar tanpa benar-benar mengorbit, jadi
# tidak dihukum): deteksi orbit memberi penalti R_LOOP (membesar bila terulang di area yang sama) dan seluruh
# jalur orbit (tiap LOOP_MARK keputusan) dicatat sebagai titik macet agar penalti zona menutupi sabuk orbit.
# Episode TIDAK diakhiri: robot tetap menelusur jejak mundur lalu zona terlarang berlaku.
R_LOOP, LOOP_GROW, LOOP_GROW_CAP, LOOP_MARK, LOOP_PEN_EPS = 0.0, 0.0, 100.0, 10, 0.5      # R_LOOP = 0: tanpa penalti (reward = Q-learning); titik orbit tetap dicatat

EPISODES = int(os.environ.get("MM_EPISODES", "6000"))
SEED = int(os.environ.get("MM_SEED", "7"))
PATIENCE = int(os.environ.get("MM_PATIENCE", "800"))
EVAL_ONLY = os.environ.get("MM_EVAL_ONLY") == "1"
OUT_DIR = Path(os.environ.get("MM_OUT") or HERE)      # folder keluaran (log, model, eval); bawaan folder ini
START_HEADING = (0.0, 90.0)       # arah awal training: acak seragam di antara dua sudut (derajat kompas); sama = tetap


def features(raw, arena, n_decisions, prev_ir=None):
    """Vektor 13 angka riil + IR saat ini (untuk selisih di langkah berikutnya):
      4 IR berskala akar kuadrat sqrt(ADC/IR_SCALE) dalam [0, 1]  (membesarkan sinyal di jarak jauh),
      posisi odometri (x, y), arah hadap (sin, cos), waktu terpakai n/MAX_DECISIONS,
      4 selisih IR terhadap keputusan sebelumnya x IR_DELTA_GAIN (tren mendekat; 0 di awal episode).
    Waktu terpakai ada di state karena imbalan goal dan timeout bergantung padanya."""
    fl, fr, dl, dr, _b, _d, _h, x, y, yaw = raw
    ir = [math.sqrt(min(max(v / IR_SCALE, 0.0), 1.0)) for v in (fl, fr, dl, dr)]
    delta = [0.0] * 4 if (prev_ir is None or not USE_IR_DELTA) else [(a - b) * IR_DELTA_GAIN for a, b in zip(ir, prev_ir)]
    t = min(n_decisions / arena_env.MAX_DECISIONS, 1.0) if USE_TIME_FEATURE else 0.0
    vec = np.array(ir + [x / arena, y / arena, math.sin(yaw), math.cos(yaw), t] + delta, dtype=np.float32)
    return vec, ir


def shield(raw):
    """Lapisan pengaman (refleks): kembalikan aksi menghindar bila IR mendeteksi rintangan, atau None.
    Aturan dasar sama dengan aturan reaktif terbaik hasil uji (ambang ADC 150 ~ 100-175 mm):
      FL/FR >= SH_FRONT : belok tajam menjauh (rintangan lebih kuat di kiri -> kanan tajam, dan sebaliknya)
      DL dan DR >= SH_DIAG : menuju sisi lebih lapang (landai), lurus pelan bila selisih <= SH_BALANCE;
      hanya DL : kanan landai;  hanya DR : kiri landai.
    Sudut cekung (minimal SH_CORNER_MIN dari 4 sensor membaca rintangan dekat): putar di tempat menjauhi
    sisi yang lebih dekat, supaya badan tidak bergeser ke dinding seperti pada belok tajam."""
    fl, fr, dl, dr = raw[0], raw[1], raw[2], raw[3]
    n_hit = (fl >= SH_FRONT) + (fr >= SH_FRONT) + (dl >= SH_DIAG) + (dr >= SH_DIAG)
    if n_hit >= SH_CORNER_MIN:
        return A_PUTAR_KANAN if (fl + dl) > (fr + dr) else A_PUTAR_KIRI
    if fl >= SH_FRONT or fr >= SH_FRONT:
        return A_KANAN_TAJAM if fl > fr else A_KIRI_TAJAM
    if dl >= SH_DIAG and dr >= SH_DIAG:
        # lorong sempit (kedua sisi dekat): menuju sisi yang lebih lapang, atau lurus pelan bila seimbang
        # (dulu selalu kiri tajam, yang mendorong robot ke dinding kiri)
        if abs(dl - dr) <= SH_BALANCE * max(dl, dr):
            return A_LURUS_PELAN
        return A_KANAN_LANDAI if dl > dr else A_KIRI_LANDAI
    if dl >= SH_DIAG:
        return A_KANAN_LANDAI
    if dr >= SH_DIAG:
        return A_KIRI_LANDAI
    return None


def side_filter(a, raw, mem_l, mem_r):
    """Ingatan sisi: larang belok ke sisi yang rintangannya baru saja terlihat DL/DR (mem_l/mem_r = ada
    pembacaan >= SH_DIAG_MEM dalam SIDE_MEM keputusan terakhir). Pengganti: lurus; bila depan terblokir
    (FL/FR >= SH_FRONT) belok tajam menjauh, atau putar di tempat bila kedua sisi terlarang."""
    bad_l, bad_r = mem_l and a in LEFT_TURNS, mem_r and a in RIGHT_TURNS
    if not (bad_l or bad_r):
        return a
    if raw[0] >= SH_FRONT or raw[1] >= SH_FRONT:
        if mem_l and mem_r:
            return A_PUTAR_KANAN if (raw[0] + raw[2]) > (raw[1] + raw[3]) else A_PUTAR_KIRI
        return A_KANAN_TAJAM if mem_l else A_KIRI_TAJAM
    return 0


def dump_hotspots(env):
    def fmt(spots):
        return [{"x": round(h[0], 3), "y": round(h[1], 3), "n": h[2]}
                for h in sorted(spots, key=lambda h: -h[2])]
    (OUT_DIR / "stuck_hotspots.json").write_text(json.dumps(
        {"stuck": fmt(env.hotspots), "tabrakan": fmt(env.crash_spots)}), encoding="utf-8")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def steer(raw, bearing):
    """Aksi menuju arah `bearing` (rad, kerangka odometri) dari yaw odometri: putar di tempat bila selisih besar."""
    e = _wrap(bearing - raw[9])
    if abs(e) > math.radians(60):
        return A_PUTAR_KIRI if e > 0 else A_PUTAR_KANAN
    if abs(e) > math.radians(20):
        return A_KIRI_TAJAM if e > 0 else A_KANAN_TAJAM
    if abs(e) > math.radians(8):
        return A_KIRI_LANDAI if e > 0 else A_KANAN_LANDAI
    return 0


class LoopGuard:
    """Deteksi orbit/putaran (yaw bersih >= LOOP_YAW dalam LOOP_WINDOW), telusur jejak mundur, zona terlarang."""

    def __init__(self, window=None, yaw=None):
        self.window = window or LOOP_WINDOW
        self.yaw_need = yaw or LOOP_YAW
        self.trail = np.zeros((1200, 2))      # jejak odometri tanpa lingkaran (loop-erased)
        self.n = 0
        self.win = deque(maxlen=self.window + 1)      # (x, y, yaw tak terlipat)
        self.yaw_u, self._last = None, 0.0
        self.mode = None                      # None | "balik"
        self.wps, self.steps = [], 0
        self.tabu = []                        # [cx, cy, radius, berlaku sampai keputusan ke-]

    def update(self, x, y, yaw):
        if self.yaw_u is None:
            self.yaw_u = yaw
        else:
            self.yaw_u += _wrap(yaw - self._last)      # yaw tak terlipat: akumulasi putaran bersih
        self._last = yaw
        self.win.append((x, y, self.yaw_u))
        p = np.array([x, y])
        if self.n > LOOP_ERASE_LAG:
            d = np.hypot(*(self.trail[:self.n - LOOP_ERASE_LAG] - p).T)
            hit = np.flatnonzero(d < LOOP_ERASE_R)
            if hit.size:
                self.n = int(hit[0]) + 1       # hapus lingkaran: kembali ke titik lama
                return
        if self.n < len(self.trail):
            self.trail[self.n] = p
            self.n += 1

    def detect(self):
        if len(self.win) <= self.window:
            return False
        a, b = self.win[0], self.win[-1]
        return abs(b[2] - a[2]) >= self.yaw_need and math.hypot(b[0] - a[0], b[1] - a[1]) < LOOP_DISP

    def trigger(self, n_dec):
        """Mulai telusur mundur; kembalikan pusat jebakan (x, y) odometri."""
        pts = np.array([(w[0], w[1]) for w in self.win])
        c = pts.mean(0)
        r = float(np.hypot(*(pts - c).T).max()) + 0.05
        self.tabu.append([float(c[0]), float(c[1]), min(max(r, TABU_MIN), TABU_MAX), n_dec + TABU_STEPS])
        # titik tuju: dari ujung jejak bersih mundur sejauh LOOP_BACK, tiap ~LOOP_WP
        wps, acc, wps_acc = [], 0.0, 0.0
        for i in range(self.n - 2, -1, -1):
            step = float(np.hypot(*(self.trail[i] - self.trail[i + 1])))
            acc += step
            wps_acc += step
            if wps_acc >= LOOP_WP:
                wps.append((float(self.trail[i][0]), float(self.trail[i][1])))
                wps_acc = 0.0
            if acc >= LOOP_BACK:
                break
        if not wps:
            wps.append((float(self.trail[0][0]), float(self.trail[0][1])))
        self.wps, self.steps, self.mode = wps, 0, "balik"
        self.win.clear()
        self.yaw_u = None
        return float(c[0]), float(c[1])

    def retrace_action(self, raw):
        self.steps += 1
        while self.wps and math.hypot(self.wps[0][0] - raw[7], self.wps[0][1] - raw[8]) < 0.12:
            self.wps.pop(0)
        if not self.wps or self.steps > LOOP_BACK_MAX:
            self.mode = None
            return None
        return steer(raw, math.atan2(self.wps[0][1] - raw[8], self.wps[0][0] - raw[7]))

    def tabu_action(self, raw, n_dec):
        self.tabu = [z for z in self.tabu if z[3] > n_dec]
        for cx, cy, r, _ in self.tabu:
            d = math.hypot(raw[7] - cx, raw[8] - cy)
            if d < r:
                away = math.atan2(raw[8] - cy, raw[7] - cx) if d > 1e-3 else raw[9] + math.pi
                return steer(raw, away)
        return None


def run_episode(env, agent, eps, learn, record_path=False, start_yaw=None):
    env.record_stuck = learn            # evaluasi tidak menambah hitungan titik macet
    raw = env.reset(start_yaw)
    s, ir = features(raw, env.arena, 0)
    total, path = 0.0, []
    agent.ep_counts, agent.ep_greedy, agent.ep_shield = [0] * len(ACTIONS), 0, 0     # statistik episode ini
    agent.ep_escape = 0
    q_sum, q_n = 0.0, 0
    odo = []                            # riwayat posisi odometri untuk pemecah loop
    esc = None                          # None | ["putar", yaw0, aksi_putar, langkah] | ["maju", sisa]
    side_mem = deque(maxlen=SIDE_MEM)   # (DL dekat, DR dekat) tiap keputusan terakhir
    agent.ep_loop = 0
    lg = LoopGuard()
    lg.update(raw[7], raw[8], raw[9])
    while True:
        side_mem.append((raw[2] >= SH_DIAG_MEM, raw[3] >= SH_DIAG_MEM))
        # --- pemutus orbit: berputar >= 1 kali -> telusur jejak mundur + zona terlarang sementara ---
        # --- pemecah loop: terjebak (hampir tidak berpindah) -> putar balik lalu maju keluar ---
        if ESCAPE_LAYER and esc is None and len(odo) > ESC_WINDOW and math.hypot(
                odo[-1][0] - odo[-1 - ESC_WINDOW][0], odo[-1][1] - odo[-1 - ESC_WINDOW][1]) < ESC_DIST:
            left_side, right_side = raw[0] + raw[2], raw[1] + raw[3]
            esc = ["putar", raw[9], A_PUTAR_KANAN if left_side > right_side else A_PUTAR_KIRI, 0]
            agent.ep_escape += 1
            odo = []                    # hitung ulang deteksi terjebak setelah manuver
            env._pos_hist = []          # dan mulai ulang jendela deteksi stuck milik lingkungan
        a_esc = None
        if esc is not None:
            if esc[0] == "putar":
                turned = abs((raw[9] - esc[1] + math.pi) % (2 * math.pi) - math.pi)
                if turned >= math.radians(ESC_TURN_DEG) or esc[3] >= ESC_TURN_MAX:
                    esc = ["maju", ESC_FORWARD]
                else:
                    esc[3] += 1
                    a_esc = esc[2]
            if esc is not None and esc[0] == "maju":
                if esc[1] <= 0 or raw[0] >= SH_FRONT or raw[1] >= SH_FRONT:
                    esc = None
                else:
                    esc[1] -= 1
                    a_esc = 0
        a_back = lg.retrace_action(raw) if (LOOP_LAYER and lg.mode == "balik" and a_esc is None) else None
        if a_esc is not None:
            a = a_esc                       # aksi manuver keluar (dijalankan dan disimpan di replay)
        elif a_back is not None:
            a = a_back                      # telusur jejak mundur; refleks pengaman tetap didahulukan
            a_safe = shield(raw) if SAFETY_LAYER else None
            if a_safe is not None:
                a = a_safe
        else:
            a = agent.act(s, eps)
            a_tabu = lg.tabu_action(raw, env.n_decisions) if LOOP_LAYER else None
            if a_tabu is not None:
                agent.ep_shield += a_tabu != a
                a = a_tabu                  # zona terlarang: menjauh dari jebakan
                agent.last_greedy = False
            if SAFETY_LAYER:
                a_safe = shield(raw)
                if a_safe is not None:
                    agent.ep_shield += a_safe != a
                    a = a_safe              # aksi yang dijalankan (dan disimpan di replay) = aksi pengaman
                else:
                    agent.ep_greedy += agent.last_greedy
            else:
                agent.ep_greedy += agent.last_greedy
            if SIDE_MEMORY:
                a_mem = side_filter(a, raw, any(m[0] for m in side_mem), any(m[1] for m in side_mem))
                if a_mem != a:
                    agent.ep_shield += 1
                    a = a_mem
        agent.ep_counts[a] += 1
        raw, r, outcome = env.step(a)
        odo.append((raw[7], raw[8]))
        lg.update(raw[7], raw[8], raw[9])
        if LOOP_LAYER and outcome is None and lg.mode is None and esc is None and lg.detect():
            pts = list(lg.win)[::LOOP_MARK]
            lg.trigger(env.n_decisions)                   # telusur jejak mundur + zona terlarang sementara
            agent.ep_loop += 1
            odo = []
            env._pos_hist = []
            if learn and eps <= LOOP_PEN_EPS:
                tx, ty, _ = env.true_pose()
                h = env._hot(tx, ty)
                n_prev = h[2] if h else 0
                for px, py, _yaw in pts:
                    env._register(env.hotspots, px, py)   # sabuk orbit ikut penalti zona
                r += R_LOOP - min(arena_env.grow(LOOP_GROW, n_prev), LOOP_GROW_CAP)
        raw2 = raw
        s2, ir2 = features(raw2, env.arena, env.n_decisions, ir)
        if learn:
            agent.remember(s, a, r, s2, outcome is not None)
            agent.learn()
            q_sum += agent.last_q
            q_n += 1
        total += r
        if record_path:
            x, y, _ = env.true_pose()
            path.append((round(x, 4), round(y, 4)))
        s, ir = s2, ir2
        if outcome:
            if learn:
                agent.end_episode(outcome == "goal")
            agent.ep_q = q_sum / q_n if q_n else 0.0
            return outcome, env.n_decisions, total, path


def evaluate(env, agent, fname="eval_result.json", meta=None):
    """Evaluasi greedy (epsilon 0) dari beberapa arah awal; format sama dengan Q-learning."""
    results, k = [], 0
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
    (OUT_DIR / fname).write_text(json.dumps({
        "success": ok, "n": n, "arena": env.arena, "target": list(lay["target"]), "start": list(env.start),
        "target_r": env.target_r, "obstacles": lay["obstacles"], "hazards": lay["hazards"],
        "headings": list(EVAL_HEADINGS), "repeats": EVAL_REPEATS, "by_heading": by_heading,
        "pure_dqn": PURE_DQN, "gamma": GAMMA, "n_step": N_STEP, "seed": SEED, "train": meta or {},
        "episodes": results}), encoding="utf-8")


def main():
    t_start = time.time()
    env = ArenaEnv()
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    agent = DQNAgent(N_FEATURES, len(ACTIONS), rng, gamma=GAMMA, n_policy=N_POLICY_ACTIONS, n_step=N_STEP)
    print(f"continuous_dqn: seed={SEED} episodes<={EPISODES} gamma={GAMMA} r_step={R_STEP_DQN} "
          f"MODE={'DQN MURNI' if PURE_DQN else 'DQN+lapisan'} n_step={N_STEP} pengaman={'ON' if SAFETY_LAYER else 'off'} pemecah_loop={'ON' if ESCAPE_LAYER else 'off'} stuck_grow={arena_env.STUCK_GROW} zone_step={arena_env.ZONE_STEP} coll_grow={arena_env.COLL_GROW} r_goal_time={R_GOAL_TIME_DQN} r_collision_near={R_COLLISION_NEAR_DQN} r_stuck={R_STUCK_DQN} "
          f"target_known={TARGET_KNOWN} "
          f"start_heading={START_HEADING} aksi={list(ACTION_NAMES)}")

    if EVAL_ONLY:
        agent.load_state_dict(torch.load(HERE / "dqn_model_best.pt")["state"])
        evaluate(env, agent)
        return env

    log = open(OUT_DIR / "episode_log.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(log)
    w.writerow(["episode", "epsilon", "decisions", "reward", "outcome", "hazard_steps", "sim_time",
                "states", "start_heading", "end_x", "end_y", "greedy"]
               + [f"aksi_{i}" for i in range(len(ACTIONS))] + ["pengaman", "pemecah", "q_rata", "putaran"])
    eps = EPS_START
    heading_rng = random.Random(SEED + 1)
    window = deque(maxlen=WINDOW)
    best_rate, since_best, stop, rate = -1.0, 0, "max_episodes", 0.0
    best_sd, best_ep = None, 0
    for ep in range(1, EPISODES + 1):
        heading = heading_rng.uniform(*START_HEADING)
        outcome, steps, total, _ = run_episode(env, agent, eps, True,
                                               start_yaw=math.radians(90.0 - heading))
        end_x, end_y, _ = env.true_pose()
        window.append(outcome == "goal")
        rate = sum(window) / len(window)
        w.writerow([ep, f"{eps:.4f}", steps, f"{total:.2f}", outcome, env.hazard_steps,
                    f"{env.sim_time():.1f}", agent.size, f"{heading:.1f}", f"{end_x:.3f}", f"{end_y:.3f}",
                    agent.ep_greedy] + agent.ep_counts + [agent.ep_shield, agent.ep_escape, f"{agent.ep_q:.3f}", agent.ep_loop])
        log.flush()
        if ep % 50 == 0:
            dump_hotspots(env)
            torch.save({"state": agent.state_dict(), "rate": rate, "episode": ep}, OUT_DIR / "dqn_model_last.pt")
        print(f"EP {ep:4d} eps={eps:.2f} {outcome:9s} decisions={steps:3d} R={total:7.1f} "
              f"hazard={env.hazard_steps:3d} rate{len(window)}={rate:.2f} loss={agent.last_loss:.4f} "
              f"buffer={agent.size} sukses={agent.succ.size} pengaman={agent.ep_shield} pemecah={agent.ep_escape} putaran={agent.ep_loop} Q={agent.ep_q:.2f}")
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if len(window) == WINDOW:
            if rate > best_rate:
                best_rate, since_best = rate, 0
                best_sd, best_ep = agent.state_dict(), ep
                torch.save({"state": best_sd, "rate": rate, "episode": ep}, OUT_DIR / "dqn_model.pt")
            elif eps <= EPS_MIN + 1e-12:
                since_best += 1
            if EARLY_STOP and rate >= CONVERGED:
                stop = "converged"
                break
            if EARLY_STOP and since_best >= PATIENCE:
                stop = "patience"
                break
        if STOP_AT_EPS_MIN and eps <= EPS_MIN + 1e-12:
            stop = "eps_min"
            break
    log.close()
    dump_hotspots(env)
    print(f"TRAINING selesai: {stop}, episode={ep}, rate_akhir={rate:.2f}, buffer={agent.size}")
    torch.save({"state": agent.state_dict(), "rate": rate, "episode": ep}, OUT_DIR / "dqn_model_last.pt")
    if best_sd is not None:
        print(f"Model rate tertinggi: episode {best_ep}, success rate {best_rate:.2f} (disimpan di dqn_model.pt, "
              f"hanya pembanding)")
    else:
        torch.save({"state": agent.state_dict(), "rate": rate, "episode": ep}, OUT_DIR / "dqn_model.pt")
    meta = {"stop": stop, "episodes": ep, "updates": agent.updates, "wall_seconds": round(time.time() - t_start),
            "rate_akhir": round(rate, 3), "best_rate": round(best_rate, 3), "best_episode": best_ep}
    print(f"Evaluasi 1: JARINGAN TERAKHIR (episode {ep}) -> eval_result.json; updates={agent.updates} "
          f"waktu={meta['wall_seconds']} s")
    evaluate(env, agent, "eval_result.json", meta)
    if best_sd is not None and best_ep != ep:
        agent.load_state_dict(best_sd)
        print(f"Evaluasi 2: MODEL RATE TERTINGGI (episode {best_ep}) -> eval_result_rate_tertinggi.json "
              f"(pembanding Q-learning yang memakai tabel rate tertinggi)")
        evaluate(env, agent, "eval_result_rate_tertinggi.json", meta)
    return env


if __name__ == "__main__":
    env = main()
    if os.environ.get("MICROMOUSE_QUIT") == "1":
        env.robot.simulationQuit(0)
    else:
        while env.robot.step(16) != -1:
            pass
