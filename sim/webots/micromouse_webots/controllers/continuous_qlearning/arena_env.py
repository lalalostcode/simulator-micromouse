"""Lingkungan arena kontinu untuk Q-learning (Webots, tanpa grid).

Robot micromouse yang sama dengan proyek sebelumnya, tetapi tugasnya menemukan
dan mencapai area target di arena berisi rintangan. Dunia fisik kontinu: kecepatan
roda bernilai riil, fisika dihitung tiap 16 ms. Hanya state dan aksi untuk Q-table
yang didiskretkan (bin), dan itu semua ada di file ini:

  * ArenaEnv.observe_raw()  -> sensor/odometri mentah (desimal)
  * discretize(raw)         -> tuple bilangan bulat (kunci Q-table)
  * ACTIONS                 -> pasangan kecepatan roda (rad/s), nilai riil

Dua mode (konstanta TARGET_KNOWN di bawah):
  * TARGET_KNOWN = False (bawaan): robot TIDAK tahu letak target maupun area bahaya.
    State = bacaan IR + posisi (odometri, sel 0,3 m) + arah hadap. Satu-satunya
    imbalan menuju target adalah +R_GOAL saat pusat robot masuk area target; ada
    bonus kecil tiap kali robot memasuki sel baru supaya menjelajah. Lewat banyak
    episode Q-learning mengingat jalur yang berhasil.
  * TARGET_KNOWN = True: robot diberi koordinat target dan area bahaya; state
    memuat arah dan jarak ke target, dan reward memuat progres mendekati target.

Untuk DQN minggu depan cukup memakai observe_raw()/vektor desimal tanpa
discretize(); world, aksi, dan reward tidak perlu diubah.

Tata letak dibaca dari world saat controller mulai (lihat read_world_layout):
rintangan, area bahaya, target, dan pose awal robot bisa digeser/diputar langsung
di GUI Webots lalu world disimpan (Ctrl+S); tidak perlu mengubah kode.

Sensor yang dipakai agen: IR (ds_fl, ds_fr, ds_dl, ds_dr), gyro, dan encoder
roda. Posisi dan arah robot dihitung dari odometri (encoder + gyro), bukan
dari posisi asli. Posisi asli dari Supervisor hanya dipakai untuk reward,
deteksi sampai target, deteksi tabrakan, area bahaya, dan log.
"""
import json
import math

from controller import Supervisor

TIME_STEP = 16
DT = TIME_STEP / 1000.0
WHEEL_R = 0.0145
TRACK = 0.0746            # jarak antar pusat roda
MAX_WHEEL = 31.416        # rad/s, batas motor
DECISION_STEPS = 8        # satu aksi dijalankan 8 x 16 ms = 128 ms
SETTLE_STEPS = 10
CONTACT_MIN_Z = 0.003     # kontak di bawah ini = lantai, bukan dinding/rintangan

# Aksi: (omega_kiri, omega_kanan) rad/s. Indeks 0..4 = indeks kolom Q-table.
ACTIONS = (
    (10.0, 10.0),    # 0 maju lurus
    (6.0, 10.0),     # 1 maju sambil belok kiri
    (10.0, 6.0),     # 2 maju sambil belok kanan
    (-4.0, 4.0),     # 3 putar kiri di tempat
    (4.0, -4.0),     # 4 putar kanan di tempat
)
ACTION_NAMES = ("lurus", "maju-kiri", "maju-kanan", "putar-kiri", "putar-kanan")

# Batas bin sensor IR (nilai ADC, makin besar makin dekat; urutan dari terdekat).
# FL/FR (lurus ke depan), jarak moncong robot ke rintangan hasil ukur respons sensor:
#   >= 1150 : < ~40 mm     >= 470 : 40-80 mm     >= 150 : 80-150 mm     lainnya : > 150 mm
# DL/DR (diagonal 45 derajat) cukup 3 tingkat.
IR_FRONT_EDGES = (1150.0, 470.0, 150.0)       # -> 4 tingkat (0 = paling dekat)
IR_DIAG_EDGES = (1150.0, 300.0)               # -> 3 tingkat
# Batas bin arah ke target (rad, positif = target di kiri) dan jarak (m).
BEARING_EDGES = (math.radians(-60), math.radians(-15), math.radians(15), math.radians(60))
DIST_EDGES = (0.35, 1.00)
HAZARD_SENSE = 0.30     # m dari tepi area bahaya: mulai dianggap "dekat"
N_STATES = 4 * 4 * 3 * 3 * 5 * 3 * 4      # FL, FR, DL, DR, arah, jarak, bahaya (TARGET_KNOWN)

# Mode: robot diberi tahu letak target (True) atau harus menemukannya sendiri (False).
TARGET_KNOWN = False
# State untuk mode TARGET_KNOWN = False: IR + posisi + arah hadap (tanpa koordinat target).
IR_FRONT3_EDGES = (1150.0, 470.0)     # FL/FR: 3 tingkat (<~40 mm, 40-80 mm, > 80 mm)
IR_DIAG2_EDGES = (600.0,)             # DL/DR: 2 tingkat (dekat / jauh)
POS_CELL = 0.30                       # m, lebar sel posisi untuk state
HEADING_BINS = 8                      # arah hadap dibagi 8 (45 derajat per bin)
_NPOS = [8]                           # jumlah sel per sumbu; diisi ArenaEnv dari ukuran arena

# Reward
R_GOAL = 100.0
R_COLLISION = -100.0
R_STEP = -0.05
R_SPIN = -0.10            # penalti kecil untuk putar di tempat (supaya tidak berputar terus)
R_TIMEOUT = -20.0         # penalti saat waktu episode habis (berputar-putar tidak lebih aman dari menabrak)
R_HAZARD = -3.0           # per langkah keputusan selama pusat robot di dalam area bahaya
R_PROGRESS = 15.0         # per meter mendekati target (jarak asli); hanya bila TARGET_KNOWN
# Bonus saat pertama kali memasuki sebuah sel posisi di episode itu (bila tidak TARGET_KNOWN).
# Makin jauh sel dari titik start, makin besar bonusnya, supaya robot terdorong keluar dari area start.
R_EXPLORE = 0.3           # sel di dekat start
R_EXPLORE_FAR = 1.5       # sel terjauh (sudut arena yang berlawanan dari start)
R_STUCK = -30.0           # episode diakhiri bila robot macet (lihat STUCK_*)
STUCK_WINDOW = 20         # jumlah keputusan terakhir yang diperiksa (~2,6 detik)
STUCK_DIST = 0.08         # m: perpindahan bersih di bawah ini selama STUCK_WINDOW = macet
MAX_DECISIONS = 400 if TARGET_KNOWN else 800


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _bin_ir(v, edges):
    """Tingkat IR: 0 = paling dekat. Makin besar nilai sensor, makin dekat."""
    for i, e in enumerate(edges):
        if v >= e:
            return i
    return len(edges)


def _bin_edges(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def pos_cell(x, y):
    """Sel posisi (i, j) dari posisi hasil odometri."""
    n = _NPOS[0]
    return (min(max(int(x // POS_CELL), 0), n - 1), min(max(int(y // POS_CELL), 0), n - 1))


def heading_bin(yaw):
    width = 2 * math.pi / HEADING_BINS
    return int(((yaw + width / 2) % (2 * math.pi)) // width) % HEADING_BINS


def discretize(raw):
    """raw = (fl, fr, dl, dr, bearing, dist, hazard, x, y, yaw) -> tuple int (kunci Q-table).

    TARGET_KNOWN = True : (FL, FR, DL, DR, arah ke target, jarak ke target, zona bahaya)
                          hazard: 0 aman, 1 di dalam, 2 dekat di kiri depan, 3 dekat di kanan depan.
    TARGET_KNOWN = False: (FL, FR, DL, DR, sel x, sel y, arah hadap)
    """
    fl, fr, dl, dr, bearing, dist, hazard, x, y, yaw = raw
    if TARGET_KNOWN:
        return (_bin_ir(fl, IR_FRONT_EDGES), _bin_ir(fr, IR_FRONT_EDGES),
                _bin_ir(dl, IR_DIAG_EDGES), _bin_ir(dr, IR_DIAG_EDGES),
                _bin_edges(bearing, BEARING_EDGES), _bin_edges(dist, DIST_EDGES), hazard)
    cx, cy = pos_cell(x, y)
    return (_bin_ir(fl, IR_FRONT3_EDGES), _bin_ir(fr, IR_FRONT3_EDGES),
            _bin_ir(dl, IR_DIAG2_EDGES), _bin_ir(dr, IR_DIAG2_EDGES),
            cx, cy, heading_bin(yaw))


def _yaw(node):
    o = node.getOrientation()
    return math.atan2(o[3], o[0])


def _geometry(shape_like):
    """Geometri dari Shape (field geometry) atau dari boundingObject (langsung geometri)."""
    if shape_like.getTypeName() == "Shape":
        return shape_like.getField("geometry").getSFNode()
    return shape_like


def _local_points(geom):
    coord = geom.getField("coord").getSFNode()
    pf = coord.getField("point")
    pts = [pf.getMFVec3f(i) for i in range(pf.getCount())]
    return [(p[0], p[1]) for p in pts[:len(pts) // 2]]      # alas bawah (z = 0)


def read_world_layout(robot):
    """Baca rintangan, area bahaya, dan target dari node di world.

    Aturan nama (field `name` pada Solid di root):
      * 'bahaya...'  -> area bahaya (Cylinder pipih, tanpa boundingObject)
      * 'target'     -> target (DEF TARGET), radius dari Cylinder-nya
      * Solid lain yang punya boundingObject, selain 'floor' dan 'wall...'
                      -> rintangan (Cylinder, Box, atau prisma IndexedFaceSet)
    Menyalin Solid di GUI lalu menamainya 'rintangan_baru' menambah rintangan.
    """
    children = robot.getRoot().getField("children")
    obstacles, hazards, target = [], [], None
    for i in range(children.getCount()):
        n = children.getMFNode(i)
        if n.getTypeName() != "Solid":
            continue
        name = n.getField("name").getSFString()
        x, y, _ = n.getPosition()
        if name == "target":
            geom = _geometry(n.getField("children").getMFNode(0))
            target = ((x, y), geom.getField("radius").getSFFloat())
        elif name.startswith("bahaya"):
            geom = _geometry(n.getField("children").getMFNode(0))
            hazards.append((x, y, geom.getField("radius").getSFFloat()))
        elif name == "floor" or name.startswith("wall"):
            continue
        else:
            bo = n.getField("boundingObject").getSFNode()
            if bo is None:
                continue
            g = _geometry(bo)
            t = g.getTypeName()
            if t == "Cylinder":
                obstacles.append({"name": name, "shape": "cyl", "x": x, "y": y,
                                  "r": g.getField("radius").getSFFloat()})
            elif t == "Box":
                sz = g.getField("size").getSFVec3f()
                obstacles.append({"name": name, "shape": "box", "x": x, "y": y,
                                  "sx": sz[0], "sy": sz[1], "yaw": _yaw(n)})
            elif t == "IndexedFaceSet":
                o = n.getOrientation()
                pts = [[x + o[0] * lx + o[1] * ly, y + o[3] * lx + o[4] * ly]
                       for lx, ly in _local_points(g)]
                obstacles.append({"name": name, "shape": "tri", "pts": pts})
    if target is None:
        raise RuntimeError("Solid bernama 'target' tidak ditemukan di world.")
    return obstacles, hazards, target


class ArenaEnv:
    def __init__(self):
        self.robot = Supervisor()
        self.me = self.robot.getSelf()
        cfg = json.loads(self.robot.getCustomData())
        self.arena = cfg["arena"]
        _NPOS[0] = max(1, int(math.ceil(self.arena / POS_CELL - 1e-9)))
        # Tata letak dibaca dari world dan TETAP selama run (controller tidak memindahkannya).
        # Pose awal robot = pose robot di world saat dimuat.
        x0, y0, _ = self.me.getPosition()
        self.start = (x0, y0)
        self.start_yaw = _yaw(self.me)
        obstacles, hazards, (tpos, self.target_r) = read_world_layout(self.robot)
        self.layout = {"target": tpos, "yaw": self.start_yaw,
                       "obstacles": obstacles, "hazards": hazards}

        self.ds = {n: self.robot.getDevice(n) for n in ("ds_fl", "ds_fr", "ds_dl", "ds_dr")}
        for d in self.ds.values():
            d.enable(TIME_STEP)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        self.enc = [self.robot.getDevice("left_wheel_sensor"), self.robot.getDevice("right_wheel_sensor")]
        for e in self.enc:
            e.enable(TIME_STEP)
        self.motors = [self.robot.getDevice("left_wheel_motor"), self.robot.getDevice("right_wheel_motor")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.me.enableContactPointsTracking(TIME_STEP, True)

        self.n_decisions = 0
        self.hazard_steps = 0
        self._pos_hist = []
        self._visited = set()
        self.est = [0.0, 0.0, 0.0]        # odometri: x, y, yaw
        self.target_est = (0.0, 0.0)
        self._last_enc = [0.0, 0.0]
        self._prev_dist = 0.0
        self.sim_time0 = self.robot.getTime()

    # ---------------- dunia -----------------
    def true_pose(self):
        x, y, _ = self.me.getPosition()
        o = self.me.getOrientation()
        return x, y, math.atan2(o[3], o[0])

    def _in_hazard(self, x, y):
        return any(math.hypot(x - hx, y - hy) < hr for hx, hy, hr in self.layout["hazards"])

    def _hazard_feature(self):
        """Fitur zona bahaya dari odometri (bukan posisi asli)."""
        x, y, yaw = self.est
        best = None
        if not self.layout["hazards"]:
            return 0
        for hx, hy, hr in self.layout["hazards"]:
            edge = math.hypot(hx - x, hy - y) - hr
            if best is None or edge < best[0]:
                best = (edge, math.atan2(hy - y, hx - x))
        edge, ang = best
        if edge <= 0.0:
            return 1
        if edge > HAZARD_SENSE:
            return 0
        rel = wrap(ang - yaw)
        if abs(rel) > math.pi / 2:      # di belakang: tidak relevan
            return 0
        return 2 if rel > 0 else 3

    def _true_dist(self):
        x, y, _ = self.true_pose()
        tx, ty = self.layout["target"]
        return math.hypot(tx - x, ty - y)

    def set_wheels(self, wl, wr):
        s = max(1.0, abs(wl) / MAX_WHEEL, abs(wr) / MAX_WHEEL)
        self.motors[0].setVelocity(wl / s)
        self.motors[1].setVelocity(wr / s)

    def _step(self):
        if self.robot.step(TIME_STEP) == -1:
            raise SystemExit
        self._update_odometry()

    def _update_odometry(self):
        e = [self.enc[0].getValue(), self.enc[1].getValue()]
        d = ((e[0] - self._last_enc[0]) + (e[1] - self._last_enc[1])) * WHEEL_R / 2.0
        self._last_enc = e
        w = self.gyro.getValues()[2]
        yaw0 = self.est[2]
        yaw1 = yaw0 + w * DT
        ym = (yaw0 + yaw1) / 2.0
        self.est[0] += d * math.cos(ym)
        self.est[1] += d * math.sin(ym)
        self.est[2] = yaw1

    def _contact(self):
        return any(c.point[2] > CONTACT_MIN_Z for c in self.me.getContactPoints(True))

    # ---------------- episode -----------------
    def reset(self, yaw=None):
        """Mulai episode di titik start. yaw (rad, 0 = timur, pi/2 = utara) opsional;
        tanpa argumen dipakai arah robot di world."""
        layout = self.layout
        sx, sy = self.start
        self.me.getField("translation").setSFVec3f([sx, sy, 0.0145])
        self.me.getField("rotation").setSFRotation([0, 0, 1, layout["yaw"] if yaw is None else yaw])
        self.me.resetPhysics()
        self.set_wheels(0.0, 0.0)
        for _ in range(SETTLE_STEPS):
            self.robot.step(TIME_STEP)
        # Pose awal robot dianggap diketahui; odometri mulai dari sini.
        self._last_enc = [self.enc[0].getValue(), self.enc[1].getValue()]
        x, y, yaw = self.true_pose()
        self.est = [x, y, yaw]
        self.target_est = layout["target"]       # hanya dipakai bila TARGET_KNOWN
        self._prev_dist = self._true_dist()
        self.n_decisions = 0
        self.hazard_steps = 0
        self._pos_hist = []
        self._visited = {pos_cell(self.est[0], self.est[1])}
        self._start_xy = (self.est[0], self.est[1])
        A = self.arena
        self._dmax = max(math.hypot(cx - self.est[0], cy - self.est[1])
                         for cx in (0.0, A) for cy in (0.0, A))
        return self.observe_raw()

    def _explore_bonus(self, cell):
        """Bonus sel baru: naik linear dari R_EXPLORE (dekat start) ke R_EXPLORE_FAR (sudut terjauh)."""
        cx = min((cell[0] + 0.5) * POS_CELL, self.arena)
        cy = min((cell[1] + 0.5) * POS_CELL, self.arena)
        f = min(1.0, math.hypot(cx - self._start_xy[0], cy - self._start_xy[1]) / self._dmax)
        return R_EXPLORE + (R_EXPLORE_FAR - R_EXPLORE) * f

    def observe_raw(self):
        fl, fr, dl, dr = (self.ds[n].getValue() for n in ("ds_fl", "ds_fr", "ds_dl", "ds_dr"))
        x, y, yaw = self.est
        if TARGET_KNOWN:
            tx, ty = self.target_est
            bearing = wrap(math.atan2(ty - y, tx - x) - yaw)
            dist = math.hypot(tx - x, ty - y)
            hazard = self._hazard_feature()
        else:
            bearing = dist = hazard = 0       # tidak dipakai: robot tidak tahu letak target/bahaya
        return (fl, fr, dl, dr, bearing, dist, hazard, x, y, yaw)

    def step(self, action):
        wl, wr = ACTIONS[action]
        self.set_wheels(wl, wr)
        collided = False
        for _ in range(DECISION_STEPS):
            self._step()
            if self._contact():
                collided = True
                break
        self.n_decisions += 1
        d = self._true_dist()
        reward = R_STEP + (R_PROGRESS * (self._prev_dist - d) if TARGET_KNOWN else 0.0)
        if not TARGET_KNOWN:
            cell = pos_cell(self.est[0], self.est[1])
            if cell not in self._visited:
                self._visited.add(cell)
                reward += self._explore_bonus(cell)
        if action >= 3:
            reward += R_SPIN
        self._prev_dist = d
        x, y, _ = self.true_pose()
        if self._in_hazard(x, y):
            reward += R_HAZARD
            self.hazard_steps += 1
        self._pos_hist.append((x, y))
        stuck = (len(self._pos_hist) > STUCK_WINDOW and math.hypot(
            x - self._pos_hist[-1 - STUCK_WINDOW][0], y - self._pos_hist[-1 - STUCK_WINDOW][1]) < STUCK_DIST)
        outcome = None
        if collided:
            reward, outcome = R_COLLISION, "collision"
        elif d <= self.target_r:
            reward, outcome = R_GOAL, "goal"
        elif stuck:
            reward, outcome = R_STUCK, "stuck"
        elif self.n_decisions >= MAX_DECISIONS:
            reward += R_TIMEOUT
            outcome = "timeout"
        if outcome:
            self.set_wheels(0.0, 0.0)
        return self.observe_raw(), reward, outcome

    def sim_time(self):
        return self.robot.getTime() - self.sim_time0
