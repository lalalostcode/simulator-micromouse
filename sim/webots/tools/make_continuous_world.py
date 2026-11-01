#!/usr/bin/env python3
"""Buat dunia Webots arena kontinu (tanpa grid) untuk Q-learning menuju target.

Robot (model micromouse) disalin apa adanya dari dunia micromouse_qlearning.wbt,
jadi sensor IR, gyro, encoder, dan motornya sama dengan proyek sebelumnya.

Arena 2,4 x 2,4 m, tata letak TETAP (hardcode di OBSTACLES dan HAZARDS):
  * START kiri-bawah, TARGET kanan-atas
  * 10 rintangan padat (bertabrakan) dari berbagai bentuk dan ukuran:
    lingkaran besar/sedang/kecil, persegi, persegi panjang (dinding), segitiga
  * 2 area bahaya berbentuk lingkaran di sekitar tengah peta. Tidak bertabrakan
    dan tidak terbaca IR (hanya warna lantai); controller memberi reward minus
    tiap langkah keputusan selama robot berada di dalamnya.

Pakai:  python tools/make_continuous_world.py          # hanya bila world belum ada
        python tools/make_continuous_world.py --force  # timpa world yang ada
Keluaran: micromouse_webots/worlds/micromouse_continuous.wbt

PENTING: tata letak di sini hanya titik awal. Setelah world dibuat, atur posisi
rintangan, area bahaya, target, dan robot LANGSUNG di GUI Webots lalu simpan
(Ctrl+S); controller membaca posisi dari world (lihat arena_env.read_world_layout).
Karena itu skrip ini menolak menimpa world yang sudah ada kecuali dengan --force
(perubahan dari GUI akan hilang).
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORLDS = HERE.parent / "micromouse_webots" / "worlds"
SOURCE = WORLDS / "micromouse_qlearning.wbt"
OUT = WORLDS / "micromouse_continuous.wbt"

ARENA = 2.4          # m, sisi dalam arena
WALL_T = 0.012       # m, tebal dinding luar
WALL_H = 0.05        # m, tinggi dinding dan rintangan
TARGET_R = 0.25      # m, radius area target (robot sampai bila pusatnya masuk)
START = (0.30, 0.30)
TARGET = (2.10, 2.10)
START_YAW = math.pi / 4   # menghadap diagonal


def cyl(name, x, y, r):
    return {"name": name, "shape": "cyl", "x": x, "y": y, "r": r}


def box(name, x, y, sx, sy, yaw=0.0):
    return {"name": name, "shape": "box", "x": x, "y": y, "sx": sx, "sy": sy, "yaw": yaw}


def tri(name, p0, p1, p2):
    return {"name": name, "shape": "tri", "pts": [list(p0), list(p1), list(p2)]}


OBSTACLES = [
    cyl("lingkaran_besar", 0.80, 0.55, 0.12),
    box("persegi_panjang_1", 1.30, 0.40, 0.35, 0.12, 0.5),
    tri("segitiga_1", (1.75, 0.60), (2.10, 0.75), (1.80, 1.05)),
    cyl("lingkaran_kecil_1", 2.00, 1.40, 0.06),
    box("persegi_kecil", 0.45, 1.00, 0.14, 0.14, 0.0),
    tri("segitiga_2", (0.35, 1.55), (0.75, 1.70), (0.45, 1.95)),
    box("dinding_panjang", 1.30, 1.85, 0.50, 0.08, -0.3),
    cyl("lingkaran_sedang", 1.80, 1.90, 0.09),
    box("persegi_panjang_2", 0.95, 0.95, 0.22, 0.16, 1.0),
    cyl("lingkaran_kecil_2", 1.55, 1.55, 0.05),
]
# Area bahaya (lingkaran di lantai, tanpa tabrakan): (x, y, radius)
HAZARDS = [(0.90, 1.40, 0.22), (1.50, 1.00, 0.22)]
HAZARD_RGB = "0.95 0.45 0.10"


def _prism(pts):
    """Prisma segitiga (IndexedFaceSet) berpusat di titik beratnya.

    Mengembalikan (geometri, (cx, cy)); koordinat titik relatif terhadap pusat,
    jadi Solid-nya bisa digeser/diputar di GUI. Titik dibuat berlawanan jarum jam.
    """
    (x0, y0), (x1, y1), (x2, y2) = pts
    if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0:
        pts = [pts[0], pts[2], pts[1]]
    cx = sum(p[0] for p in pts) / 3
    cy = sum(p[1] for p in pts) / 3
    pts = [(px - cx, py - cy) for px, py in pts]
    low = " ".join(f"{x:.4f} {y:.4f} 0," for x, y in pts)
    high = " ".join(f"{x:.4f} {y:.4f} {WALL_H}," for x, y in pts)
    return (f"IndexedFaceSet {{ coord Coordinate {{ point [ {low} {high} ] }} "
            f"coordIndex [ 0 2 1 -1 3 4 5 -1 0 1 4 3 -1 1 2 5 4 -1 2 0 3 5 -1 ] }}", (cx, cy))


def solid(o, rgb):
    """Solid padat. Geometri visual dan boundingObject adalah SATU node (DEF/USE),
    jadi mengubah ukuran di GUI otomatis mengubah hitbox juga."""
    name = o["name"]
    if o["shape"] == "tri":
        geom, (cx, cy) = _prism(o["pts"])
        return (f"Solid {{\n  translation {cx:.4f} {cy:.4f} 0\n  children [\n    Shape {{\n      appearance PBRAppearance {{ baseColor {rgb} roughness 1 metalness 0 }}\n"
                f"      geometry DEF GEO_{name} {geom}\n    }}\n  ]\n  name \"{name}\"\n  boundingObject USE GEO_{name}\n}}\n")
    if o["shape"] == "cyl":
        geom = f"Cylinder {{ height {WALL_H} radius {o['r']} }}"
        x, y, yaw = o["x"], o["y"], 0.0
    else:
        geom = f"Box {{ size {o['sx']} {o['sy']} {WALL_H} }}"
        x, y, yaw = o["x"], o["y"], o["yaw"]
    return (f"Solid {{\n  translation {x:.4f} {y:.4f} {WALL_H / 2:.4f}\n  rotation 0 0 1 {yaw:.4f}\n  children [\n"
            f"    Shape {{\n      appearance PBRAppearance {{ baseColor {rgb} roughness 1 metalness 0 }}\n"
            f"      geometry DEF GEO_{name} {geom}\n    }}\n  ]\n  name \"{name}\"\n  boundingObject USE GEO_{name}\n}}\n")


def disc(name, x, y, r, rgb, z, transparency=0.0):
    """Cakram tipis di lantai tanpa boundingObject (tidak ditabrak, tidak terbaca IR)."""
    return (f"Solid {{\n  translation {x:.4f} {y:.4f} {z}\n  children [\n    Shape {{\n"
            f"      appearance PBRAppearance {{ baseColor {rgb} roughness 1 metalness 0 transparency {transparency} }}\n"
            f"      geometry Cylinder {{ height 0.001 radius {r} }}\n    }}\n  ]\n  name \"{name}\"\n}}\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="timpa world yang sudah ada")
    if OUT.exists() and not ap.parse_args().force:
        raise SystemExit(f"{OUT.name} sudah ada; atur tata letak di GUI Webots. "
                         "Pakai --force bila memang ingin membuat ulang dari konstanta.")
    src = SOURCE.read_text(encoding="utf-8")
    robot = src[src.index("\nRobot {") + 1:]
    robot = re.sub(r"translation [^\n]+", f"translation {START[0]:.4f} {START[1]:.4f} 0.0145", robot, count=1)
    robot = re.sub(r"rotation [^\n]+", f"rotation 0 0 1 {START_YAW:.7f}", robot, count=1)
    robot = robot.replace('controller "qlearning"', 'controller "continuous_qlearning"')
    config = {"arena": ARENA}      # sisanya dibaca controller dari node di world
    robot = re.sub(r"(  supervisor TRUE\n)", lambda m: m.group(1) +
                   "  customData " + json.dumps(json.dumps(config)) + "\n", robot, count=1)

    c = ARENA / 2
    walls = lambda n, x, y, sx, sy: solid(box(n, x, y, sx, sy), "0.85 0.1 0.1")
    out = ["#VRML_SIM R2025a utf8\n\nWorldInfo {\n  basicTimeStep 16\n}\n",
           f"Viewpoint {{\n  orientation -0.25 0.2 0.95 1.3\n  position {c - 2.5:.3f} {c - 2.5:.3f} 3.6\n}}\n",
           "Background {\n  skyColor [ 0.75 0.82 0.92 ]\n}\n",
           "DirectionalLight {\n  direction -0.3 -0.4 -1\n  intensity 1.6\n  ambientIntensity 0.5\n}\n",
           f"Solid {{\n  translation {c} {c} -0.005\n  children [\n    Shape {{\n"
           f"      appearance PBRAppearance {{ baseColor 0.80 0.80 0.78 roughness 1 }}\n"
           f"      geometry Box {{ size {ARENA + 0.1} {ARENA + 0.1} 0.01 }}\n    }}\n  ]\n"
           f"  name \"floor\"\n  boundingObject Box {{ size {ARENA + 0.1} {ARENA + 0.1} 0.01 }}\n}}\n"]
    half = WALL_T / 2
    out.append(walls("wall_s", c, -half, ARENA + 2 * WALL_T, WALL_T))
    out.append(walls("wall_n", c, ARENA + half, ARENA + 2 * WALL_T, WALL_T))
    out.append(walls("wall_w", -half, c, WALL_T, ARENA))
    out.append(walls("wall_e", ARENA + half, c, WALL_T, ARENA))
    out.append(disc("pad_start", START[0], START[1], 0.18, "0.30 0.50 0.95", 0.0005))
    for i, (hx, hy, hr) in enumerate(HAZARDS):
        out.append(disc(f"bahaya_{i + 1}", hx, hy, hr, HAZARD_RGB, 0.0007))
    out.append(f"DEF TARGET Solid {{\n  translation {TARGET[0]} {TARGET[1]} 0.001\n  children [\n    Shape {{\n"
               f"      appearance PBRAppearance {{ baseColor 0.1 0.8 0.2 roughness 1 }}\n"
               f"      geometry Cylinder {{ height 0.001 radius {TARGET_R} }}\n    }}\n  ]\n  name \"target\"\n}}\n")
    colors = {"cyl": "0.35 0.35 0.40", "box": "0.55 0.40 0.25", "tri": "0.45 0.30 0.55"}
    for o in OBSTACLES:
        out.append(solid(o, colors[o["shape"]]))
    out.append(robot)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print("ditulis:", OUT, f"({len(OBSTACLES)} rintangan, {len(HAZARDS)} area bahaya)")


if __name__ == "__main__":
    main()
