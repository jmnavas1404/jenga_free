"""
visualizer_open3d.py  —  con forzado X11 para Wayland
======================================================
Fuerza XWayland antes de inicializar Open3D para evitar:
  - "Wayland: The platform does not support setting the window position"
  - "Failed to initialize GLEW"
  - Segmentation fault de Filament

Dependencias:  pip install open3d numpy

Uso:
    python visualizer_open3d.py --N 6
    python visualizer_open3d.py --N 10 --no-labels

Controles:
    Clic izquierdo + arrastrar  → rotar
    Clic derecho   + arrastrar  → pan/zoom
    Q                           → cerrar

 EDITA LAS RUTAS AQUÍ:
"""

ROBOT_GLB_PATH = "assets/robot/YU_5_INDUSTRIAL_13_002.glb"
BLOCK_GLB_PATH = "assets/jenga/fichita.glb"

# ── Forzar X11 ANTES de importar open3d ──────────────────────────────────────
import os
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d

from pose_engine import generar_vectores, pick, place, h


BLOQUE_L = 0.025
BLOQUE_W = 0.075
BLOQUE_H = h

COLOR_PICK  = [0.47, 0.72, 0.87]
COLOR_PLACE = [0.40, 0.78, 0.54]
COLOR_TABLE = [0.70, 0.70, 0.70]
COLOR_LABEL = [1.00, 0.85, 0.15]
COLOR_ROBOT = [0.30, 0.30, 0.35]


def _load_glb_as_mesh(path):
    p = Path(path)
    if not p.exists():
        print(f"  [AVISO] No encontrado: {p.resolve()}")
        return None
    try:
        model = o3d.io.read_triangle_model(str(p))
        combined = o3d.geometry.TriangleMesh()
        for mi in model.meshes:
            combined += mi.mesh.to_legacy()
        combined.compute_vertex_normals()
        print(f"  [OK] {p.name}  ({len(model.meshes)} mesh(es))")
        return combined
    except Exception as e:
        print(f"  [ERROR] {p.name}: {e}")
        return None


def _fallback_block(cx, cy, cz, yaw):
    m = o3d.geometry.TriangleMesh.create_box(BLOQUE_L, BLOQUE_W, BLOQUE_H)
    m.compute_vertex_normals()
    m.translate((-BLOQUE_L / 2, -BLOQUE_W / 2, -BLOQUE_H / 2))
    m.rotate(m.get_rotation_matrix_from_xyz((0, 0, yaw)), center=(0, 0, 0))
    m.translate((cx, cy, cz))
    return m


def _one_block(source, cx, cy, cz, yaw):
    m = copy.deepcopy(source)
    m.translate(-m.get_center())
    m.rotate(m.get_rotation_matrix_from_xyz((0, 0, yaw)), center=(0, 0, 0))
    m.translate((cx, cy, cz))
    return m


def _build_tower(poses, source, use_glb, color):
    merged = o3d.geometry.TriangleMesh()
    for pose in poses:
        blk = (_one_block(source, pose.x, pose.y, pose.z, pose.yaw)
               if use_glb else
               _fallback_block(pose.x, pose.y, pose.z, pose.yaw))
        blk.paint_uniform_color(color)
        merged += blk
    merged.compute_vertex_normals()
    return merged


def _build_labels(poses):
    merged = o3d.geometry.TriangleMesh()
    for pose in poses:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        s.compute_vertex_normals()
        s.paint_uniform_color(COLOR_LABEL)
        s.translate((pose.x, pose.y, pose.z + BLOQUE_H / 2 + 0.006))
        merged += s
    merged.compute_vertex_normals()
    return merged


def _make_table():
    t = o3d.geometry.TriangleMesh.create_box(1.10, 0.50, 0.015)
    t.compute_vertex_normals()
    t.paint_uniform_color(COLOR_TABLE)
    t.translate((-0.55, 0.25, 0.030))
    return t


def build_scene(pisos, show_labels, robot_glb, block_glb):
    print(f"\n{'─'*55}")
    print(f" Jenga visualizer — Open3D  |  {pisos} pisos")
    print(f"{'─'*55}")

    generar_vectores(pisos=pisos)
    n = len(pick)
    print(f" Poses generadas: {n} pick + {n} place\n")

    print("[1/4] Cargando modelo del bloque Jenga...")
    src = _load_glb_as_mesh(block_glb)
    use_glb = src is not None
    if not use_glb:
        print("       → Usando caja procedural.")

    print(f"\n[2/4] Fusionando {n} bloques → torre origen...")
    torre_pick = _build_tower(pick, src, use_glb, COLOR_PICK)

    print(f"[3/4] Fusionando {n} bloques → torre destino...")
    torre_place = _build_tower(place, src, use_glb, COLOR_PLACE)

    print("\n[4/4] Cargando modelo del robot...")
    robot_mesh = _load_glb_as_mesh(robot_glb)
    if robot_mesh is None:
        robot_mesh = o3d.geometry.TriangleMesh.create_cylinder(0.05, 0.30)
        robot_mesh.compute_vertex_normals()
        robot_mesh.paint_uniform_color(COLOR_ROBOT)
        print("       → Usando cilindro como fallback.")

    geoms = [
        torre_pick,
        torre_place,
        _make_table(),
        robot_mesh,
        o3d.geometry.TriangleMesh.create_coordinate_frame(0.08),
    ]

    if show_labels:
        print(f"      Fusionando {n} etiquetas...")
        geoms.append(_build_labels(pick))

    print(f"\n Escena lista: {len(geoms)} objetos\n{'─'*55}\n")
    return geoms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--robot-glb", type=str, default=ROBOT_GLB_PATH)
    p.add_argument("--block-glb", type=str, default=BLOCK_GLB_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    geoms = build_scene(
        pisos       = args.N,
        show_labels = not args.no_labels,
        robot_glb   = args.robot_glb,
        block_glb   = args.block_glb,
    )

    print("Controles:")
    print("  Clic izquierdo + arrastrar  → rotar")
    print("  Clic derecho   + arrastrar  → pan/zoom")
    print("  Q                           → cerrar\n")

    o3d.visualization.draw_geometries(
        geoms,
        window_name         = f"Torre Jenga – {args.N} pisos",
        width               = 1400,
        height              = 850,
        mesh_show_back_face = True,
    )


if __name__ == "__main__":
    main()