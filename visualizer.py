"""
visualizer.py
=============
Visualización estática 3D de la torre Jenga y las trayectorias del robot.

Dependencias:
    pip install pyvista numpy

Uso:
    python visualizer.py --N 10
    python visualizer.py --N 6

Controles de cámara en la ventana PyVista:
    Clic izquierdo + arrastrar  → rotar
    Clic derecho  + arrastrar   → zoom
    Clic medio    + arrastrar   → pan
    Q / Escape                  → cerrar
"""

import argparse
import numpy as np
import pyvista as pv
from math import cos, sin

from pose_engine import (
    generar_vectores,
    pick, place,
    pre_pick, pre_place,
    pre_pre_pick, pre_pre_place,
    h,   # altura del bloque
)

# ── Dimensiones físicas del bloque Jenga (metros) ────────────────────────────
BLOQUE_L = 0.075   # largo
BLOQUE_W = 0.025   # ancho
BLOQUE_H = h       # alto  (igual que en program.py)

# ── Paleta de colores ─────────────────────────────────────────────────────────
COLOR_PICK_TOWER    = "#AED6F1"   # azul claro  – bloques en torre origen
COLOR_PLACE_TOWER   = "#A9DFBF"   # verde claro – bloques en torre destino
COLOR_TRAYECTORIA   = "#E74C3C"   # rojo        – líneas pick → place
COLOR_PRE_PICK      = "#F0B27A"   # naranja      – puntos pre-pick
COLOR_PRE_PLACE     = "#BB8FCE"   # morado       – puntos pre-place
COLOR_TABLA         = "#D5D8DC"   # gris claro   – mesa
COLOR_ORIGIN        = "#2ECC71"   # verde        – origen robot
COLOR_NUMBER        = "white"     # etiquetas


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Matriz de rotación 3×3 alrededor del eje Z."""
    c, s = cos(angle_rad), sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _add_block(plotter: pv.Plotter, cx: float, cy: float, cz: float,
               yaw: float, color: str, opacity: float = 1.0,
               label: str = None):
    """
    Añade un bloque orientado (rotado en Z) al plotter.
    cx, cy, cz  → centro del bloque
    yaw         → rotación en Z (radianes)
    """
    # Crear caja centrada en origen
    half = np.array([BLOQUE_L / 2, BLOQUE_W / 2, BLOQUE_H / 2])
    corners_local = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=float) * half  # 8 vértices

    R = _rotation_matrix_z(yaw)
    corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])

    # Índices de las 6 caras (PyVista face format: [n_pts, i, j, k, ...])
    faces = np.hstack([
        [4, 0, 1, 2, 3],   # bottom
        [4, 4, 5, 6, 7],   # top
        [4, 0, 1, 5, 4],   # front
        [4, 2, 3, 7, 6],   # back
        [4, 0, 3, 7, 4],   # left
        [4, 1, 2, 6, 5],   # right
    ])
    mesh = pv.PolyData(corners_world, faces)
    plotter.add_mesh(mesh, color=color, opacity=opacity,
                     show_edges=True, edge_color="gray", line_width=0.5)

    if label is not None:
        plotter.add_point_labels(
            np.array([[cx, cy, cz + BLOQUE_H / 2 + 0.003]]),
            [label],
            font_size=7,
            text_color=COLOR_NUMBER,
            always_visible=True,
            shape=None,
        )


def _add_trajectory(plotter: pv.Plotter, p_from: np.ndarray,
                    p_to: np.ndarray, color: str,
                    width: int = 2, arrow: bool = True):
    """Dibuja una línea (y opcionalmente una flecha) entre dos puntos 3D."""
    line = pv.Line(p_from, p_to)
    plotter.add_mesh(line, color=color, line_width=width)

    if arrow:
        direction = p_to - p_from
        length    = np.linalg.norm(direction)
        if length > 1e-6:
            mid = p_from + direction * 0.5
            plotter.add_arrows(
                mid.reshape(1, 3),
                (direction / length).reshape(1, 3),
                mag=0.01,
                color=color,
            )


def _add_waypoints(plotter: pv.Plotter, poses, color: str, size: float = 6.0):
    """Dibuja esferas pequeñas en cada waypoint."""
    pts = np.array([[p.x, p.y, p.z] for p in poses])
    cloud = pv.PolyData(pts)
    plotter.add_mesh(cloud, render_points_as_spheres=True,
                     point_size=size, color=color)


def _add_table(plotter: pv.Plotter):
    """Mesa simple como caja plana."""
    table = pv.Box(bounds=(-0.55, 0.55, 0.25, 0.75, 0.03, 0.045))
    plotter.add_mesh(table, color=COLOR_TABLA, opacity=0.4)


def _add_origin_axes(plotter: pv.Plotter):
    """Ejes del sistema de referencia del robot en (0,0,0)."""
    plotter.add_axes_at_origin(line_width=3, labels_off=False)


def build_scene(pisos: int, show_waypoints: bool, show_labels: bool,
                show_trajectories: bool):
    """
    Construye y muestra la escena completa.

    Parámetros
    ----------
    pisos             : número de pisos de la torre
    show_waypoints    : si True, muestra pre_pick y pre_place
    show_labels       : si True, muestra número de operación en cada bloque
    show_trajectories : si True, dibuja líneas pick → place
    """
    generar_vectores(pisos=pisos)

    n_moves = len(pick)  # total de movimientos = 2 * 3 * pisos

    plotter = pv.Plotter(window_size=[1400, 800])
    plotter.set_background("#1A1A2E")   # fondo oscuro elegante

    # ── Mesa ─────────────────────────────────────────────────────────────────
    _add_table(plotter)

    # ── Ejes del robot ────────────────────────────────────────────────────────
    _add_origin_axes(plotter)

    # ── Bloques en torre ORIGEN (posiciones pick) ─────────────────────────────
    print(f"\n[visualizer] Generando {n_moves} bloques en torre origen...")
    for j, pose in enumerate(pick):
        label = str(j) if show_labels else None
        _add_block(
            plotter,
            cx=pose.x, cy=pose.y, cz=pose.z,
            yaw=pose.yaw,
            color=COLOR_PICK_TOWER,
            opacity=0.85,
            label=label,
        )

    # ── Bloques en torre DESTINO (posiciones place) ───────────────────────────
    print(f"[visualizer] Generando {n_moves} bloques en torre destino...")
    for j, pose in enumerate(place):
        _add_block(
            plotter,
            cx=pose.x, cy=pose.y, cz=pose.z,
            yaw=pose.yaw,
            color=COLOR_PLACE_TOWER,
            opacity=0.85,
        )

    # ── Trayectorias pick → place ─────────────────────────────────────────────
    if show_trajectories:
        print("[visualizer] Dibujando trayectorias...")
        for p_pick, p_place in zip(pick, place):
            _add_trajectory(
                plotter,
                p_from=np.array([p_pick.x,  p_pick.y,  p_pick.z]),
                p_to  =np.array([p_place.x, p_place.y, p_place.z]),
                color =COLOR_TRAYECTORIA,
                width =1,
                arrow =True,
            )

    # ── Waypoints pre_pick y pre_place ────────────────────────────────────────
    if show_waypoints:
        print("[visualizer] Dibujando waypoints...")
        _add_waypoints(plotter, pre_pick,  COLOR_PRE_PICK,  size=5)
        _add_waypoints(plotter, pre_place, COLOR_PRE_PLACE, size=5)

        # Líneas verticales: pre_pick  → pick
        for pp, p in zip(pre_pick, pick):
            _add_trajectory(
                plotter,
                p_from=np.array([pp.x, pp.y, pp.z]),
                p_to  =np.array([p.x,  p.y,  p.z]),
                color ="orange",
                width =1,
                arrow =False,
            )
        # Líneas verticales: pre_place → place
        for pp, p in zip(pre_place, place):
            _add_trajectory(
                plotter,
                p_from=np.array([pp.x, pp.y, pp.z]),
                p_to  =np.array([p.x,  p.y,  p.z]),
                color ="#BB8FCE",
                width =1,
                arrow =False,
            )

    # ── Leyenda ───────────────────────────────────────────────────────────────
    legend_entries = [
        ["Bloques origen (pick)",      COLOR_PICK_TOWER],
        ["Bloques destino (place)",    COLOR_PLACE_TOWER],
    ]
    if show_trajectories:
        legend_entries.append(["Trayectoria pick→place", COLOR_TRAYECTORIA])
    if show_waypoints:
        legend_entries.append(["Pre-pick waypoints",     COLOR_PRE_PICK])
        legend_entries.append(["Pre-place waypoints",    COLOR_PRE_PLACE])

    plotter.add_legend(legend_entries, bcolor=(0.1, 0.1, 0.2), border=True,
                       size=(0.22, 0.18), loc="upper right")

    # ── Título ────────────────────────────────────────────────────────────────
    plotter.add_title(
        f"Torre Jenga – {pisos} pisos  |  {n_moves} movimientos",
        font_size=12, color="white",
    )

    # ── Cámara inicial ────────────────────────────────────────────────────────
    plotter.camera_position = [
        (0.8, -0.3, 0.9),   # posición cámara
        (-0.05, 0.45, 0.35), # punto focal
        (0, 0, 1),           # vector "arriba"
    ]

    print("\n[visualizer] Abriendo ventana 3D...")
    print("  Clic izquierdo + arrastrar → rotar")
    print("  Clic derecho   + arrastrar → zoom")
    print("  Clic medio     + arrastrar → pan")
    print("  Q / Escape                 → cerrar\n")

    plotter.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizador estático PyVista para torre Jenga + robot"
    )
    parser.add_argument(
        "--N", type=int, default=10,
        help="Número de pisos (par). Por defecto: 10"
    )
    parser.add_argument(
        "--no-waypoints", action="store_true",
        help="Ocultar waypoints pre_pick / pre_place"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Ocultar números de operación sobre los bloques"
    )
    parser.add_argument(
        "--no-trajectories", action="store_true",
        help="Ocultar líneas de trayectoria"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_scene(
        pisos             =args.N,
        show_waypoints    =not args.no_waypoints,
        show_labels       =not args.no_labels,
        show_trajectories =not args.no_trajectories,
    )
