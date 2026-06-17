import argparse
import numpy as np
import pyvista as pv
from math import cos, sin, radians

from pose_engine import (
    generar_vectores,
    pick, place,
    pre_pick, pre_place,
    pre_pre_pick, pre_pre_place,
    h,   # altura del bloque = 0.022 (de program_yu_prime)
)

# ── Dimensiones físicas del bloque Jenga (metros) ────────────────────────────
# Dimensiones reales del bloque Jenga estándar
BLOQUE_L = 0.025   # largo  (eje Y cuando dr=0°, eje X cuando dr=90°)
BLOQUE_W = 0.075   # ancho
BLOQUE_H = h       # alto = 0.022 m  (fuente: program_yu_prime)

# ── Parámetros de la torre visual — FUENTE DE VERDAD: visualization.txt ──────
# Centro base de la primera ficha (nivel 0, ficha 0) en create_zeugs:
#   zeug([-0.3093 + xg, 0.4304 + yg, 0.51 + zg], ...)
# Con c=0 al inicio: xg=0, yg=0 → centro de referencia = [-0.3093, 0.4304, 0.51]
TORRE_ORIGEN_X  = 0 #-0.3093
TORRE_ORIGEN_Y  = 0 # 0.4304
TORRE_ORIGEN_Z  = 0 # 0.51      # altura Z de la primera ficha

TOL             =  0.041     # separación entre fichas (tol en create_zeugs)
ZG_PASO         =  0.0255    # incremento Z por nivel  (zg += 0.0255)

# Torre destino: desplazada +dx en X respecto a la torre origen
# dx = 0.25 (de program_yu_prime / pose_engine)
from pose_engine import dx as DX_TORRES

TORRE_DESTINO_X = TORRE_ORIGEN_X + DX_TORRES
TORRE_DESTINO_Y = TORRE_ORIGEN_Y
TORRE_DESTINO_Z = TORRE_ORIGEN_Z

# ── Paleta de colores ─────────────────────────────────────────────────────────
COLOR_PICK_TOWER    = "#AED6F1"   # azul claro  – bloques en torre origen
COLOR_PLACE_TOWER   = "#A9DFBF"   # verde claro – bloques en torre destino
COLOR_TRAYECTORIA   = "#E74C3C"   # rojo        – líneas pick → place
COLOR_PRE_PICK      = "#F0B27A"   # naranja      – puntos pre-pick
COLOR_PRE_PLACE     = "#BB8FCE"   # morado       – puntos pre-place
COLOR_TABLA         = "#D5D8DC"   # gris claro   – mesa


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
    half = np.array([BLOQUE_L / 2, BLOQUE_W / 2, BLOQUE_H / 2])
    corners_local = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=float) * half

    R = _rotation_matrix_z(yaw)
    corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])

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
            text_color="white",
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
    """
    Mesa según visualization.txt:
        table(position=[0, 0.60, 0.04], rotation=[0, 0, radians(90)])
    Se representa como una caja plana centrada en esa posición.
    """
    # Dimensiones aproximadas de una mesa de laboratorio estándar
    MESA_L  = 1.1    # largo  (eje X tras rotación 90°)
    MESA_W  = 0.6    # ancho  (eje Y tras rotación 90°)
    MESA_H  = 0.01   # grosor

    cx, cy, cz = 0.0, 0.60, 0.04
    bounds = (
        cx - MESA_L / 2, cx + MESA_L / 2,
        cy - MESA_W / 2, cy + MESA_W / 2,
        cz - MESA_H / 2, cz + MESA_H / 2,
    )
    table = pv.Box(bounds=bounds)
    plotter.add_mesh(table, color=COLOR_TABLA, opacity=0.4)


def _build_tower_blocks(plotter: pv.Plotter, pisos: int,
                        cx0: float, cy0: float, cz0: float,
                        color: str, show_labels: bool,
                        label_offset: int = 0):
    """
    Construye la torre Jenga según la lógica de create_zeugs de visualization.txt.

    La torre alterna orientación cada nivel:
      c=0 (nivel par)  → fichas alineadas en Y (dr=0°,   90°, 180°, ...)
      c=1 (nivel impar) → fichas alineadas en X (dr=90°, 180°, ...)

    Parámetros
    ----------
    cx0, cy0, cz0 : centro de la primera ficha del nivel 0
    label_offset  : offset para el número de etiqueta (0 para origen, n para destino)
    """
    c_local = 0
    dr      = 0
    zg      = 0

    # Índice global de bloque para etiquetas
    bloque_idx = label_offset

    for i in range(pisos):
        xg =  0.0
        yg = -TOL   # valor inicial de yg antes del bucle de j

        for j in range(3):
            # Lógica EXACTA de create_zeugs:
            if c_local == 0 or c_local == 2:
                yg = 0.0
            else:
                xg = -TOL

            bx = cx0 + xg
            by = cy0 + yg
            bz = cz0 + zg

            label = str(bloque_idx) if show_labels else None
            _add_block(
                plotter,
                cx=bx, cy=by, cz=bz,
                yaw=radians(dr),
                color=color,
                opacity=0.85,
                label=label,
            )
            bloque_idx += 1

            # Actualizar xg/yg según lógica de create_zeugs
            if c_local == 0 or c_local == 2:
                xg = xg - TOL
                if c_local == 2:
                    c_local = 0

            yg = yg + TOL

        c_local += 1
        zg      += ZG_PASO
        dr      += 90


def build_scene(pisos: int, show_waypoints: bool, show_labels: bool,
                show_trajectories: bool):
    """
    Construye y muestra la escena completa.

    La escena integra dos fuentes de verdad:
      1. visualization.txt  → posición y orientación visual de los bloques
      2. program_yu_prime   → poses del robot (via pose_engine)

    Parámetros
    ----------
    pisos             : número de pisos de la torre
    show_waypoints    : si True, muestra pre_pick y pre_place
    show_labels       : si True, muestra número de bloque sobre cada uno
    show_trajectories : si True, dibuja líneas pick → place
    """
    generar_vectores(pisos=pisos)

    n_moves = len(pick)   # total movimientos = 2 * 3 * pisos

    plotter = pv.Plotter(window_size=[1400, 800])
    plotter.set_background("#1A1A2E")

    # ── Mesa ──────────────────────────────────────────────────────────────────
    _add_table(plotter)

    # ── Ejes del robot en el origen ───────────────────────────────────────────
    plotter.add_axes_at_origin(line_width=3, labels_off=False)

    # ── Torre ORIGEN (geometría de create_zeugs) ──────────────────────────────
    print(f"\n[visualizer] Construyendo torre origen ({pisos} pisos)...")
    _build_tower_blocks(
        plotter, pisos,
        cx0=TORRE_ORIGEN_X, cy0=TORRE_ORIGEN_Y, cz0=TORRE_ORIGEN_Z,
        color=COLOR_PICK_TOWER,
        show_labels=show_labels,
        label_offset=0,
    )

    # ── Torre DESTINO (misma geometría, desplazada +DX en X) ─────────────────
    print(f"[visualizer] Construyendo torre destino ({pisos} pisos)...")
    _build_tower_blocks(
        plotter, pisos,
        cx0=TORRE_DESTINO_X, cy0=TORRE_DESTINO_Y, cz0=TORRE_DESTINO_Z,
        color=COLOR_PLACE_TOWER,
        show_labels=show_labels,
        label_offset=pisos * 3,
    )

    # ── Poses del robot (pick / place de pose_engine) ─────────────────────────
    # Se muestran como puntos sobre las fichas para verificar alineación
    if show_waypoints:
        print("[visualizer] Dibujando poses del robot...")

        # Puntos pick (agarre real del robot sobre la torre origen)
        _add_waypoints(plotter, pick,  "#FFFFFF", size=8)     # blanco
        # Puntos place (depósito real del robot sobre la torre destino)
        _add_waypoints(plotter, place, "#FFD700", size=8)     # dorado

        # Pre-pick y pre-place
        _add_waypoints(plotter, pre_pick,  COLOR_PRE_PICK,  size=5)
        _add_waypoints(plotter, pre_place, COLOR_PRE_PLACE, size=5)

        # Líneas verticales: pre_pick → pick
        for pp, p in zip(pre_pick, pick):
            _add_trajectory(
                plotter,
                p_from=np.array([pp.x, pp.y, pp.z]),
                p_to  =np.array([p.x,  p.y,  p.z]),
                color ="orange", width=1, arrow=False,
            )
        # Líneas verticales: pre_place → place
        for pp, p in zip(pre_place, place):
            _add_trajectory(
                plotter,
                p_from=np.array([pp.x, pp.y, pp.z]),
                p_to  =np.array([p.x,  p.y,  p.z]),
                color ="#BB8FCE", width=1, arrow=False,
            )

    # ── Trayectorias pick → place ─────────────────────────────────────────────
    if show_trajectories:
        print("[visualizer] Dibujando trayectorias pick→place...")
        for p_pick, p_place in zip(pick, place):
            _add_trajectory(
                plotter,
                p_from=np.array([p_pick.x,  p_pick.y,  p_pick.z]),
                p_to  =np.array([p_place.x, p_place.y, p_place.z]),
                color =COLOR_TRAYECTORIA, width=1, arrow=True,
            )

    # ── Leyenda ───────────────────────────────────────────────────────────────
    legend_entries = [
        ["Torre origen  (bloques)",        COLOR_PICK_TOWER],
        ["Torre destino (bloques)",        COLOR_PLACE_TOWER],
    ]
    if show_waypoints:
        legend_entries += [
            ["Pose pick del robot",        "#FFFFFF"],
            ["Pose place del robot",       "#FFD700"],
            ["Pre-pick waypoints",         COLOR_PRE_PICK],
            ["Pre-place waypoints",        COLOR_PRE_PLACE],
        ]
    if show_trajectories:
        legend_entries.append(["Trayectoria pick→place", COLOR_TRAYECTORIA])

    plotter.add_legend(legend_entries, bcolor=(0.1, 0.1, 0.2), border=True,
                       size=(0.26, 0.22), loc="upper right")

    # ── Título ────────────────────────────────────────────────────────────────
    plotter.add_title(
        f"Torre Jenga – {pisos} pisos  |  {n_moves} movimientos",
        font_size=12, color="white",
    )

    # ── Cámara inicial ────────────────────────────────────────────────────────
    plotter.camera_position = [
        (0.8, -0.3, 0.9),    # posición cámara
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
        "--N", type=int, default=6,
        help="Número de pisos (par). Por defecto: 6"
    )
    parser.add_argument(
        "--no-waypoints", action="store_true",
        help="Ocultar puntos de pose del robot"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Ocultar números de bloque"
    )
    parser.add_argument(
        "--no-trajectories", action="store_true",
        help="Ocultar líneas de trayectoria pick→place"
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