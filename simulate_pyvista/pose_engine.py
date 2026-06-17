"""
pose_engine.py
==============
Extrae la lógica de generación de poses de program_yu_prime.py
SIN depender de voraus_robot_arm ni de simulación real.

Parámetros geométricos tomados EXACTAMENTE de program_yu_prime.py (fuente de verdad).

Uso desde program_yu_prime.py (modificación mínima):
    from pose_engine import generar_vectores, pick, place, pre_pick, pre_place, ...

Uso desde visualizer.py:
    from pose_engine import generar_vectores, pick, place, ...
    generar_vectores(pisos=6)
"""

from math import radians
import numpy as np

# ── Parámetros geométricos — FUENTE DE VERDAD: program_yu_prime.py ────────────
# Posiciones X/Y de los puntos de agarre en la torre origen
x1 = float(-0.2214)
x2 = x1                  # x2 == x1 en program_yu_prime

y1 = float(0.5294)
y2 = float(0.4541)

# Altura base de la torre (6 niveles)
z0 = float(0.1270)

# Alturas de aproximación
h_slow  = 0.05
h_fast  = 0.1

# Altura de cada bloque Jenga (metros)
h = 0.022

# Tolerancia en place (offset Z al depositar)
tol_place = 0.005

# Desplazamiento X entre torre origen y torre destino
dx = 0.25

# Orientación del TCP (igual que en program_yu_prime)
rx = float(180)
ry = float(0)
rz = float(98.19)

# Altura inicial de la torre destino (linux offset, de program_yu_prime)
HO_INICIAL = -0.0026

# ── Listas de poses (se rellenan al llamar generar_vectores) ──────────────────
pre_pre_pick  = []
pre_pick      = []
pick          = []

pre_pre_place = []
pre_place     = []
place         = []


# ── Clase mínima que imita CartesianPose de voraus ───────────────────────────
class CartesianPose:
    """Sustituye voraus_robot_arm.CartesianPose para uso offline."""
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.x     = x
        self.y     = y
        self.z     = z
        self.roll  = roll
        self.pitch = pitch
        self.yaw   = yaw

    def __repr__(self):
        return (f"CartesianPose(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, "
                f"yaw={np.degrees(self.yaw):.1f}°)")

    def as_array(self):
        return np.array([self.x, self.y, self.z])


# ── Funciones internas — lógica IDÉNTICA a program_yu_prime.py ───────────────

def _crear_puntos_piso(dz, m, new_p1, new_p2):
    """
    Calcula los 6 puntos de un piso:
      p1 → bloque izquierdo torre origen
      p2 → bloque derecho  torre origen
      p3 → bloque central  torre origen
      p4, p5, p6 → equivalentes en torre destino (piso siguiente hacia abajo)

    Reproducción exacta del cálculo de program_yu_prime.py.
    """
    p1 = np.array([x1, y1, z0 + dz]) * m - (m - 1) * (new_p1 + np.array([dx, 0, h + dz]))
    p2 = np.array([x2, y2, z0 + dz]) * m - (m - 1) * (new_p2 + np.array([dx, 0, h + dz]))
    p3 = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, z0 + dz])

    # Punto base del piso siguiente (un bloque más abajo)
    p5  = np.array([p3[0], p3[1], p3[2] - h])
    d23 = np.linalg.norm(p2 - p3)
    d13 = np.linalg.norm(p1 - p3)

    p45 = d23 * np.cross((p3 - p2), np.array([0, 0, -1])) / (
        np.linalg.norm(np.cross((p3 - p2), np.array([0, 0, -1])))
    )
    p4 = np.array([p5[0] + p45[0], p5[1] + p45[1], p5[2]])

    p65 = d13 * np.cross((p3 - p2), np.array([0, 0, 1])) / (
        np.linalg.norm(np.cross((p3 - p2), np.array([0, 0, 1])))
    )
    p6 = np.array([p5[0] + p65[0], p5[1] + p65[1], p5[2]])

    return p1, p2, p3, p4, p5, p6


def _agregar_poses(points, rz_offsets, m, ho, c):
    """
    Añade las CartesianPose de un piso a las listas globales.
    Reprodución exacta del bucle interno de program_yu_prime.py.
    """
    for p, rz_off in zip(points, rz_offsets):
        rz_ = rz + rz_off
        if rz_ == 180 or rz_ == 360:
            rz_ = 0

        # --- Poses en torre ORIGEN (pick) ---
        pre_pre_pick.append(CartesianPose(
            p[0], p[1], p[2] + h_fast,
            radians(rx), radians(ry), radians(rz_)
        ))
        pre_pick.append(CartesianPose(
            p[0], p[1], p[2] + h_slow,
            radians(rx), radians(ry), radians(rz_)
        ))
        pick.append(CartesianPose(
            p[0], p[1], p[2],
            radians(rx), radians(ry), radians(rz_)
        ))

        # --- Poses en torre DESTINO (place) ---
        # Fórmula exacta de program_yu_prime:
        # (p[0]+dx)*m - (m-1)*(p[0]-dx)
        # m=1 → place_x = p[0]+dx
        # m=0 → place_x = p[0]-dx  (viaje de vuelta)
        place_x = (p[0] + dx) * m - (m - 1) * (p[0] - dx)

        pre_pre_place.append(CartesianPose(
            place_x, p[1], ho + h_fast,
            radians(rx), radians(ry), radians(rz_)
        ))
        pre_place.append(CartesianPose(
            place_x, p[1], ho + h_slow,
            radians(rx), radians(ry), radians(rz_)
        ))
        place.append(CartesianPose(
            place_x, p[1], ho + tol_place,
            radians(rx), radians(ry), radians(rz_)
        ))

        c += 1
        if c % 3 == 0:
            ho += h

    return ho, c


# ── API pública ───────────────────────────────────────────────────────────────

def generar_vectores(pisos: int = 6):
    """
    Rellena las listas globales pick[], place[], pre_pick[], etc.

    Reproduce EXACTAMENTE el bucle de generación de program_yu_prime.py:
      - dz se reinicia a 0 al inicio de cada k
      - ho se reinicia a HO_INICIAL al inicio de cada k
      - m empieza en 1 y decrementa a 0 tras el primer k

    Parámetros
    ----------
    pisos : int
        Número de pisos de la torre Jenga (debe ser par). Por defecto 6.
    """
    # Limpiar listas por si se llama varias veces
    for lst in [pre_pre_pick, pre_pick, pick, pre_pre_place, pre_place, place]:
        lst.clear()

    new_p1 = 0
    new_p2 = 0
    m      = 1

    for k in range(2):          # k=0: origen→destino ; k=1: destino→origen
        dz = 0
        ho = HO_INICIAL         # ← se reinicia en cada k (crítico)
        c  = 0                  # contador interno para incremento de ho

        for i in range(int(pisos / 2)):
            p1, p2, p3, p4, p5, p6 = _crear_puntos_piso(dz, m, new_p1, new_p2)

            # Solo en el primer piso del primer viaje se guardan new_p1/new_p2
            if i == 0 and k == 0:
                new_p1 = p4
                new_p2 = p6

            points     = [p1, p3, p2, p4, p5, p6]
            rz_offsets = [0,  0,  0, -90, -90, -90]

            ho, c = _agregar_poses(points, rz_offsets, m, ho, c)

            dz = dz - 2 * h - 0.002   # bajar un piso (bloque + gap)

        m = m - 1   # k=0 → m=1 ; k=1 → m=0