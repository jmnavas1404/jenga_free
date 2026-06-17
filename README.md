# jenga_free

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