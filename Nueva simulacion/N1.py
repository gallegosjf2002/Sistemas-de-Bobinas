import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- 1. Definir las constantes físicas ---
mu_0 = 4 * np.pi * 1e-7  # Permeabilidad del espacio libre (T*m/A)

# --- 2. Definir los parámetros de las bobinas (todas iguales) ---
N = 110       # Número de espiras
R = 0.0565      # Radio de la espira (m)
I = 2     # Corriente (A)
# Distancia de los centros de las bobinas desde el origen (m) para formar la cruz
offset_distance = 0.115
set_distance = 0.075

# --- NUEVA VARIABLE: Altura Z para medir el campo ---
# Puedes cambiar este valor para visualizar el campo a diferentes alturas
z_height_to_measure = 0.0 # Altura en metros (0.0 es el centro de la cruz)

# --- 3. Definir la función para la magnitud aproximada del campo de una sola bobina con eje arbitrario ---
def calculate_approx_B_magnitude(x_point, y_point, z_point, N_espiras, R_radio, I_corriente, mu_0_const, coil_center, coil_axis):
    """
    Aproxima la magnitud del campo magnético de una bobina circular con su eje a lo largo
    de 'coil_axis' (0 for X, 1 for Y, 2 for Z), centrada en 'coil_center'.

    Args:
        x_point, y_point, z_point (float or numpy array): Coordenadas del punto en el espacio global.
        N_espiras, R_radio, I_corriente, mu_0_const: Parámetros de la bobina.
        coil_center (tuple): (xc, yc, zc) del centro de la bobina.
        coil_axis (int): 0 para eje X, 1 para eje Y, 2 para eje Z.

    Returns:
        float or numpy array: Magnitud aproximada del campo magnético en Teslas (T).
    """
    # Coordenadas del punto relativas al centro de la bobina
    rx = x_point - coil_center[0]
    ry = y_point - coil_center[1]
    rz = z_point - coil_center[2]

    # Distancia axial y perpendicular relativas a la orientación de la bobina
    if coil_axis == 0:  # Eje de la bobina a lo largo del eje X global
        axial_dist_sq = rx**2
        r_perpendicular_sq = ry**2 + rz**2
    elif coil_axis == 1: # Eje de la bobina a lo largo del eje Y global
        axial_dist_sq = ry**2
        r_perpendicular_sq = rx**2 + rz**2
    else: # Eje de la bobina a lo largo del eje Z global (como en el ejemplo original)
        axial_dist_sq = rz**2
        r_perpendicular_sq = rx**2 + ry**2

    # Esta fórmula es una generalización de la fórmula axial para una visualización 3D.
    # El término (R_radio**2 + axial_dist_sq + r_perpendicular_sq) es una aproximación
    # de la distancia efectiva al cuadrado desde la "fuente" de la bobina al punto en 3D.
    B_magnitude = (mu_0_const * N_espiras * I_corriente * R_radio**2) / \
                  (2 * (R_radio**2 + axial_dist_sq + r_perpendicular_sq)**(3/2))
    return B_magnitude

# --- 4. Definir las configuraciones de las 4 bobinas en forma de cruz ---
# Cada tupla contiene (center_x, center_y, center_z, axis_orientation)
# axis_orientation: 0 para Eje X, 1 para Eje Y, 2 para Eje Z
coil_configurations = [
    (offset_distance, 0, 0, 0),    # Bobina 1: Eje X, centrada en (+X, 0, 0)
    (-offset_distance, 0, 0, 0),   # Bobina 2: Eje X, centrada en (-X, 0, 0)
    (0, offset_distance, 0, 1),    # Bobina 3: Eje Y, centrada en (0, +Y, 0)
    (0, -offset_distance, 0, 1)    # Bobina 4: Eje Y, centrada en (0, -Y, 0)
]

# --- Calcular el campo magnético (magnitud aproximada) en el centro del arreglo (0,0,0) ---
# Este cálculo es para el punto (0,0,0) fijo, no afectado por z_height_to_measure
B_origin_approx = 0.0 # Inicializar la magnitud del campo en el origen

for xc, yc, zc, axis_orient in coil_configurations:
    B_origin_approx += calculate_approx_B_magnitude(0, 0, 0, N, R, I, mu_0, (xc, yc, zc), axis_orient)

print(f"El campo magnético (magnitud aproximada) en el centro del arreglo (0,0,0) es: {B_origin_approx:.4e} T")


# --- 5. Preparar datos para la gráfica 3D (Malla de magnitud del campo combinado) ---
x_3d_range = np.linspace(-set_distance, set_distance, 40)
y_3d_range = np.linspace(-set_distance, set_distance, 40)

# Crear la malla 2D de puntos para el plano XY a la altura z_height_to_measure
X_surface, Y_surface = np.meshgrid(x_3d_range, y_3d_range)

# La coordenada Z para todos los puntos de la superficie es la altura especificada
Z_surface_fixed_height = np.full_like(X_surface, z_height_to_measure)


# Inicializar la malla total de magnitud de campo para el arreglo en la altura especificada
total_B_magnitudes_at_height = np.zeros_like(X_surface)

# Sumar las contribuciones de cada una de las 4 bobinas en cada punto de la malla
for xc, yc, zc, axis_orient in coil_configurations:
    # Calcular la magnitud aproximada del campo de esta bobina en el plano a la altura Z_surface_fixed_height
    total_B_magnitudes_at_height += calculate_approx_B_magnitude(
        X_surface, Y_surface, Z_surface_fixed_height, N, R, I, mu_0, (xc, yc, zc), axis_orient
    )

# Multiplicar por 1e6 para mostrar los valores en microTeslas (µT)
total_B_magnitudes_at_height_uT = total_B_magnitudes_at_height * 1e6

# --- 6. Visualización 3D (Malla de colores con superficie suave y bobinas) ---
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Dibujar la superficie del campo magnético en el plano XY a la altura z_height_to_measure
# La altura de la superficie (eje Z del plot) es la magnitud del campo.
surf = ax.plot_surface(X_surface, Y_surface, total_B_magnitudes_at_height_uT,
                       cmap=cm.viridis,
                       rstride=1, cstride=1,
                       linewidth=0, antialiased=False, alpha=0.9)



ax.set_title(f'Magnitud Aproximada del Campo Magnético (Plano Z = {z_height_to_measure:.2f} m)')
ax.set_xlabel('Eje X (m)')
ax.set_ylabel('Eje Y (m)')
ax.set_zlabel('Magnitud del Campo B (µT)') # El eje Z ahora representa la magnitud del campo
ax.set_aspect('auto') # Mantiene la proporción de los ejes

# Añadir una barra de color para interpretar la magnitud del campo
m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(total_B_magnitudes_at_height_uT) # La barra de color se basa en la magnitud de la superficie
cbar = fig.colorbar(m, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Magnitud del Campo B (µT)')

# Ajustar los límites de los ejes X e Y para la visualización
ax.set_xlim([-offset_distance, offset_distance])
ax.set_ylim([-offset_distance, offset_distance])
# Ajustar los límites del eje Z (magnitud del campo) para que la superficie sea visible
ax.set_zlim([total_B_magnitudes_at_height_uT.min() * 0.9, total_B_magnitudes_at_height_uT.max() * 1.1])


plt.tight_layout()
plt.show()
