import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- 1. Definir las constantes físicas y parámetros de realismo ---
mu_0 = 4 * np.pi * 1e-7  # Permeabilidad del espacio libre (T*m/A)

# Permeabilidad del material (1.0 para aire/vacío)
relative_permeability = 1.0  # Permeabilidad relativa (ej., 1.0 para aire, ~1000-100000 para hierro)
mu_material = mu_0 * relative_permeability # Permeabilidad del material efectivo

# Campo magnético externo (ej., campo terrestre en Teslas)
# Este es un vector (Bx, By, Bz) que se suma al campo de las bobinas.
# Ejemplo: Campo terrestre típico (~50 µT) en la dirección Z
earth_magnetic_field_vector = np.array([0.0, 0.0, 50e-6]) # En Teslas, 

# --- NUEVA CONSIDERACIÓN: Parámetros de la fuente de alimentación ---
source_voltage_dc = 12.0 # Voltaje DC máximo de la fuente (V)
source_current_limit_dc = 2 # Límite de corriente DC de la fuente (A)

# --- NUEVA CONSIDERACIÓN: Propiedades del cable de la bobina (asunciones para el cálculo de resistencia) ---
wire_material_resistivity = 1.68e-8 # Resistividad del cobre (Ohm*m a 20°C)
# Diámetro del cable (ej. AWG 22 = 0.643 mm)
wire_diameter = 0.643e-3 # m
wire_cross_section_area = np.pi * (wire_diameter / 2)**2 # m^2

# --- 2. Definir los parámetros de las bobinas (todas iguales) ---
N = 110       # Número total de espiras por bobina
R = 0.0565    # Radio de la espira (m)

# Longitud de cada bobina y número de segmentos para aproximación
coil_length = 0.035  # Longitud de cada bobina individual (m)
num_segments_per_coil = 110 # Número de bucles para simular la longitud de cada bobina

# Distancia de los centros de las bobinas desde el origen (m) para formar la cruz
offset_distance = 0.115 # Distancia de los centros de las bobinas al origen
set_distance = 0.075 # Rango de visualización del campo (ejes X e Y del plot)

# --- Cálculo de la Corriente (I) basada en la fuente y la resistencia ---
# 1. Longitud total del cable por bobina: Circunferencia * N espiras
wire_length_per_coil = N * (2 * np.pi * R)
# 2. Resistencia de una bobina: Resistencia = (resistividad * longitud) / área_transversal
resistance_per_coil = wire_material_resistivity * (wire_length_per_coil / wire_cross_section_area)
# 3. Resistencia total para 4 bobinas en serie
total_system_resistance = 4 * resistance_per_coil

# 4. Corriente Real que el sistema puede extraer de la fuente (Ley de Ohm, limitado por la fuente)
# La corriente será el mínimo entre la que la fuente puede suministrar (límite)
# y la que se obtiene por Ley de Ohm con el voltaje máximo de la fuente.
I = min(source_current_limit_dc, source_voltage_dc / total_system_resistance)

print(f"Propiedades calculadas del sistema de bobinas:")
print(f"  Longitud del cable por bobina: {wire_length_per_coil:.3f} m")
print(f"  Resistencia de una bobina: {resistance_per_coil:.2f} Ohm")
print(f"  Resistencia total del sistema (4 bobinas en serie): {total_system_resistance:.2f} Ohm")
print(f"  Voltaje máximo de la fuente: {source_voltage_dc:.1f} V")
print(f"  Límite de corriente de la fuente: {source_current_limit_dc:.1f} A")
print(f"  Corriente (I) usada en la simulación: {I:.4f} A") # Actualiza I con el valor calculado

# --- Altura Z para medir el campo ---
# Puedes cambiar este valor para visualizar el campo a diferentes alturas
z_height_to_measure = 0.0 # Altura en metros (0.0 es el centro de la cruz)

# --- 3. Función para calcular el vector de campo magnético de UN SOLO BUCLE ---
# Usa la aproximación del dipolo magnético.
def _calculate_single_loop_B_vector_approx(x_point, y_point, z_point, N_loop, R_loop, I_current, mu_eff, loop_center, loop_axis):
    """
    Calcula el vector de campo magnético aproximado de un solo bucle usando la aproximación del dipolo magnético.
    Retorna [Bx, By, Bz].
    """
    # Coordenadas del punto relativas al centro del bucle
    rx = x_point - loop_center[0]
    ry = y_point - loop_center[1]
    rz = z_point - loop_center[2]
    
    # Vector de posición desde el centro del bucle al punto
    r_vec = np.array([rx, ry, rz])
    
    # Manejar el caso especial donde el punto de medición es el centro del bucle
    # El campo de un dipolo es infinito en el centro, el campo real no.
    # Para la visualización, si r_mag es muy cercano a cero, usamos la fórmula del centro de la bobina.
    r_mag = np.linalg.norm(r_vec) # Magnitud del vector de posición
    
    if r_mag < 1e-9: # Usar un umbral pequeño para evitar división por cero
        # Campo en el centro de un bucle real (aproximación)
        B_center_mag = (mu_eff * N_loop * I_current) / (2 * R_loop)
        
        # Dirección del campo en el centro es a lo largo del eje del bucle
        if loop_axis == 0:  # Eje del bucle a lo largo del eje X global
            return np.array([B_center_mag, 0.0, 0.0])
        elif loop_axis == 1: # Eje del bucle a lo largo del eje Y global
            return np.array([0.0, B_center_mag, 0.0])
        else: # Eje del bucle a lo largo del eje Z global
            return np.array([0.0, 0.0, B_center_mag])

    # Calcular el momento dipolar magnético del bucle
    # Asumimos que el momento dipolar está alineado con el eje del bucle
    m_magnitude = N_loop * I_current * np.pi * R_loop**2 # Magnitud del momento dipolar
    
    m_vec = np.array([0.0, 0.0, 0.0])
    if loop_axis == 0:  # Eje del bucle a lo largo del eje X global
        m_vec = np.array([m_magnitude, 0.0, 0.0])
    elif loop_axis == 1: # Eje del bucle a lo largo del eje Y global
        m_vec = np.array([0.0, m_magnitude, 0.0])
    else: # Eje del bucle a lo largo del eje Z global
        m_vec = np.array([0.0, 0.0, m_magnitude])

    # Fórmula del campo magnético de un dipolo
    # B(r) = (mu / (4*pi)) * ( (3 * (m . r) * r) / |r|^5 - m / |r|^3 )
    
    # Calcular el término (m . r)
    m_dot_r = np.dot(m_vec, r_vec)
    
    # Calcular el campo magnético vectorial
    B_vec = (mu_eff / (4 * np.pi)) * \
            (3 * (m_dot_r / (r_mag**5)) * r_vec - m_vec / (r_mag**3))
    
    return B_vec


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
total_B_vector_origin = np.array([0.0, 0.0, 0.0]) # Inicializar el vector de campo en el origen

# Espiras por segmento
segment_n = N / num_segments_per_coil 
# Longitud de cada segmento
segment_length = coil_length / num_segments_per_coil 

# Calcular el campo en el origen por suma de segmentos de cada bobina
for xc, yc, zc, axis_orient in coil_configurations:
    for i in range(num_segments_per_coil):
        # Calcular la posición del centro de cada segmento de bucle
        # El centro de la bobina (xc,yc,zc) es el punto medio de su longitud
        if axis_orient == 0: # Eje X
            segment_offset_x = (i - (num_segments_per_coil - 1) / 2) * segment_length
            segment_center_i = (xc + segment_offset_x, yc, zc)
        elif axis_orient == 1: # Eje Y
            segment_offset_y = (i - (num_segments_per_coil - 1) / 2) * segment_length
            segment_center_i = (xc, yc + segment_offset_y, zc)
        else: # Eje Z
            segment_offset_z = (i - (num_segments_per_coil - 1) / 2) * segment_length
            segment_center_i = (xc, yc, zc + segment_offset_z)

        # Sumar la contribución vectorial de cada segmento al campo total en el origen (0,0,0)
        total_B_vector_origin += _calculate_single_loop_B_vector_approx(
            0, 0, 0, segment_n, R, I, mu_material, segment_center_i, axis_orient
        )

# Añadir el campo magnético externo al campo total en el origen
total_B_vector_origin += earth_magnetic_field_vector
B_origin_approx_magnitude = np.linalg.norm(total_B_vector_origin)

print(f"El campo magnético (magnitud aproximada) en el centro del arreglo (0,0,0) es: {B_origin_approx_magnitude:.4e} T")


# --- 5. Preparar datos para la gráfica 3D (Malla de magnitud del campo combinado) ---
# Definir el rango de visualización para X y Y usando set_distance
x_3d_range = np.linspace(-set_distance, set_distance, 40)
y_3d_range = np.linspace(-set_distance, set_distance, 40)

# Crear la malla 2D de puntos para el plano XY a la altura z_height_to_measure
X_surface, Y_surface = np.meshgrid(x_3d_range, y_3d_range)

# La coordenada Z para todos los puntos de la superficie es la altura especificada
Z_surface_fixed_height = np.full_like(X_surface, z_height_to_measure)


# Inicializar la malla total de vectores de campo para el arreglo en la altura especificada
# Esto almacenará los vectores [Bx, By, Bz] en cada punto de la malla
total_B_vectors_at_height = np.zeros(X_surface.shape + (3,)) # Shape (40, 40, 3)

# Sumar las contribuciones de cada una de las 4 bobinas (y sus segmentos) en cada punto de la malla
# Itera sobre cada punto de la malla (X_surface, Y_surface)
for i in range(X_surface.shape[0]):
    for j in range(X_surface.shape[1]):
        current_x = X_surface[i, j]
        current_y = Y_surface[i, j]
        current_z = Z_surface_fixed_height[i, j]

        B_total_at_point = np.array([0.0, 0.0, 0.0])

        for xc, yc, zc, axis_orient in coil_configurations:
            for k in range(num_segments_per_coil):
                if axis_orient == 0: # Eje X
                    segment_offset_x = (k - (num_segments_per_coil - 1) / 2) * segment_length
                    segment_center_k = (xc + segment_offset_x, yc, zc)
                elif axis_orient == 1: # Eje Y
                    segment_offset_y = (k - (num_segments_per_coil - 1) / 2) * segment_length
                    segment_center_k = (xc, yc + segment_offset_y, zc)
                else: # Eje Z
                    segment_offset_z = (k - (num_segments_per_coil - 1) / 2) * segment_length
                    segment_center_k = (xc, yc, zc + segment_offset_z)
                
                # Calcular el vector de campo de este segmento en el punto actual (current_x, current_y, current_z)
                B_segment_vector = _calculate_single_loop_B_vector_approx(
                    current_x, current_y, current_z, segment_n, R, I, mu_material, segment_center_k, axis_orient
                )
                B_total_at_point += B_segment_vector # Suma vectorial
        
        # Añadir el campo magnético externo al campo total del punto
        B_total_at_point += earth_magnetic_field_vector
        total_B_vectors_at_height[i, j, :] = B_total_at_point


# Calcular la magnitud final del campo en cada punto de la malla
total_B_magnitudes_at_height_uT = np.linalg.norm(total_B_vectors_at_height, axis=-1) * 1e6 # en microTeslas

# --- 6. Visualización 3D (Malla de colores con superficie suave y bobinas) ---
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Dibujar la superficie del campo magnético en el plano XY a la altura z_height_to_measure
# La altura de la superficie (eje Z del plot) es la magnitud del campo.
surf = ax.plot_surface(X_surface, Y_surface, total_B_magnitudes_at_height_uT,
                       cmap=cm.viridis,
                       rstride=1, cstride=1,
                       linewidth=0, antialiased=False, alpha=0.9)


# Dibujar las 4 bobinas en la gráfica 3D con sus orientaciones correctas
theta = np.linspace(0, 2*np.pi, 100)
for xc, yc, zc, axis_orient in coil_configurations:
    # Para dibujar la "bobina" de longitud finita, dibujamos el centro y los extremos.
    # Esta es una representación visual simple de la bobina.
    if axis_orient == 0: # Eje X
        coil_x_start = xc - coil_length / 2
        coil_x_end = xc + coil_length / 2
        
        # Dibujar los bucles extremos
        ax.plot(np.full_like(theta, coil_x_start), yc + R * np.cos(theta), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(np.full_like(theta, coil_x_end), yc + R * np.cos(theta), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        # Dibujar las líneas que conectan los bucles extremos
        ax.plot([coil_x_start, coil_x_end], [yc + R, yc + R], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start, coil_x_end], [yc - R, yc - R], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        # Las líneas que conectan los radios superior/inferior para una vista más tridimensional
        ax.plot([coil_x_start, coil_x_end], [yc, yc], [zc + R, zc + R], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start, coil_x_end], [yc, yc], [zc - R, zc - R], color='red', linewidth=1.5, alpha=0.8)

    elif axis_orient == 1: # Eje Y
        coil_y_start = yc - coil_length / 2
        coil_y_end = yc + coil_length / 2

        ax.plot(xc + R * np.cos(theta), np.full_like(theta, coil_y_start), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(xc + R * np.cos(theta), np.full_like(theta, coil_y_end), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        
        ax.plot([xc + R, xc + R], [coil_y_start, coil_y_end], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc - R, xc - R], [coil_y_start, coil_y_end], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [coil_y_start, coil_y_end], [zc + R, zc + R], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [coil_y_start, coil_y_end], [zc - R, zc - R], color='red', linewidth=1.5, alpha=0.8)

    else: # Eje Z (no usadas en esta configuración específica de cruz, pero se mantienen por completitud)
        coil_z_start = zc - coil_length / 2
        coil_z_end = zc + coil_length / 2

        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), np.full_like(theta, coil_z_start),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), np.full_like(theta, coil_z_end),
                color='red', linewidth=1.5, alpha=0.8)
        
        ax.plot([xc + R, xc + R], [yc, yc], [coil_z_start, coil_z_end], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc - R, xc - R], [yc, yc], [coil_z_start, coil_z_end], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [yc + R, yc + R], [coil_z_start, coil_z_end], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [yc - R, yc - R], [coil_z_start, coil_z_end], color='red', linewidth=1.5, alpha=0.8)


ax.set_title(f'Magnitud Aproximada del Campo Magnético (Plano Z = {z_height_to_measure:.2f} m)')
ax.set_xlabel('Eje X (m)')
ax.set_ylabel('Eje Y (m)')
ax.set_zlabel('Magnitud del Campo B (µT)') # El eje Z ahora representa la magnitud del campo
ax.set_aspect('auto') # Mantiene la proporción de los ejes para mejor visualización

# Añadir una barra de color para interpretar la magnitud del campo
m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(total_B_magnitudes_at_height_uT) # La barra de color se basa en la magnitud de la superficie
cbar = fig.colorbar(m, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Magnitud del Campo B (µT)')

# Ajustar los límites de los ejes X e Y para la visualización
ax.set_xlim([-set_distance, set_distance]) # Corregido: usar set_distance para el rango del plot
ax.set_ylim([-set_distance, set_distance]) # Corregido: usar set_distance para el rango del plot
# Ajustar los límites del eje Z (magnitud del campo) para que la superficie sea visible
ax.set_zlim([total_B_magnitudes_at_height_uT.min() * 0.9, total_B_magnitudes_at_height_uT.max() * 1.1])


plt.tight_layout()
plt.show()
