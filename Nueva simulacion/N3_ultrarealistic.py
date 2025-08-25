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
earth_magnetic_field_vector = np.array([0.0, 0.0, 50e-6]) # En Teslas

# --- Parámetros de la fuente de alimentación ---
source_voltage_dc = 12.0 # Voltaje DC máximo de la fuente (V)
source_current_limit_dc = 2.0 # Límite de corriente DC de la fuente (A)

# --- Propiedades del cable de la bobina (asunciones para el cálculo de resistencia) ---
wire_material_resistivity = 1.68e-8 # Resistividad del cobre (Ohm*m a 20°C)
wire_diameter = 0.643e-3 # Diámetro del cable en metros (ej. AWG 22)
wire_cross_section_area = np.pi * (wire_diameter / 2)**2 # m^2

# NUEVA CONSIDERACIÓN: Parámetros térmicos del cobre y de la bobina
temperature_coefficient_resistivity = 0.0039 # Alfa para el cobre (1/°C)
initial_temperature = 20.0 # Temperatura ambiente inicial (°C)
ambient_temperature = 20.0 # Temperatura ambiente (°C)
# Resistencia térmica de la bobina al ambiente (ej. °C/W). Esto es una aproximación,
# depende del diseño, ventilación, etc. Valores típicos pueden variar mucho.
thermal_resistance_to_ambient = 5.0 # °C/W (valor de ejemplo, ajustar según diseño)

# --- 2. Definir los parámetros de las bobinas (todas iguales) ---
N = 110       # Número total de espiras por bobina
R = 0.0565    # Radio de la espira (m)

# Longitud de cada bobina y número de segmentos para aproximación
coil_length = 0.035  # Longitud de cada bobina individual (m)
# Aumentar este número para mayor precisión en Biot-Savart, a costa de rendimiento.
num_segments_per_coil = 1 # Número de bucles para simular la longitud de cada bobina

# Distancia de los centros de las bobinas desde el origen (m) para formar la cruz
offset_distance = 0.115 # Distancia de los centros de las bobinas al origen
set_distance = 0.075 # Rango de visualización del campo (ejes X e Y del plot)

# --- Cálculo de la Corriente (I) basada en la fuente y la resistencia (con temperatura) ---
# 1. Longitud total del cable por bobina: Circunferencia * N espiras
wire_length_per_coil = N * (2 * np.pi * R)
# 2. Resistencia de una bobina a temperatura inicial
resistance_per_coil_at_initial_temp = wire_material_resistivity * (wire_length_per_coil / wire_cross_section_area)
# 3. Resistencia total para 4 bobinas en serie a temperatura inicial
total_system_resistance_initial = 4 * resistance_per_coil_at_initial_temp

# Iterar para encontrar la corriente y temperatura de equilibrio (simplificado)
current_I = min(source_current_limit_dc, source_voltage_dc / total_system_resistance_initial)
coil_temperature = initial_temperature

# Pequeña iteración para acercarse a la temperatura de equilibrio (Estado Estacionario)
# En una simulación más avanzada, esto sería un modelo de transferencia de calor.
for _ in range(5): # Iterar unas pocas veces para converger un poco
    P_dissipated = current_I**2 * total_system_resistance_initial # Potencia disipada
    # Cambio de temperatura basado en la disipación de potencia
    delta_T = P_dissipated * thermal_resistance_to_ambient
    coil_temperature = ambient_temperature + delta_T
    
    # Actualizar resistencia con la nueva temperatura
    total_system_resistance_current = total_system_resistance_initial * (1 + temperature_coefficient_resistivity * (coil_temperature - initial_temperature))
    
    # Recalcular corriente con la nueva resistencia
    current_I = min(source_current_limit_dc, source_voltage_dc / total_system_resistance_current)

I = current_I # Corriente final usada en la simulación

print(f"Propiedades calculadas del sistema de bobinas:")
print(f"  Longitud del cable por bobina: {wire_length_per_coil:.3f} m")
print(f"  Resistencia de una bobina (inicial): {resistance_per_coil_at_initial_temp:.2f} Ohm")
print(f"  Resistencia total del sistema (4 bobinas en serie, final): {total_system_resistance_current:.2f} Ohm")
print(f"  Temperatura de operación estimada de la bobina: {coil_temperature:.2f} °C")
print(f"  Voltaje máximo de la fuente: {source_voltage_dc:.1f} V")
print(f"  Límite de corriente de la fuente: {source_current_limit_dc:.1f} A")
print(f"  Corriente (I) usada en la simulación: {I:.4f} A") # Actualiza I con el valor calculado

# --- Altura Z para medir el campo ---
# Puedes cambiar este valor para visualizar el campo a diferentes alturas
z_height_to_measure = 0.0 # Altura en metros (0.0 es el centro de la cruz)

# --- 3. Función para calcular el vector de campo magnético de UN SOLO SEGMENTO DE CABLE (Biot-Savart) ---
def _calculate_B_vector_from_wire_segment(point_coord, segment_start_coord, segment_end_coord, I_current, mu_eff):
    """
    Calcula el vector de campo magnético (dB) en 'point_coord' debido a un segmento de cable rectilíneo.
    Utiliza la Ley de Biot-Savart.
    point_coord, segment_start_coord, segment_end_coord son arrays de 3 elementos [x, y, z].
    Retorna [dBx, dBy, dBz].
    """
    r_prime = np.array(point_coord) - np.array(segment_start_coord) # Vector desde inicio del segmento al punto
    r_double_prime = np.array(point_coord) - np.array(segment_end_coord) # Vector desde fin del segmento al punto
    
    dl_vec = np.array(segment_end_coord) - np.array(segment_start_coord) # Vector del segmento de corriente
    dl_mag = np.linalg.norm(dl_vec) # Longitud del segmento

    # Evitar singularidades si el punto está exactamente en el segmento o muy cerca
    if np.linalg.norm(r_prime) < 1e-12 or np.linalg.norm(r_double_prime) < 1e-12:
        return np.array([0.0, 0.0, 0.0]) # Campo infinito, lo aproximamos a 0 para evitar errores.
                                        # En la realidad, esto significaría estar tocando el cable.

    # Fórmula para el campo de un segmento recto (aproximación, la integral es más compleja)
    # Sin embargo, para segmentos muy pequeños, es una buena aproximación de dBL.
    # En realidad, Biot-Savart es para dl x r_unit / r^2
    # Pero para un segmento finito recto, la integral da una forma específica.
    # Para segmentos infinitesimales, usamos dl x r / r^3
    
    r_avg = (np.linalg.norm(r_prime) + np.linalg.norm(r_double_prime)) / 2 # Distancia promedio
    if r_avg < 1e-9: # Protegemos contra división por cero o números muy pequeños
        return np.array([0.0, 0.0, 0.0])

    # Biot-Savart para un segmento infinitesimal: dB = (mu * I / 4*pi) * (dl x r_unit) / r^2
    # Esto es más preciso para sumar muchos pequeños dl.
    # Usamos el punto medio del segmento como el punto de aplicación de dl
    segment_midpoint = (np.array(segment_start_coord) + np.array(segment_end_coord)) / 2
    r_vec_to_midpoint = np.array(point_coord) - segment_midpoint
    r_mag_to_midpoint = np.linalg.norm(r_vec_to_midpoint)

    if r_mag_to_midpoint < 1e-12: # Si el punto de medición está exactamente en el segmento
        return np.array([0.0, 0.0, 0.0]) # Evitar singularidad

    dl_cross_r = np.cross(dl_vec, r_vec_to_midpoint)
    
    dB_vec = (mu_eff * I_current / (4 * np.pi)) * (dl_cross_r / (r_mag_to_midpoint**3))
    
    return dB_vec


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
total_B_vector_origin = np.array([0.0, 0.0, 0.0]) # Inicializar el vector de campo en el origen

# Espiras por segmento (del modelo de longitud de bobina)
segments_per_loop = 36 # Número de segmentos rectos para aproximar cada bucle circular (mayor = más preciso)
segment_angle = (2 * np.pi) / segments_per_loop
dl_loop_segment_length = 2 * R * np.sin(segment_angle / 2) # Longitud del segmento para el arco

# Calcular el campo en el origen por suma de segmentos de cada bobina
# Cada 'num_segments_per_coil' ahora significa la longitud de la bobina.
# N total de espiras, así que N_current_per_segment es I
for xc, yc, zc, axis_orient in coil_configurations:
    for loop_idx in range(N): # Para cada espira real
        # Calcular el centro Z de esta espira (para bobinas en Z) o similar para X/Y
        # Aquí, 'num_segments_per_coil' se usaba para la longitud de la bobina.
        # Ahora, cada espira de la bobina de longitud finita es un bucle.
        # coil_length / N es la separación entre espiras.
        loop_z_offset = (loop_idx - (N - 1) / 2) * (coil_length / N) # Offset a lo largo del eje de la bobina

        # El centro del bucle actual
        current_loop_center = [xc, yc, zc]
        if axis_orient == 0: # Eje X
            current_loop_center[0] += loop_z_offset # Desplazar a lo largo del eje X
        elif axis_orient == 1: # Eje Y
            current_loop_center[1] += loop_z_offset # Desplazar a lo largo del eje Y
        else: # Eje Z
            current_loop_center[2] += loop_z_offset # Desplazar a lo largo del eje Z


        for seg_idx in range(segments_per_loop): # Para cada segmento del bucle
            angle_start = seg_idx * segment_angle
            angle_end = (seg_idx + 1) * segment_angle

            # Coordenadas de los puntos de inicio y fin del segmento de cable en el bucle
            if axis_orient == 0: # Eje X
                # Bucle en el plano YZ
                seg_start = np.array([current_loop_center[0], current_loop_center[1] + R * np.cos(angle_start), current_loop_center[2] + R * np.sin(angle_start)])
                seg_end = np.array([current_loop_center[0], current_loop_center[1] + R * np.cos(angle_end), current_loop_center[2] + R * np.sin(angle_end)])
            elif axis_orient == 1: # Eje Y
                # Bucle en el plano XZ
                seg_start = np.array([current_loop_center[0] + R * np.cos(angle_start), current_loop_center[1], current_loop_center[2] + R * np.sin(angle_start)])
                seg_end = np.array([current_loop_center[0] + R * np.cos(angle_end), current_loop_center[1], current_loop_center[2] + R * np.sin(angle_end)])
            else: # Eje Z
                # Bucle en el plano XY (como en el ejemplo original)
                seg_start = np.array([current_loop_center[0] + R * np.cos(angle_start), current_loop_center[1] + R * np.sin(angle_start), current_loop_center[2]])
                seg_end = np.array([current_loop_center[0] + R * np.cos(angle_end), current_loop_center[1] + R * np.sin(angle_end), current_loop_center[2]])

            # Sumar la contribución vectorial de este segmento al campo total en el origen (0,0,0)
            total_B_vector_origin += _calculate_B_vector_from_wire_segment(
                np.array([0.0, 0.0, 0.0]), seg_start, seg_end, I, mu_material
            )

# Añadir el campo magnético externo al campo total en el origen
total_B_vector_origin += earth_magnetic_field_vector
B_origin_approx_magnitude = np.linalg.norm(total_B_vector_origin)

print(f"El campo magnético (magnitud) en el centro del arreglo (0,0,0) es: {B_origin_approx_magnitude:.4e} T")


# --- 5. Preparar datos para la gráfica 3D (Malla de magnitud del campo combinado) ---
# Definir el rango de visualización para X y Y usando set_distance
x_3d_range = np.linspace(-set_distance, set_distance, 40)
y_3d_range = np.linspace(-set_distance, set_distance, 40)

# Crear la malla 2D de puntos para el plano XY a la altura z_height_to_measure
X_surface, Y_surface = np.meshgrid(x_3d_range, y_3d_range)

# La coordenada Z para todos los puntos de la superficie es la altura especificada
Z_surface_fixed_height = np.full_like(X_surface, z_height_to_measure)


# Inicializar la malla total de vectores de campo para el arreglo en la altura especificada
total_B_vectors_at_height = np.zeros(X_surface.shape + (3,)) # Shape (40, 40, 3)

# Sumar las contribuciones de cada una de las 4 bobinas (y sus segmentos) en cada punto de la malla
for i in range(X_surface.shape[0]):
    for j in range(X_surface.shape[1]):
        current_point = np.array([X_surface[i, j], Y_surface[i, j], Z_surface_fixed_height[i, j]])
        B_total_at_point = np.array([0.0, 0.0, 0.0])

        for xc, yc, zc, axis_orient in coil_configurations:
            for loop_idx in range(N): # Para cada espira real
                loop_z_offset = (loop_idx - (N - 1) / 2) * (coil_length / N)

                current_loop_center = [xc, yc, zc]
                if axis_orient == 0: # Eje X
                    current_loop_center[0] += loop_z_offset
                elif axis_orient == 1: # Eje Y
                    current_loop_center[1] += loop_z_offset
                else: # Eje Z
                    current_loop_center[2] += loop_z_offset

                for seg_idx in range(segments_per_loop): # Para cada segmento del bucle
                    angle_start = seg_idx * segment_angle
                    angle_end = (seg_idx + 1) * segment_angle

                    # Coordenadas de los puntos de inicio y fin del segmento de cable en el bucle
                    if axis_orient == 0: # Eje X
                        seg_start = np.array([current_loop_center[0], current_loop_center[1] + R * np.cos(angle_start), current_loop_center[2] + R * np.sin(angle_start)])
                        seg_end = np.array([current_loop_center[0], current_loop_center[1] + R * np.cos(angle_end), current_loop_center[2] + R * np.sin(angle_end)])
                    elif axis_orient == 1: # Eje Y
                        seg_start = np.array([current_loop_center[0] + R * np.cos(angle_start), current_loop_center[1], current_loop_center[2] + R * np.sin(angle_start)])
                        seg_end = np.array([current_loop_center[0] + R * np.cos(angle_end), current_loop_center[1], current_loop_center[2] + R * np.sin(angle_end)])
                    else: # Eje Z
                        seg_start = np.array([current_loop_center[0] + R * np.cos(angle_start), current_loop_center[1] + R * np.sin(angle_start), current_loop_center[2]])
                        seg_end = np.array([current_loop_center[0] + R * np.cos(angle_end), current_loop_center[1] + R * np.sin(angle_end), current_loop_center[2]])
                    
                    B_total_at_point += _calculate_B_vector_from_wire_segment(current_point, seg_start, seg_end, I, mu_material)
        
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
    # Para dibujar cada bobina, dibujamos un contorno representativo
    # Esto sigue siendo una simplificación visual de la bobina de longitud finita.
    # En la vida real, es una espiral densa de hilo.
    
    if axis_orient == 0: # Eje X
        # Dibuja la espira inicial y final de la bobina y las líneas que las unen
        # La bobina se extiende desde (xc - coil_length/2) hasta (xc + coil_length/2)
        coil_x_start_vis = xc - coil_length / 2
        coil_x_end_vis = xc + coil_length / 2

        ax.plot(np.full_like(theta, coil_x_start_vis), yc + R * np.cos(theta), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(np.full_like(theta, coil_x_end_vis), yc + R * np.cos(theta), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start_vis, coil_x_end_vis], [yc + R, yc + R], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start_vis, coil_x_end_vis], [yc - R, yc - R], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start_vis, coil_x_end_vis], [yc, yc], [zc + R, zc + R], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([coil_x_start_vis, coil_x_end_vis], [yc, yc], [zc - R, zc - R], color='red', linewidth=1.5, alpha=0.8)

    elif axis_orient == 1: # Eje Y
        coil_y_start_vis = yc - coil_length / 2
        coil_y_end_vis = yc + coil_length / 2

        ax.plot(xc + R * np.cos(theta), np.full_like(theta, coil_y_start_vis), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(xc + R * np.cos(theta), np.full_like(theta, coil_y_end_vis), zc + R * np.sin(theta),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc + R, xc + R], [coil_y_start_vis, coil_y_end_vis], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc - R, xc - R], [coil_y_start_vis, coil_y_end_vis], [zc, zc], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [coil_y_start_vis, coil_y_end_vis], [zc + R, zc + R], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [coil_y_start_vis, coil_y_end_vis], [zc - R, zc - R], color='red', linewidth=1.5, alpha=0.8)

    else: # Eje Z (No se usan en esta configuración de cruz, pero se mantienen por completitud)
        coil_z_start_vis = zc - coil_length / 2
        coil_z_end_vis = zc + coil_length / 2

        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), np.full_like(theta, coil_z_start_vis),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), np.full_like(theta, coil_z_end_vis),
                color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc + R, xc + R], [yc, yc], [coil_z_start_vis, coil_z_end_vis], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc - R, xc - R], [yc, yc], [coil_z_start_vis, coil_z_end_vis], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [yc + R, yc + R], [coil_z_start_vis, coil_z_end_vis], color='red', linewidth=1.5, alpha=0.8)
        ax.plot([xc, xc], [yc - R, yc - R], [coil_z_start_vis, coil_z_end_vis], color='red', linewidth=1.5, alpha=0.8)


ax.set_title(f'Magnitud del Campo Magnético (Plano Z = {z_height_to_measure:.2f} m) - Aproximación Numérica')
ax.set_xlabel('Eje X (m)')
ax.set_ylabel('Eje Y (m)')
ax.set_zlabel('Magnitud del Campo B (µT)')
ax.set_aspect('auto')

m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(total_B_magnitudes_at_height_uT)
cbar = fig.colorbar(m, ax=ax, pad=0.1, shrink=0.5)
cbar.set_label('Magnitud del Campo B (µT)')
   
ax.set_xlim([-set_distance, set_distance])
ax.set_ylim([-set_distance, set_distance])
ax.set_zlim([total_B_magnitudes_at_height_uT.min() * 0.9, total_B_magnitudes_at_height_uT.max() * 1.1])

plt.tight_layout()
plt.show()
