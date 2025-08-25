# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe # Usado para el cálculo correcto de B_z
from mpl_toolkits.mplot3d import Axes3D

# Definir parámetros
N = 110 # Número de vueltas
R = 0.0565 # Radio de las bobinas (m)
I =  0.002   # Corriente (A) - Considera si este valor es apropiado; es bastante pequeño.
d = 0.15/2 # Separación de las bobinas (m) (interno (15 cm)) externo (23 cm)
mu_0 = 4 * np.pi * 1e-7 # Permeabilidad magnética del vacío

# Definir función para calcular la componente z del campo magnético de una bobina
def B_z_corrected(x_obs, y_obs, z_obs, x_coil, y_coil, z_coil, R, N, I, mu_0):
    """
    Calcula la componente z del campo magnético de una bobina circular en un punto arbitrario.
    La bobina se asume que está centrada en (x_coil, y_coil, z_coil).

    Args:
        x_obs, y_obs, z_obs: Coordenadas del punto de observación.
        x_coil, y_coil, z_coil: Coordenadas del centro de la bobina.
        R: Radio de la bobina.
        N: Número de vueltas.
        I: Corriente en la bobina.
        mu_0: Permeabilidad magnética del vacío.

    Returns:
        Componente z del campo magnético en el punto de observación.
    """
    # Coordenadas relativas al centro de la bobina
    rho = np.sqrt((x_obs - x_coil)**2 + (y_obs - y_coil)**2) # Distancia radial desde el eje de la bobina
    z_rel = z_obs - z_coil # Distancia axial desde el plano de la bobina

    if np.isclose(rho, 0): # Caso on-axis para evitar división por cero
        Bz = (mu_0 * N * I * R**2) / (2 * (R**2 + z_rel**2)**(3/2))
    else:
        # Parámetros para las integrales elípticas
        k_sq = (4 * R * rho) / ((R + rho)**2 + z_rel**2)
        k_sq = np.clip(k_sq, 0, 1.0) # Asegura que k_sq esté entre 0 y 1
        
        K_val = ellipk(k_sq)
        E_val = ellipe(k_sq)

        denominator_sqrt = np.sqrt((R + rho)**2 + z_rel**2)
        denominator_term = (R - rho)**2 + z_rel**2

        if np.isclose(denominator_term, 0): # Punto directamente en el cable (singularidad)
            return np.nan # O elige un manejo adecuado (ej., un valor muy grande o interpolación)
        
        Bz = (mu_0 * N * I / (2 * np.pi * denominator_sqrt)) * \
             (K_val + (R**2 - rho**2 - z_rel**2) / denominator_term * E_val)
    
    return Bz

# Crear mallas de puntos para el cálculo en 3D
x_points = np.linspace(-0.2, 0.2, 50) # Rango de -0.2 m a 0.2 m
y_points = np.linspace(-0.2, 0.2, 50)
X, Y = np.meshgrid(x_points, y_points)
Z_obs_plane = np.zeros_like(X) # Campo magnético en el plano z=0 (donde se observa el campo)

# Calcular el campo magnético total en el plano z=0
B = np.zeros_like(X) # Inicializar el campo total

# Coordenadas de las 4 bobinas (ahora con un desplazamiento en Z)
# Usaremos 'd' como el desplazamiento en Z para las bobinas
z_coil_offset = d 

# Bobina 1: Eje X positivo, Z positivo
X_coil1, Y_coil1, Z_coil1 = d, 0, z_coil_offset
# Bobina 2: Eje X negativo, Z positivo
X_coil2, Y_coil2, Z_coil2 = -d, 0, z_coil_offset
# Bobina 3: Eje Y positivo, Z negativo
X_coil3, Y_coil3, Z_coil3 = 0, d, -z_coil_offset
# Bobina 4: Eje Y negativo, Z negativo
X_coil4, Y_coil4, Z_coil4 = 0, -d, -z_coil_offset

# Sumar las contribuciones de cada bobina
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_obs, y_obs, z_obs = X[i,j], Y[i,j], Z_obs_plane[i,j] # z_obs es 0 para el plano de observación

        # Campo de la bobina 1
        B1_z = B_z_corrected(x_obs, y_obs, z_obs, X_coil1, Y_coil1, Z_coil1, R, N, I, mu_0)
        # Campo de la bobina 2
        B2_z = B_z_corrected(x_obs, y_obs, z_obs, X_coil2, Y_coil2, Z_coil2, R, N, I, mu_0)
        # Campo de la bobina 3
        B3_z = B_z_corrected(x_obs, y_obs, z_obs, X_coil3, Y_coil3, Z_coil3, R, N, I, mu_0)
        # Campo de la bobina 4
        B4_z = B_z_corrected(x_obs, y_obs, z_obs, X_coil4, Y_coil4, Z_coil4, R, N, I, mu_0)

        B[i,j] = B1_z + B2_z + B3_z + B4_z

# Visualización del campo magnético 3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Ajustes de la superficie
# Filtrar NaNs si los hay (de puntos de singularidad)
# Reemplaza NaNs con la media de los puntos válidos para graficar
B_filtered = np.nan_to_num(B, nan=np.mean(B[~np.isnan(B)])) 

surf = ax.plot_surface(X, Y, B_filtered, cmap='magma', linewidth=0, antialiased=True)

# Añadir barra de color con escala y unidades
# Usa el array B original para el cálculo de la norma si los NaNs fueron filtrados
norm = plt.Normalize(np.nanmin(B), np.nanmax(B)) 
cbar = fig.colorbar(surf, shrink=0.5, aspect=5, norm=norm, label='Magnitud del Campo (T)')
cbar.ax.tick_params(labelsize=10)

# Etiquetas, título y leyenda
ax.set_xlabel('Posición X (m)', fontsize=12)
ax.set_ylabel('Posición Y (m)', fontsize=12)
ax.set_zlabel('Magnitud del Campo (T)', fontsize=12)
plt.title('Campo Magnético Generado por 4 Bobinas Circulares en Cruz (Frente a Frente)', fontsize=14)

# Habilitar la interacción 3D y ajustar vista
ax.set_zlim(np.nanmin(B), np.nanmax(B)) # Usa el array B original para los límites
ax.view_init(30, 45) # Ángulo de elevación y acimut para la vista inicial
plt.show()
