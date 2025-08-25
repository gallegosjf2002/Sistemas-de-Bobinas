import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# =============================================================================
# Parámetros Físicos y Geométricos
# =============================================================================
N = 110                   # Vueltas por bobina
R = 0.055                 # Radio de las bobinas (m)
I = 2.0                   # Corriente (A)
mu_0 = 4e-7 * np.pi       # Permeabilidad magnética (T·m/A)
L = 0.04                  # Longitud axial de cada bobina (m)
d_posterior = 0.15        # Distancia entre bordes posteriores (m)
d_anterior = 0.23         # Distancia entre bordes anteriores (m)
M_segmentos = 200          # Segmentos axiales por bobina
K_segmentos = 360          # Segmentos angulares por bobina

# Posiciones de los centros de las bobinas
centros = [
    (d_posterior/2 + L/2, 0),   # Bobina derecha (eje X)
    (-(d_posterior/2 + L/2), 0), # Bobina izquierda (eje X)
    (0, d_posterior/2 + L/2),    # Bobina superior (eje Y)
    (0, -(d_posterior/2 + L/2))  # Bobina inferior (eje Y)
]

# =============================================================================
# Función de Cálculo del Campo Magnético
# =============================================================================
def B_total(x, y, z):
    B = np.zeros(3)
    z_vals = np.linspace(-L/2, L/2, M_segmentos)  # Posiciones axiales
    phi_vals = np.linspace(0, 2*np.pi, K_segmentos, endpoint=False)
    
    for centro_x, centro_y in centros:
        for z_prime in z_vals:
            for phi in phi_vals:
                # Posición del segmento de corriente
                if centro_y == 0:  # Bobina en eje X
                    xc = centro_x + z_prime
                    yc = centro_y + R * np.cos(phi)
                    zc = R * np.sin(phi)
                else:              # Bobina en eje Y
                    xc = centro_x + R * np.cos(phi)
                    yc = centro_y + z_prime
                    zc = R * np.sin(phi)
                
                # Vector dl
                dl = np.array([-R * np.sin(phi), R * np.cos(phi), 0]) * (2*np.pi/K_segmentos)
                dl *= L/M_segmentos  # Ajuste por discretización axial
                
                # Vector r
                rx = x - xc
                ry = y - yc
                rz = z - zc
                r_mag = (rx**2 + ry**2 + rz**2)**1.5
                
                # Contribución al campo
                B += (mu_0 * N * I / (4 * np.pi)) * np.cross(dl, [rx, ry, rz]) / r_mag
    return np.linalg.norm(B)

# =============================================================================
# Malla 3D desde los bordes posteriores
# =============================================================================
x_min = -d_posterior/2
x_max = d_posterior/2
y_min = -d_posterior/2
y_max = d_posterior/2

x = np.linspace(x_min, x_max, 15)
y = np.linspace(y_min, y_max, 15)
X, Y = np.meshgrid(x, y)

B = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B[i,j] = B_total(X[i,j], Y[i,j], 0)

# =============================================================================
# Visualización
# =============================================================================
fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Gráfica 3D
X_fine, Y_fine = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
B_fine = griddata((X.ravel(), Y.ravel()), B.ravel(), (X_fine, Y_fine), method='cubic')
ax1.plot_surface(X_fine, Y_fine, B_fine, cmap='plasma', alpha=0.9)
ax1.set(xlabel='X (m)', ylabel='Y (m)', zlabel='|B| (T)', title='Campo Magnético desde Bordes Posteriores')

# Mapa de calor 2D
heatmap = ax2.imshow(B_fine, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='plasma')
plt.colorbar(heatmap, ax=ax2, label='|B| (T)')
ax2.set(xlabel='X (m)', ylabel='Y (m)', title='Distribución en Plano XY')

plt.tight_layout()
plt.show()