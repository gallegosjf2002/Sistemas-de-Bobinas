import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Parámetros ajustados para visualización
N = 110
R = 0.0565  # Radio de las bobinas
I = 2
d = 0.15/2  # Separación aumentada para mejor visualización
d_e=0.09
mu_0 = 4 * np.pi * 1e-7

# Función corregida (magnitud absoluta)
def B_z(x, y, z, a, b):
    k = (mu_0 * N * I) / (4 * np.pi)
    
    def integrand(phi):
        x_coil = a + R * np.cos(phi)
        y_coil = b + R * np.sin(phi)
        z_coil = 0
        
        dl_x = -R * np.sin(phi)
        dl_y = R * np.cos(phi)
        
        r_x = x - x_coil
        r_y = y - y_coil
        r_z = z - z_coil
        r_mag = (r_x**2 + r_y**2 + r_z**2)**1.5
        
        cross_z = dl_x * r_y - dl_y * r_x
        
        return cross_z / r_mag
    
    integral, _ = quad(integrand, 0, 2*np.pi)
    return abs(k * integral)  # Magnitud absoluta para evitar valores negativos

# Malla 3D optimizada (incluyendo eje Z)
x = np.linspace(-d, d, 100)
y = np.linspace(-d, d, 100)
z_val = 0  # Focalizar en plano Z=0 para la curva principal
X, Y = np.meshgrid(x, y)

# Cálculo del campo corregido
B = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B[i,j] = (
            B_z(X[i,j], Y[i,j], z_val, d+d_e, 0) +
            B_z(X[i,j], Y[i,j], z_val, -d-d_e, 0) +
            B_z(X[i,j], Y[i,j], z_val, 0, d+d_e) +
            B_z(X[i,j], Y[i,j], z_val, 0, -d-d_e)
        )

# Interpolación suave
x_fine = np.linspace(-d, d, 100)
y_fine = np.linspace(-d, d, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
B_fine = griddata((X.flatten(), Y.flatten()), B.flatten(), (X_fine, Y_fine), method='cubic')

# Visualización 3D profesional
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superficie con curva en Z
surf = ax.plot_surface(
    X_fine, Y_fine, B_fine,
    cmap='plasma',
    rstride=2,
    cstride=2,
    alpha=0.9,
    linewidth=0.2,
    antialiased=True
)

# Bobinas verticales autoajustadas
theta = np.linspace(0, 2*np.pi, 100)
for pos in [(d,0), (-d,0), (0,d), (0,-d)]:
    if pos[0] != 0:  # Bobinas en X
        x_coil = pos[0] * np.ones_like(theta)
        y_coil = R * np.cos(theta)
        z_coil = R * np.sin(theta)
    else:  # Bobinas en Y
        x_coil = R * np.cos(theta)
        y_coil = pos[1] * np.ones_like(theta)
        z_coil = R * np.sin(theta)
    
    # Ajuste Z para visualización 3D
    ax.plot(
        x_coil, 
        y_coil, 
        z_coil * 0.8 * B_fine.max()/R,  # Escalado a la magnitud del campo
        'r-',
        lw=2,
        alpha=0.7
    )

# Ajustes de ejes y perspectiva
ax.set_zlim(0, 1.2*B_fine.max())
ax.set_xlim(-d, d)
ax.set_ylim(-d, d)
ax.view_init(30, 45)
ax.dist = 10

# Etiquetas y estilo profesional
ax.set_xlabel('X (m)', fontsize=12, labelpad=12)
ax.set_ylabel('Y (m)', fontsize=12, labelpad=12)
ax.set_zlabel('|B| (T)', fontsize=12, labelpad=12)
plt.title('Configuración de 4 Bobinas en Cruz con Campo Magnético Optimizado', pad=20)

# Barra de color mejorada
cbar = fig.colorbar(surf, shrink=0.7, aspect=20, pad=0.1)
cbar.set_label('Densidad de Flujo Magnético (T)', rotation=270, labelpad=25)

plt.tight_layout()
plt.show()