import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata 

# Parámetros ajustados para visualización
N = 110
R = 0.0565  # Radio de las bobinas
I = 2
d = 0.23/2   # Separación aumentada para mejor visualización
d_e=0.08
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

# Parámetros térmicos (valores típicos para cobre)
rho_cobre = 1.68e-8  # Resistividad (Ω·m)
diam_alambre = 1e-3  # Diámetro del alambre (1 mm)
h_conveccion = 25  # Coeficiente convectivo (W/m²K)
T_ambiente = 22.6  # Temperatura ambiente (°C)
C_cobre = 385  # Capacidad calorífica (J/kg·K)
densidad_cobre = 8960  # Densidad (kg/m³)

# 1. Cálculo de resistencia y potencia térmica
longitud_bobina = N * 2 * np.pi * R  # Longitud total del alambre
area_seccion = np.pi * (diam_alambre/2)**2  # Área transversal del alambre
resistencia_bobina = rho_cobre * longitud_bobina / area_seccion
potencia_termica = I**2 * resistencia_bobina  # Potencia por bobina (W)
tiempo_calentamiento = 10 * 60  # 30 minutos (en segundos)

# 2. Modelo térmico dinámico
def modelo_termico(t, T, h, A_superficie, masa, P):
    dTdt = (P - h * A_superficie * (T - T_ambiente)) / (masa * C_cobre)
    return dTdt

# Parámetros geométricos para transferencia de calor
area_superficial = 2 * np.pi * R * diam_alambre * N  # Área superficial de la bobina
masa_bobina = densidad_cobre * longitud_bobina * area_seccion  # Masa total

# Solución de la ecuación diferencial
t_simulacion = np.linspace(0, tiempo_calentamiento, 100)  # 30 minutos (en segundos)
solucion = solve_ivp(
    fun=modelo_termico,
    t_span=(0, tiempo_calentamiento),
    y0=[T_ambiente],
    args=(h_conveccion, area_superficial, masa_bobina, potencia_termica),
    t_eval=t_simulacion
)

# 3. Gráfico de temperatura vs tiempo
plt.figure(figsize=(10, 6))
plt.plot(solucion.t/60, solucion.y[0], 'b-', linewidth=2)
plt.title('Evolución de la Temperatura de las Bobinas', fontsize=14)
plt.xlabel('Tiempo (minutos)', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate(f'Temperatura final: {solucion.y[0,-1]:.1f}°C',
             xy=(solucion.t[-1]/60, solucion.y[0,-1]), 
             xytext=(solucion.t[-1]/60 - 5, solucion.y[0,-1] + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()

# Generar malla de temperatura-MAPA DE CALOR 2D
T_final = solucion.y[0, -1]
coil_positions = [(d + d_e, 0), (-d - d_e, 0), (0, d + d_e), (0, -d - d_e)]
Temp = np.zeros_like(X_fine)
for i in range(X_fine.shape[0]):
    for j in range(X_fine.shape[1]):
        dist_min = min(np.sqrt((X_fine[i,j] - cx)**2 + (Y_fine[i,j] - cy)**2) 
                     for (cx, cy) in coil_positions)
        Temp[i,j] = T_final if dist_min <= R else T_ambiente

plt.figure(figsize=(10, 6))
heatmap = plt.imshow(
    Temp,
    extent=[-d, d, -d, d],
    cmap='hot',
    origin='lower',
    aspect='auto'
)
plt.colorbar(heatmap, label='Temperatura (°C)')
plt.scatter([d+d_e, -d-d_e, 0, 0], [0, 0, d+d_e, -d-d_e], c='red', s=50, label='Bobinas')
plt.title('Mapa de Calor 2D - Temperatura Final (30 minutos)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.show()

#Mapa 3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    X_fine, Y_fine, Temp,
    cmap='hot',
    rstride=2,
    cstride=2,
    alpha=0.8,
    linewidth=0.2
)
ax.set_zlabel('Temperatura (°C)')
ax.set_title('Distribución de Temperatura 3D')
cbar = fig.colorbar(surf, shrink=0.7, label='Temperatura (°C)')
plt.show()