import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

# =============================================================================
# 1. Configuración Inicial y Parámetros
# =============================================================================
plt.close('all')
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Campo magnético 3D
ax2 = fig.add_subplot(gs[0, 1])                   # Temperatura vs tiempo
ax3 = fig.add_subplot(gs[1, 0], projection='3d')  # Campo + Calor 3D
ax4 = fig.add_subplot(gs[1, 1])                   # Mapa de calor 2D

# Parámetros comunes
N = 110
R = 0.0565  # Radio de las bobinas
I = 2
d = 0.15  # Separación aumentada para mejor visualización
d_e=0.1
mu_0 = 4 * np.pi * 1e-7

# =============================================================================
# 2. Cálculo del Campo Magnético
# =============================================================================
def B_z(x, y, z, a, b):
    k = (mu_0 * N * I)/(4 * np.pi)
    def integrand(phi):
        xc, yc = a + R*np.cos(phi), b + R*np.sin(phi)
        dlx, dly = -R*np.sin(phi), R*np.cos(phi)
        rx, ry, rz = x - xc, y - yc, z
        r_mag = (rx**2 + ry**2 + rz**2)**1.5
        return (dlx*ry - dly*rx)/r_mag
    integral, _ = quad(integrand, 0, 2*np.pi)
    return abs(k*integral)

# Malla y cálculo
x = y = np.linspace(-d, d, 10)
X, Y = np.meshgrid(x, y)
B = np.array([[B_z(x,y,0,d+d_e,0)+B_z(x,y,0,-d-d_e,0)+
              B_z(x,y,0,0,d+d_e)+B_z(x,y,0,0,-d-d_e) 
              for y in Y.ravel()] for x in X.ravel()]).reshape(X.shape)

# Interpolación
X_fine, Y_fine = np.meshgrid(np.linspace(-d, d, 100), np.linspace(-d, d, 100))
B_fine = griddata((X.ravel(), Y.ravel()), B.ravel(), (X_fine, Y_fine), method='cubic')

# =============================================================================
# 3. Gráfica 3D del Campo Magnético (Subplot 1)
# =============================================================================
surf = ax1.plot_surface(X_fine, Y_fine, B_fine, cmap='plasma', rstride=2, cstride=2, alpha=0.9)
ax1.set(title='Campo Magnético 3D', xlabel='X (m)', ylabel='Y (m)', zlabel='|B| (T)')
fig.colorbar(surf, ax=ax1, label='Densidad de Flujo (T)', shrink=0.6)

# Bobinas 3D
theta = np.linspace(0, 2*np.pi, 100)
for pos in [(d,0), (-d,0), (0,d), (0,-d)]:
    if pos[0] !=0:
        coil = [pos[0]+np.zeros_like(theta), R*np.cos(theta), R*np.sin(theta)]
    else:
        coil = [R*np.cos(theta), pos[1]+np.zeros_like(theta), R*np.sin(theta)]
    ax1.plot(*coil, 'r-', lw=1.5, alpha=0.7)

# =============================================================================
# 4. Modelado Térmico (Subplot 2)
# =============================================================================
# Parámetros térmicos
rho_cobre = 1.68e-8 # Resistividad (Ω·m)
diam_alambre = 1e-3 # Diámetro del alambre (1 mm)
h_conv = 25 # Coeficiente convectivo (W/m²K)
T_amb = 25 # Temperatura ambiente (°C)
C_cobre = 385 # Capacidad calorífica (J/kg·K)
dens_cobre = 8960 # Densidad (kg/m³)

# Cálculos térmicos
long_bobina = N*2*np.pi*R
A_seccion = np.pi*(diam_alambre/2)**2
R_elect = rho_cobre*long_bobina/A_seccion
P_term = I**2 * R_elect

# Modelo diferencial
def modelo(t, T): return (P_term - h_conv*2*np.pi*R*diam_alambre*N*(T-T_amb))/(dens_cobre*long_bobina*A_seccion*C_cobre)
t_eval = np.linspace(0, 1800, 100)
sol = solve_ivp(modelo, [0,1800], [T_amb], t_eval=t_eval)

# Gráfica térmica
ax2.plot(t_eval/60, sol.y[0], 'r-', lw=2)
ax2.set(title='Evolución Térmica', xlabel='Tiempo (min)', ylabel='Temperatura (°C)',
        grid=True, ylim=(T_amb, None))
ax2.annotate(f'T° máxima: {sol.y[0,-1]:.1f}°C', xy=(30, sol.y[0,-1]), 
            xytext=(10, sol.y[0,-1]+5), arrowprops=dict(arrowstyle='->'))

# =============================================================================
# 5. Gráficas Combinadas (Subplots 3 y 4)
# =============================================================================
# Subplot 3: Campo 3D con superposición térmica
surf2 = ax3.plot_surface(X_fine, Y_fine, B_fine, facecolors=cm.hot(B_fine/np.max(B_fine)), 
                        rstride=2, cstride=2, alpha=0.8)
ax3.set(title='Campo Magnético con Distribución Térmica', xlabel='X (m)', ylabel='Y (m)', zlabel='|B| (T)')

# Subplot 4: Mapa de calor 2D
heatmap = ax4.imshow(B_fine, extent=[-d, d, -d, d], cmap='hot', origin='lower')
ax4.set(title='Mapa de Calor 2D', xlabel='X (m)', ylabel='Y (m)')
fig.colorbar(heatmap, ax=ax4, label='Densidad de Potencia (W/m²)')

# =============================================================================
# 6. Ajustes Finales
# =============================================================================
for ax in [ax1, ax3]:
    ax.view_init(30, 45)
    ax.set_zlim(0, B_fine.max()*1.2)
    ax.dist = 10

plt.tight_layout()
plt.show()