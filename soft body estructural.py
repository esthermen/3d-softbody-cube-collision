import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Parámetros físicos
# -------------------------
N = 5                  # número de masas por eje 
d=3                     # dimensiones
m = 1                   # masas (kg)
k = 200                 # muelles verticales (N/m)
L0 = 0.5                # longitudes naturales verticales (m) por muelle vertical i
g = 9.81
dt = 0.01
amort = 0.5             # amortiguamiento
k_struct = 200.0
k_shear  = 150.0
k_bend   = 100.0


# -------------------------
# Estados: posiciones r[i,n,(x,y,z)] y velocidades v[i,n,(vx,vy,vz)]
# -------------------------
r = np.zeros((N, N, N,d))
v = np.zeros((N,N, N, d))


# Inicializar posiciones iniciales: cubo
spacing=L0
origin = np.array([- (N-1)/2*spacing, 5.0, - (N-1)/2*spacing])
for i in range(N):
    for j in range(N):
        for k in range(N):
            r[i,j,k,:] = origin + spacing * np.array([i, j, k])
        
# -------------------------
# Objeto fijo (pared rígida)
# -------------------------
dist=L0*N
dist_obj = 3 * dist
xc_min = r[0,0,0,0] + dist/2 - dist_obj/2
xc_max = r[N-1,0,0,0]  + dist/2 + dist_obj/2
obj_xc0 = 0.5 * (xc_min + xc_max)
z_obj = r[N-1,N-1,N-1,2]-2                   # altura del objeto (z)
yc_min = r[0,0,0,1]  - dist_obj/2
yc_max = r[0,N-1,0,1]  + dist_obj/2
xc = obj_xc0
e = 0.3                        # coeficiente de restitución (normal)
mu = 0.4                       # coeficiente de fricción (Coulomb impulsivo)



# -------------------------
# Figura y artistas 3D
# -------------------------
ylim_min=r[0,0,0,1]-0.5
ylim_max=r[0,N-1,0,1]-0.5
xlim_min=r[0,0,0,0]-0.5
xlim_max=r[0,N-1,0,0]+0.5
zlim_min=r[0,0,0,2]-0.5
zlim_max=r[0,0,N-1,2]+0.5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(ylim_min, ylim_max)
ax.set_zlim(zlim_min, zlim_max)
ax.set_title("Tela 3D sobre plano rígido (Z vertical)")


# Crear contenedores
spring_lines = []
for i in range(N):
    for j in range(N):
        for k in range(N):
            if i < N - 1:
                line, = ax.plot([], [], [], 'b-', lw=0.5, alpha=0.7)
                spring_lines.append((line, (i,j,k), (i+1,j,k)))
            if j < N - 1:
                line, = ax.plot([], [], [], 'b-', lw=0.5, alpha=0.7)
                spring_lines.append((line, (i,j,k), (i,j+1,k)))
            if k < N - 1:
                line, = ax.plot([], [], [], 'b-', lw=0.5, alpha=0.7)
                spring_lines.append((line, (i,j,k), (i,j,k+1)))

mass_dots = [[[ax.plot([], [], [], 'ro', markersize=2)[0]
                for k in range(N)] for j in range(N)] for i in range(N)]


# --- Crear el plano (objeto rígido) una vez ---
xx = [xc_min, xc_max, xc_max, xc_min, xc_min]
yy = [yc_min, yc_min, yc_max, yc_max, yc_min]
zz = [z_obj] * 5

plane_line, = ax.plot(xx, yy, zz, 'k-', lw=3)




# -------------------------
# Fuerzas internas
# -------------------------
def f_vertical(i, n,k, r_col):
    Fv = np.zeros(d)
    if i != 0:
        dr = r_col[i] - r_col[i-1]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_sup = (ell - L0) * (dr/ell)
        else:
            elong_sup = np.zeros(d)
        Fv += -k_struct * elong_sup
    if i < N-1:
        dr = r_col[i+1] - r_col[i]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_inf = (ell - L0) * (dr/ell)
        else:
            elong_inf = np.zeros(d)
        Fv += +k_struct * elong_inf
    return Fv

def f_horizontal(i, n,k, r_row):
    Fh = np.zeros(d)
    if n != 0:
        dr = r_row[n] - r_row[n-1]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_sup = (ell - L0) * (dr/ell)
        else:
            elong_sup = np.zeros(d)
        Fh += -k_struct * elong_sup
    if n < N - 1:
        dr = r_row[n+1] - r_row[n]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_inf = (ell - L0) * (dr/ell)
        else:
            elong_inf = np.zeros(d)
        Fh += +k_struct * elong_inf
    return Fh

def f_prof(i, n,k, r_prof):
    Fp = np.zeros(d)
    if k != 0:
        dr = r_prof[k] - r_prof[k-1]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_sup = (ell - L0) * (dr/ell)
        else:
            elong_sup = np.zeros(d)
        Fp += -k_struct * elong_sup
    if k < N - 1:
        dr = r_prof[k+1] - r_prof[k]
        ell = np.linalg.norm(dr)
        if ell != 0:
            elong_inf = (ell - L0) * (dr/ell)
        else:
            elong_inf = np.zeros(d)
        Fp += +k_struct * elong_inf
    return Fp

def acceleration(i, n,k, r_new, v_new):
    r_col = r_new[:, n, k, :]
    r_row = r_new[i, :, k, :]
    r_prof = r_new[i, n, :, :]
    v_col = v_new[:, n, k, :]
    v_row = v_new[i, :, k, :]
    v_prof = v_new[i, n, :, :]
    Fv = f_vertical(i,n,k,r_col)
    Fh = f_horizontal(i,n,k,r_row)
    Fp=f_prof(i,n,k,r_prof)
    a = np.zeros(d)
    a = (Fv + Fh + Fp - amort*v_new[i,n,k,:]) / m
    a[2] -= g
    return a

# -------------------------
# RK4 global
# -------------------------
def deriv_global(r_flat,v_flat):
    r_new = r_flat.reshape((N,N,N,d))
    v_new = v_flat.reshape((N,N,N,d))
    ar = np.zeros_like(r)
    av = np.zeros_like(v)
    for i in range(N):
        for n in range(N):
            for k in range(N):
                ar[i,n,k,:] = v_new[i,n,k,:]
                av[i,n,k,:] = acceleration(i, n, k, r_new, v_new)
    return ar.flatten(), av.flatten()

def rk4_step(r_flat,v_flat,dt):
    k1_r,k1_v = deriv_global(r_flat,v_flat)
    k2_r,k2_v = deriv_global(r_flat+0.5*dt*k1_r,v_flat+0.5*dt*k1_v)
    k3_r,k3_v = deriv_global(r_flat+0.5*dt*k2_r,v_flat+0.5*dt*k2_v)
    k4_r,k4_v = deriv_global(r_flat+dt*k3_r,v_flat+dt*k3_v)
    r_next = r_flat + dt/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v_next = v_flat + dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)
    return r_next,v_next


# -------------------------
# Empaquetar/desempaquetar
# -------------------------
def pack_state(r,v):
    return r.flatten(), v.flatten()

def unpack_state(r_flat,v_flat):
    return r_flat.reshape((N,N,N,d)), v_flat.reshape((N,N,N,d))

r_flat, v_flat = pack_state(r,v)

# -------------------------
# Función de actualización (animación)
# -------------------------
def update(frame):
    global r_flat,v_flat,r,v,r_ant
    r_ant=np.copy(r)
    r_flat,v_flat = rk4_step(r_flat,v_flat,dt)
    r,v = unpack_state(r_flat,v_flat)

    artists = []

    # --- Colisión con plano rígido ---
    for i in range(N):
        for n in range(N):
            for k in range(N):
                # posiciones de la masa
                x_m, y_m, z_m = r[i, n,k]
                vx_m, vy_m, vz_m = v[i, n,k]

                #pared izquierda del objeto
                if (r[i,n,k,0]>=xc_min) and (r_ant[i,n,k,0]<=xc_min)and (yc_min<=y_m<=yc_max) and (r[i,n,k,2]<z_obj):
                    r[i,n,k,0]=xc_min
                    v[i,n,k,0]=-e*vx_m
                    v[i,n,k,1]*=(1-mu)
                    v[i,n,k,2]*=(1-mu)
            
                #pared derecha del objeto
                if (r[i,n,k,0]<=xc_max) and (r_ant[i,n,k,0]>=xc_max)and (yc_min<=y_m<=yc_max) and (r[i,n,k,2]<z_obj):
                    r[i,n,k,0]=xc_max
                    v[i,n,k,0]=-e*vx_m
                    v[i,n,k,1]*=(1-mu)
                    v[i,n,k,2]*=(1-mu)
            
                #techo del objeto
                if (xc_min < r[i,n,k,0] < xc_max) and (yc_min <= y_m <= yc_max) and (z_m < z_obj):
                    r[i,n,k,2] = z_obj
                    if vz_m<0:
                        v[i,n,k,2] = -e * vz_m
                    v[i,n,k,0] *= (1 - mu)
                    v[i,n,k,1] *= (1 - mu)
                
                
            



    
    # Actualizar muelles
    for line, a, b in spring_lines:
        i1,j1,k1 = a
        i2,j2,k2 = b
        x0, y0, z0 = r[i1,j1,k1]
        x1, y1, z1 = r[i2,j2,k2]
        line.set_data([x0, x1], [y0, y1])
        line.set_3d_properties([z0, z1])

    # Actualizar masas
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x, y, z = r[i,j,k]
                mass_dots[i][j][k].set_data([x], [y])
                mass_dots[i][j][k].set_3d_properties([z])


    #actualizar limites de animacion
    ax.set_xlim(np.min(r[:,:,:,0])-0.5, np.max(r[:,:,:,0])+0.5)
    ax.set_ylim(np.min(r[:,:,:,1])-0.5, np.max(r[:,:,:,1])+0.5)
    ax.set_zlim(np.min(r[:,:,:,2])-0.5, np.max(r[:,:,:,2])+0.5)


    
    artists = []
    for line, _, _ in spring_lines:
        artists.append(line)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                artists.append(mass_dots[i][j][k])
    return artists

# -------------------------
# Lanzar animación
# -------------------------
ani = animation.FuncAnimation(fig, update, frames=800, interval=20, blit=False)
plt.show()
