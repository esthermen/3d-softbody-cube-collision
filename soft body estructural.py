import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================================
# 3D Mass-Spring Soft Body with Rigid Plane Collision
# Explicit RK4 time integration
# ==========================================================

# -------------------------
# Physical parameters
# -------------------------
N = 5                   # number of masses per axis (NxNxN grid)
d = 3                   # spatial dimensions
m = 1.0                 # mass of each particle (kg)
k_struct = 200.0        # structural spring stiffness (N/m)
L0 = 0.5                # natural spring length (m)
g = 9.81                # gravity (m/s²)
dt = 0.01               # time step (s)
amort = 0.5             # viscous damping coefficient

# -------------------------
# State variables
# r[i,j,k] -> position
# v[i,j,k] -> velocity
# -------------------------
r = np.zeros((N, N, N, d))
v = np.zeros((N, N, N, d))

# Initial cubic configuration
spacing = L0
origin = np.array([-(N-1)/2*spacing, 5.0, -(N-1)/2*spacing])

for i in range(N):
    for j in range(N):
        for k in range(N):
            r[i,j,k] = origin + spacing * np.array([i, j, k])

# -------------------------
# Rigid collision plane
# -------------------------
dist = L0 * N
dist_obj = 3 * dist

xc_min = r[0,0,0,0] + dist/2 - dist_obj/2
xc_max = r[N-1,0,0,0] + dist/2 + dist_obj/2
z_obj  = r[N-1,N-1,N-1,2] - 2

yc_min = r[0,0,0,1] - dist_obj/2
yc_max = r[0,N-1,0,1] + dist_obj/2

e  = 0.3   # restitution coefficient
mu = 0.4   # friction coefficient

# -------------------------
# Visualization setup
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Mass-Spring Soft Body with Collision")

spring_lines = []
for i in range(N):
    for j in range(N):
        for k in range(N):
            if i < N-1:
                line, = ax.plot([], [], [], 'b-', lw=0.5)
                spring_lines.append((line, (i,j,k), (i+1,j,k)))
            if j < N-1:
                line, = ax.plot([], [], [], 'b-', lw=0.5)
                spring_lines.append((line, (i,j,k), (i,j+1,k)))
            if k < N-1:
                line, = ax.plot([], [], [], 'b-', lw=0.5)
                spring_lines.append((line, (i,j,k), (i,j,k+1)))

mass_dots = [[[ax.plot([], [], [], 'ro', markersize=2)[0]
                for k in range(N)] for j in range(N)] for i in range(N)]

# Collision plane (visual only)
xx = [xc_min, xc_max, xc_max, xc_min, xc_min]
yy = [yc_min, yc_min, yc_max, yc_max, yc_min]
zz = [z_obj] * 5
ax.plot(xx, yy, zz, 'k-', lw=2)

# ==========================================================
# Internal spring forces
# ==========================================================
def spring_force(p1, p2):
    dr = p2 - p1
    L = np.linalg.norm(dr)
    if L == 0:
        return np.zeros(3)
    return k_struct * (L - L0) * (dr / L)

def acceleration(i,j,k, r_new, v_new):
    F = np.zeros(3)

    # Structural neighbours
    for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        ni,nj,nk = i+di, j+dj, k+dk
        if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
            F += spring_force(r_new[i,j,k], r_new[ni,nj,nk])

    # Damping
    F -= amort * v_new[i,j,k]

    # Gravity
    F[2] -= m*g

    return F / m

# ==========================================================
# RK4 integrator
# ==========================================================
def deriv_global(r_flat, v_flat):
    r_new = r_flat.reshape((N,N,N,3))
    v_new = v_flat.reshape((N,N,N,3))
    ar = np.zeros_like(r_new)
    av = np.zeros_like(v_new)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                ar[i,j,k] = v_new[i,j,k]
                av[i,j,k] = acceleration(i,j,k,r_new,v_new)

    return ar.flatten(), av.flatten()

def rk4_step(r_flat,v_flat,dt):
    k1_r,k1_v = deriv_global(r_flat,v_flat)
    k2_r,k2_v = deriv_global(r_flat+0.5*dt*k1_r,v_flat+0.5*dt*k1_v)
    k3_r,k3_v = deriv_global(r_flat+0.5*dt*k2_r,v_flat+0.5*dt*k2_v)
    k4_r,k4_v = deriv_global(r_flat+dt*k3_r,v_flat+dt*k3_v)

    r_next = r_flat + dt/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v_next = v_flat + dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)

    return r_next, v_next

r_flat = r.flatten()
v_flat = v.flatten()

# ==========================================================
# Animation update
# ==========================================================
def update(frame):
    global r_flat, v_flat, r, v

    r_flat, v_flat = rk4_step(r_flat, v_flat, dt)
    r = r_flat.reshape((N,N,N,3))
    v = v_flat.reshape((N,N,N,3))

    # Collision with plane
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if r[i,j,k,2] < z_obj:
                    r[i,j,k,2] = z_obj
                    if v[i,j,k,2] < 0:
                        v[i,j,k,2] *= -e
                    v[i,j,k,0] *= (1-mu)
                    v[i,j,k,1] *= (1-mu)

    # Update springs
    for line, a, b in spring_lines:
        x0,y0,z0 = r[a]
        x1,y1,z1 = r[b]
        line.set_data([x0,x1],[y0,y1])
        line.set_3d_properties([z0,z1])

    # Update masses
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x,y,z = r[i,j,k]
                mass_dots[i][j][k].set_data([x],[y])
                mass_dots[i][j][k].set_3d_properties([z])

    return [line for line,_,_ in spring_lines]

ani = animation.FuncAnimation(fig, update, frames=600, interval=20, blit=False)
plt.show()

