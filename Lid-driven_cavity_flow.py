import taichi as ti
import numpy as np
import pandas as pd
# Choose any of the following backend when initializing Taichi
# - ti.cpu
# - ti.gpu
# - ti.cuda
# - ti.vulkan
# - ti.metal
# - ti.opengl
ti.init(arch=ti.gpu, default_fp=ti.f64)
Q = 9
Nx = 256
Ny = 256
# determine whether is in the flowing field, 1/0/-1 for free/margins/not free
is_free = ti.field(dtype=ti.i8, shape=(Nx, Ny))
# (+-1,+-1)
first_free_neighbour = ti.Vector.field(2, dtype=ti.i8, shape=(Nx, Ny))
is_free.fill(-1)
first_free_neighbour.fill((0, 0))

# rho, u, u0, f, F
rho = ti.field(ti.f64, shape=(Nx, Ny))
u = ti.Vector.field(2, ti.f64, shape=(Nx, Ny))
u0 = ti.Vector.field(2, ti.f64, shape=(Nx, Ny))
f = ti.Vector.field(Q, ti.f64, shape=(Nx, Ny))
F = ti.Vector.field(Q, ti.f64, shape=(Nx, Ny))

dx = 1.0
dy = 1.0
Lx = dx * Nx
Ly = dy * Ny
dt = dx
c = dx / dt  # 1.0
rho0 = 1.0  # 流场初始质量密度
Re = 1000
U = 0.1  # 0.223
niu = U * Lx / Re  # 运动粘滞系数
tau_f = 3.0 * niu + 0.5  # 无量纲松弛时间tau

# e
e = ti.Vector.field(2, ti.i32, shape=9)
e[0] = (0, 0)
e[1] = (1, 0)
e[2] = (0, 1)
e[3] = (-1, 0)
e[4] = (0, -1)
e[5] = (1, 1)
e[6] = (-1, 1)
e[7] = (-1, -1)
e[8] = (1, -1)

# w
w = ti.Vector([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
               1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])


@ti.kernel
def initialize_geometry():
    # loop over all (x,y) indices in is_free's structure;
    for i in ti.grouped(is_free):
        is_free[i] = (1)
    for i, j in is_free:
        if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
            is_free[i, j] = (0)
        if i == 0:
            first_free_neighbour[i, j][0] = 1
        elif i == Nx - 1:
            first_free_neighbour[i, j][0] = -1
        elif j == 0:
            first_free_neighbour[i, j][1] = 1
        elif j == Ny - 1:
            first_free_neighbour[i, j][1] = -1
        if i == 0 and j == 0:
            first_free_neighbour[i, j] = (1, 1)
        if i == 0 and j == Ny - 1:
            first_free_neighbour[i, j] = (1, -1)
        if i == Nx - 1 and j == Ny - 1:
            first_free_neighbour[i, j] = (-1, -1)
        if i == Nx - 1 and j == 0:
            first_free_neighbour[i, j] = (-1, 1)


# 初始化
@ti.kernel
def init():
    u.fill((0, 0))
    rho.fill(rho0)
    for i, j in is_free:
        if j == Ny - 1:
            u[i, j][0] = U
        for k in range(Q):
            f[i, j][k] = feq(k, rho[i, j], u[i, j])


# 计算平衡态分配函数
@ti.func
def feq(k: int, rho: ti.f64, u: ti.types.vector(2, ti.f64)) -> ti.f64:
    eu = ti.math.dot(e[k], u)
    uv = ti.math.dot(u, u)
    feq = w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
    return feq


@ti.kernel
def evolution():
    for i, j in is_free:
        if is_free[i, j] == 1:
            # 先进行松弛
            for k in range(Q):
                ip = i - e[k][0]
                jp = j - e[k][1]
                F[i, j][k] = f[ip, jp][k] + (feq(k, rho[ip, jp], u[ip, jp]) - f[ip, jp][k]) / tau_f
            # 接下来计算宏观量

    for i, j in is_free:
        if is_free[i, j] == 1:
            u0[i, j] = u[i, j]
            rho[i, j] = 0
            u[i, j] = (0, 0)
            for k in range(Q):
                f[i, j][k] = F[i, j][k]  # 更新分布函数
                rho[i, j] += f[i, j][k]  # 密度+=当前k方向的分布函数

                # if f[i, j][k] < 0:
                #     print('f < 0')
                u[i, j] += e[k] * f[i, j][k]  # 累加各个方向的动量，获得总动量
            #print('rho = {0}'.format(rho[i,j]))

            u[i, j] /= rho[i, j]  # 动量除以密度得到速度

            # if rho[i, j] < 0.001:
            #    print('rho too small')
    for i, j in is_free:
        if is_free[i, j] == 0:
            # 此时同步进行边界处理
            (im, jm) = (i, j) + first_free_neighbour[i, j]
            rho[i, j] = rho[im, jm]
            if j == Ny - 1:
                u[i, j] = (U, 0)
            else:
                u[i, j] = (0, 0)
            for k in range(Q):
                f[i, j][k] = feq(k, rho[i, j], u[i, j]) + f[im, jm][k] - feq(k, rho[im, jm], u[im, jm])

#get the value of velocity from the fluid field
@ti.kernel
def get_velocity(v: ti.template(), u: ti.template(), scale: ti.f64):
    for i in ti.grouped(u):
        v[i] = u[i].norm() * scale

#get the value of velocity from the fluid field
@ti.kernel
def get_velocity_RGB(v: ti.template(), u: ti.template(), scale: ti.f64):
    for i in ti.grouped(u):
        intensity = u[i].norm() * scale
        v[i] = (intensity**2.55, 0, intensity)

@ti.kernel
def get_velocity_0(v: ti.template()):
    for i in ti.grouped(u):
        v[i] = u[i].norm()


@ti.kernel
def detect_error(u: ti.template(), n: int):
    flag = 0
    for i in ti.grouped(u):
        if not (u[i][0] > -100 and u[i][1] > -100):
            flag = 1
    if flag:
        print(n)
        print(tau_f)


gui = ti.GUI('Fluid Field!', (256, 256))
initialize_geometry()
init()
print('tau_f={0}'.format(tau_f))

while gui.running:
    for i in range(100):
        evolution()
    image = ti.Vector.field(3, dtype=ti.f64, shape=(Nx,Ny))
    get_velocity_RGB(image, u, 10)
    gui.set_image(image)
    gui.show()
'''
# image = ti.Vector.field(2,dtype=ti.f64, shape=(Nx,Ny))
n = 0
while n < 100000:
    evolution()
    # detect_error(u, n)
    n += 1
print(u)

out = []
for i in range(Nx):
    for j in range(Ny):
        out.append([i,j,u[i,j][0],u[i,j][1]])
out = np.array(out)
pd.DataFrame(out).to_csv('result.csv')
'''