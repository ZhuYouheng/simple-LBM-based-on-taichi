import taichi as ti
import numpy as np
import pandas as pd
import taichi.math as tm
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
Ny = 140
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
Re = 340
U = 0.1  # 0.223
#niu = U * Lx / Re  # 运动粘滞系数
niu = 0.026
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

Gamma = ti.Vector([0, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/12, 1.0/12, 1.0/12, 1.0/12])
Force_x = 1e-4
#zoom image
zoom = 4
#unit: pixel
@ti.kernel
def draw_circle(center_x: int, center_y: int, radius: ti.f64, canvas: ti.template()):
    (x_max, y_max) = (Nx, Ny)
    for i, j in canvas:
        if((center_x-i)**2+(center_y-j)**2<=radius**2):
            canvas[i, j] = 1
    for i, j in canvas:
        for dx in range(-1,2):
            for dy in range(-1,2):
                (i_, j_) = (i+dx, j+dy)
                if (i==x_max-1 or i==0 or j==y_max-1 or j==0) and canvas[i, j] == 1:#margins of the canvas
                    canvas[i, j] = 0
                    break
                if i_<x_max and i_>=0 and j_<y_max and j_>=0 and canvas[i, j] == 1 and canvas[i_, j_] == -1:#margins of the circle
                    canvas[i, j] = 0
                    break
    


@ti.func
def is_point_within_a_rect(p1: ti.types.vector(2, ti.f64), p2: ti.types.vector(2, ti.f64), p0: ti.types.vector(2, ti.f64)) -> bool:
    return tm.dot((p1-p0),(p2-p0))<=0


@ti.func
def is_point_below_a_line(k_v: ti.types.vector(2, ti.f64), p1: ti.types.vector(2, ti.f64), p2: ti.types.vector(2, ti.f64), p0: ti.types.vector(2, ti.f64)) -> bool:
    k_x = k_v.dot(tm.normalize(p2-p1))
    k_y = k_v.dot(tm.rotation2d(np.pi/2)@(tm.normalize(p2-p1)))
    k = k_y/k_x
    a = tm.length(p1-p2)/2
    x = tm.dot((p0-(p1+p2)/2), tm.normalize(p2-p1))
    y = 0.5*k*a - 0.5*k*x**2/a
    return y > tm.dot((p0-(p1+p2)/2), tm.rotation2d(np.pi/2)@(tm.normalize(p2-p1)))
    
@ti.func
def generate_next_k(r_k_ori: ti.types.vector(2, ti.f64), p1: ti.types.vector(2, ti.f64), p2: ti.types.vector(2, ti.f64)) -> ti.types.vector(2, ti.f64):
    r_k_new =  tm.normalize(-tm.reflect(r_k_ori,tm.normalize(p1-p2)))
    return r_k_new


@ti.kernel
def draw_model(canvas: ti.template(), r_k_ori: ti.types.vector(2, ti.f64), p_list: ti.template()):
    pass
    
    for i, j in canvas:
        p0 = ti.Vector([i,j], dt=ti.f64)
        is_within = True
        cur_k_v = r_k_ori
        count_how_many_times_not_within_a_rect = 0
        for k in range(6):
            p_this = p_list[k]
            p_next = p_list[k+1]
            if(not is_point_within_a_rect(p_this, p_next, p0)):
                cur_k_v = generate_next_k(cur_k_v, p_this, p_next)
                count_how_many_times_not_within_a_rect+=1
                continue
            if(not is_point_below_a_line(cur_k_v, p_this, p_next, p0)):
                is_within = False
                break
            cur_k_v = generate_next_k(cur_k_v, p_this, p_next)
        if p0.y < p_list[0].y:
            is_within = False
        
        elif count_how_many_times_not_within_a_rect == 6:#射线法判断是否在多边形内
            count_intersection = 0
            if p0.x>p_list[0].x and p0.x<=p_list[6].x:
                count_intersection+=1
            for k in range(0,6):
                p_this = p_list[k]
                p_next = p_list[k+1]
                if((p0.x<=p_this.x and p0.x>p_next.x) or (p0.x>p_this.x and p0.x<=p_next.x)) and p0.y > max(p_this.y, p_next.y):
                    count_intersection += 1
            if count_intersection%2 == 0:
                is_within = False
        if is_within:
            canvas[i, j] = 1
    
    for i, j in canvas:
        for dx in range(-1,2):
            for dy in range(-1,2):
                (i_, j_) = (i+dx, j+dy)
                if i_<Nx and i_>=0 and j_<Ny and j_>=0 and canvas[i, j] == 1 and canvas[i_, j_] == -1:#margins of the model
                    canvas[i, j] = 0
                    break
         


                    

#unit: pixel
@ti.kernel
def draw_pipe(l_x: int, l_y: int, r_x: int, r_y: int, canvas: ti.template()):
    for i, j in canvas:
        if(i<=r_x and i>=l_x and j<=r_y and j>=l_y):
            canvas[i, j] = 1
    for i, j in canvas:
        if (i == r_x or i == l_x or j == r_y or j == l_y) and canvas[i, j] == 1:#margins of the pipe
            canvas[i, j] = 0
 
@ti.kernel
def get_margin_neighbour(canvas: ti.template(), first_free_neighbour: ti.template()):
    (x_max, y_max) = (Nx, Ny)
    for i, j in canvas:
        if canvas[i, j] == 0:
            flag_for_no_neighbours = True
            for dx in range(-1,2):
                for dy in range(-1,2):
                    (i_, j_) = (i+dx, j+dy)
                    if i_<x_max and i_>=0 and j_<y_max and j_>=0 and canvas[i_, j_] == 1:#margins of the models
                        flag_for_no_neighbours = False
                        first_free_neighbour[i, j] = (dx, dy)
                        break
            if flag_for_no_neighbours: #delete if from the marginals
                canvas[i, j] = -1



# 初始化
@ti.kernel
def init():
    u.fill((0, 0))
    rho.fill(rho0)
    for i, j in is_free:
        if j < 60 and j > 20:
            u[i, j] = (0.1, 0)


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
                '''
                #体力驱动
                if j<=60:
                    F[i, j][k] += Gamma[k]*Force_x*e[k][0]
                '''
        ''' 这里是用周期性边界
        elif is_free[i, j] == 0 and ((i==0)):
            for k in range(Q):
                ip = i - e[k][0] if i - e[k][0]>=0 else Nx-1
                jp = j - e[k][1]
                F[i, j][k] = f[ip, jp][k] + (feq(k, rho[ip, jp], u[ip, jp]) - f[ip, jp][k]) / tau_f
        elif is_free[i, j] == 0 and ((i==Nx-1)):
            for k in range(Q):
                ip = i - e[k][0] if i - e[k][0]<=Nx-1 else 0
                jp = j - e[k][1]
                F[i, j][k] = f[ip, jp][k] + (feq(k, rho[ip, jp], u[ip, jp]) - f[ip, jp][k]) / tau_f
        '''
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
        if is_free[i, j] == 0 and i!=0 and i!=Nx-1:
            # 此时进行边界处理
            (im, jm) = (i, j) + first_free_neighbour[i, j]
            rho[i, j] = rho[im, jm]
            u[i, j] = (0, 0)
            for k in range(Q):
                f[i, j][k] = feq(k, rho[i, j], u[i, j]) + (f[im, jm][k] - feq(k, rho[im, jm], u[im, jm]))*(1-1/tau_f)
        elif is_free[i, j] == 0 and i==0: #左端使用压强边界，非平衡外推。近似使用p=c_s^2\rho=\frac{c^2}{3}\rho，即压强实际上与密度一一对应
            (im, jm) = (i, j) + first_free_neighbour[i, j]
            rho[i, j] = 1.01
            r = 1 - (j - 40)**2/400
            u[i, j] = (0.1*r, 0)
            for k in range(Q):
                f[i, j][k] = feq(k, rho[i, j], u[i, j]) + (f[im, jm][k] - feq(k, rho[im, jm], u[im, jm]))*(1-1/tau_f)
        elif is_free[i, j] == 0 and i==Nx-1: #右端使用压强边界，非平衡外推。近似使用p=c_s^2\rho=\frac{c^2}{3}\rho，即压强实际上与密度一一对应
            (im, jm) = (i, j) + first_free_neighbour[i, j]
            rho[i, j] = rho[im, jm]
            r = 1 - (j - 40)**2/400
            u[i, j] = (0.1*1.01/rho[im, jm]*r, 0)
            #u[i, j] = u[im, jm]
            for k in range(Q):
                f[i, j][k] = feq(k, rho[i, j], u[i, j]) + (f[im, jm][k] - feq(k, rho[im, jm], u[im, jm]))*(1-1/tau_f)
            
        '''
        elif (j==40 or j==39): #the center of the pipe
            (im, jm) = (i, j) + first_free_neighbour[i, j]
            rho[i, j] = rho[im, jm]
            u[i, j] = (U, 0)
            for k in range(Q):
                f[i, j][k] = feq(k, rho[i, j], u[i, j]) + f[im, jm][k] - feq(k, rho[im, jm], u[im, jm])
'''
#get the value of velocity from the fluid field
@ti.kernel
def get_velocity(v: ti.template(), u: ti.template(), scale: ti.f64):
    for i in ti.grouped(u):
        v[i] = u[i].norm() * scale

#get the value of velocity from the fluid field
@ti.kernel
def get_velocity_RGB(v: ti.template(), u: ti.template(), scale: ti.f64):
    for i, j in (u):
        intensity = u[i, j].norm() * scale
        for x in range(zoom):
            for y in range(zoom):
                v[i*zoom+x, j*zoom+y] = (intensity**2.55, 0, intensity**0.6)

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

@ti.kernel
def get_rho():
    print(rho[126, 40])
    
@ti.kernel
def get_speed():
    print(u[1, 40])

gui = ti.GUI('Fluid Field!', (Nx*zoom, Ny*zoom))
draw_pipe(0, 20, Nx-1, 60, is_free)                   
#draw_circle(128, 80, 40, is_free)
p_field = ti.Vector.field(2,ti.f64,7)
p_field[0] = ti.Vector([80,60],ti.f64)
p_field[1] = ti.Vector([80,80],ti.f64)
p_field[2] = ti.Vector([100,110],ti.f64)
p_field[3] = ti.Vector([120,130],ti.f64)
p_field[4] = ti.Vector([140,110],ti.f64)
p_field[5] = ti.Vector([160,80],ti.f64)
p_field[6] = ti.Vector([160,60],ti.f64)
draw_model(is_free, ti.Vector([-1,1],ti.f64)/np.sqrt(2),p_field)
get_margin_neighbour(is_free, first_free_neighbour)
init()
print('tau_f={0}'.format(tau_f))

j=0
while gui.running:
    for i in range(300):
        evolution()
    j+=300
    image = ti.Vector.field(3, dtype=ti.f64, shape=(Nx*zoom,Ny*zoom))
    get_velocity_RGB(image, u, 10)
    gui.set_image(image)
    gui.show()
    #get_rho()
    #get_speed()
    print(j)
'''    


# image = ti.Vector.field(2,dtype=ti.f64, shape=(Nx,Ny))
n = 0
while n < 20000:
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