import taichi as ti
import taichi.math as tm
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
Ny = 180
# determine whether is in the flowing field, 1/0/-1 for free/margins/not free
is_free = ti.field(dtype=ti.i8, shape=(Nx, Ny))
# (+-1,+-1)
first_free_neighbour = ti.Vector.field(2, dtype=ti.i8, shape=(Nx, Ny))
is_free.fill(-1)
first_free_neighbour.fill((0, 0))

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
def draw_circle(center_x: int, center_y: int, radius: ti.f64, canvas: ti.template()):
    (x_max, y_max) = (Nx, Ny)
    for i, j in canvas:
        if((center_x-i)**2+(center_y-j)**2<=radius**2):
            canvas[i, j] = 1
    for i, j in canvas:
        for dx in range(-1,2):
            for dy in range(-1,2):
                (i_, j_) = (i+dx, j+dy)
                if (i_==x_max-1 or i_==0 or j_==y_max-1 or j_==0) and canvas[i, j] == 1:#margins of the canvas
                    canvas[i, j] = 0
                    break
                if i_<x_max and i_>=0 and j_<y_max and j_>=0 and canvas[i, j] == 1 and canvas[i_, j_] == -1:#margins of the circle
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

@ti.kernel
def visualize_model(v: ti.template(), u: ti.template()):
    for i in ti.grouped(v):
        if u[i] == -1:
            v[i] = (0,0,0)
        elif u[i] == 0:
            v[i] = (1,0,0)
        elif u[i] == 1:
            v[i] = (1,1,1)
            
@ti.kernel
def visualize_model_margin_neighbour(v: ti.template(), u: ti.template()):
    for i in ti.grouped(v):
        if u[i][0] == 0 and u[i][1] == 0:
            v[i] = (1,1,1)
        else:
            v[i] = (0,0,0)
        
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
#get_margin_neighbour(is_free, first_free_neighbour)
print(is_free[0, 70])
'''
p1 = ti.Vector([0,1.0])
p2 = ti.Vector([2.0,1.0])
p3 = ti.Vector([1.0,-10])
k_v_0 = ti.Vector([1.0,1.0])/np.sqrt(2)
print(generate_next_k(k_v_0,p1,p2))
'''
gui = ti.GUI('Fluid Field!', (Nx, Ny))
while gui.running:
    image = ti.Vector.field(3,dtype = ti.f64, shape=(Nx, Ny))
    visualize_model(image, is_free)
    #visualize_model_margin_neighbour(image, first_free_neighbour)
    gui.set_image(image)
    gui.show()
