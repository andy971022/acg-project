import math
import time

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)
res = 1280, 720
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
max_ray_depth = 15
eps = 1e-4
inf = 8.2e0 # Scatter Radius

fov = 0.7  # field of view
dist_limit = 100

camera_pos = ti.Vector([0.00, 0.00, 5.0])  # [x, y, zoom]
light_pos = [0.0, 0.0, 0.0]  # [x, y, zoom]
light_normal = [0.00, 0.00, -0.3]
light_radius = 0.4


@ti.func
def intersect_light(pos, d):
    light_loc = ti.Vector(light_pos)
    dot = -d.dot(ti.Vector(light_normal))
    dist = d.dot(light_loc - pos)
    dist_to_light = inf
    if dot > 0 and dist > 0:
        D = dist / dot
        dist_to_center = (light_loc - (pos + D * d)).norm_sqr()
        if dist_to_center < light_radius ** 2:
            dist_to_light = D ** 2
    return dist_to_light


@ti.func
def out_dir(n):
    u = ti.Vector([0.5, 1.0, 0.5])
    if abs(n[1]) < 1 - eps:
        u = n.cross(ti.Vector([0.5, 1.0, 0.5])).normalized()
    v = n.cross(u)
    phi = 2 * math.pi * ti.random()
    ay = ti.sqrt(ti.random())
    ax = ti.sqrt(1 - ay ** 2)  # distribution
    return (
        ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n
    )  # Electron Cloud Orientation and Positioning


# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
@ti.func
def sdf(o):
    wall = min(o[1] + 10 * fov, o[2] + 3 * fov)  ## Cloud

    # Let's draw a Boron
    # Neutrons and protons:

    # Protons
    proton = (o - ti.Vector([0.09, 0.05, 0.03])).norm() - 0.05
    proton_2 = (o - ti.Vector([0.01, 0.02, 0.05])).norm() - 0.05
    proton_3 = (o - ti.Vector([0.05, -0.07, 0.02])).norm() - 0.05

    # Neutrons
    neutron = (o - ti.Vector([0.15, 0.05, 0.03])).norm() - 0.065
    neutron_2 = (o - ti.Vector([0.01, 0.09, 0.05])).norm() - 0.065
    neutron_3 = (o - ti.Vector([0.09, -0.04, 0.09])).norm() - 0.065

    # || x - bhat || - c :
    # c changes the size of the sphere
    # bhat is the center

    proton_geometry = max(proton, -(0.52 - (o[1] * 0.6 + o[2] * 0.8)))
    proton_2_geometry = max(proton_2, -(0.52 - (o[1] * 0.6 + o[2] * 0.8)))
    proton_3_geometry = max(proton_3, -(0.52 - (o[1] * 0.6 + o[2] * 0.8)))

    neutron_geometry = max(neutron, -(0.52 - (o[1] * 0.6 + o[2] * 0.8)))
    neutron_2_geometry = max(neutron_2, -(0.52 - (o[1] * 0.2 + o[2] * 0.8)))
    neutron_3_geometry = max(neutron_3, -(0.52 - (o[1] * 0.6 + o[2] * 0.8)))

    return min(
        wall,
        proton_geometry,
        proton_2_geometry,
        proton_3_geometry,
        neutron_geometry,
        neutron_2_geometry,
        neutron_3_geometry,
    )


@ti.func
def ray_march(p, d):
    j = 0  # steps, should march 20 times or more
    dist = 0.0
    while j < 25 and sdf(p + dist * d) > 1e-6 and dist < inf:
        dist += sdf(p + dist * d)
        j += 1
    return min(inf, dist)


@ti.func
def sdf_normal(p):
    d = 1e-4
    n = ti.Vector([0.0, 1.0, 0.0])
    sdf_center = sdf(p)
    for i in ti.static(range(3)):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(inc))
    return n.normalized()


@ti.func
def next_hit(pos, d):
    closest, normal, c = inf, ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest)
        hit_pos = pos + d * closest
        t = hit_pos.norm()
        c = ti.Vector([0.2 * t, 0.2 * t, 0.2 * t])  # color
    return closest, normal, c


@ti.kernel
def render(time: float, total: int):
    for u, v in color_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        d = ti.Vector(
            [
                (2 * fov * (u + ti.random()) / res[1] - fov * aspect_ratio - 1e-5),
                2 * fov * (v + ti.random()) / res[1] - fov - 1e-5,
                -1.0,
            ]
        )
        d = d.normalized()

        throughput = ti.Vector([1.0, 1.0, 1.0])  # color tone changes

        depth = 0
        hit_light = 0.0

        while depth < max_ray_depth:
            closest, normal, c = next_hit(pos, d)
            depth += 1
            dist_to_light = intersect_light(pos, d)
            if dist_to_light < closest:
                hit_light = 1
                depth = max_ray_depth
            else:
                hit_pos = pos + closest * d
                if normal.norm_sqr() != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c
                else:
                    depth = max_ray_depth
        color_buffer[u, v] += throughput * hit_light


gui = ti.GUI("A Tiny World: Atom", res)
video_manager = ti.VideoManager(output_dir="./", framerate=24, automatic_build=False)
last_t = 0
for i in range(2000):
    render(float(i), 2000)
    interval = 10
    if i % interval == 0 and i > 0:
        print("{:.2f} samples/s".format(interval / (time.time() - last_t)))
        last_t = time.time()
        img = color_buffer.to_numpy() * (1 / (i + 1))
        img = img / img.mean() * 0.24  # Normalize
        gui.set_image(np.sqrt(img))  # color smoothing
        video_manager.write_frame(np.sqrt(img))
        gui.show()

video_manager.make_video(gif=True, mp4=True)
