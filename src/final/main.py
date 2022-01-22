import taichi as ti
import random
from cloud import Neutron, Proton, Electron, ElectronCloud

ti.init(arch=ti.gpu)
pi = 3.141592653
n = 1000  # The num of the electron
pos = ti.Vector.field(3, ti.f32, shape=n)  # electron cloud layer 1
pos_2 = ti.Vector.field(3, ti.f32, shape=n)  # electron cloud layer 2
color = ti.Vector.field(3, ti.f32, shape=n)
color_2 = ti.Vector.field(3, ti.f32, shape=n)


def init():

    ## We are drawing a Carbon atom here: Six protons and Six neutrons
    ## Proton is Green
    ## Neutron is Red

    electronCloud = ElectronCloud()

    proton_1 = Proton(0.05)
    proton_1.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.09, 0.05, 0.03]))
    electronCloud.add_proton(proton_1)
    proton_2 = Proton(0.05)
    proton_2.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.01, 0.02, 0.05]))
    electronCloud.add_proton(proton_2)
    proton_3 = Proton(0.05)
    proton_3.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.05, -0.07, 0.02]))
    electronCloud.add_proton(proton_3)
    proton_4 = Proton(0.05)
    proton_4.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.03, -0.09, 0.05]))
    electronCloud.add_proton(proton_4)
    proton_5 = Proton(0.05)
    proton_5.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.13, 0.09, -0.05]))
    electronCloud.add_proton(proton_5)
    proton_6 = Proton(0.05)
    proton_6.initialize(ti.Vector([0, 1, 0]), ti.Vector([-0.13, 0.02, 0.05]))
    electronCloud.add_proton(proton_6)

    neutron_1 = Neutron(0.065)
    neutron_1.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.15, 0.05, 0.03]))
    electronCloud.add_neutron(neutron_1)
    neutron_2 = Neutron(0.065)
    neutron_2.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.01, 0.09, 0.05]))
    electronCloud.add_neutron(neutron_2)
    neutron_3 = Neutron(0.065)
    neutron_3.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.09, -0.04, 0.09]))
    electronCloud.add_neutron(neutron_3)
    neutron_4 = Neutron(0.065)
    neutron_4.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.02, -0.05, 0.07]))
    electronCloud.add_neutron(neutron_4)
    neutron_5 = Neutron(0.065)
    neutron_5.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.10, 0.07, -0.02]))
    electronCloud.add_neutron(neutron_5)
    neutron_6 = Neutron(0.065)
    neutron_6.initialize(ti.Vector([1, 0, 0]), ti.Vector([-0.09, 0.04, 0.02]))
    electronCloud.add_neutron(neutron_6)

    return electronCloud


def rotate(angle, speed):
    angle += speed
    return angle


def electron(pos, color, n, ratio, color_vec):
    for i in range(n):
        u1 = ti.acos(2 * random.random() - 1) - pi / 2
        u2 = 2 * pi * random.random()
        x = ti.cos(u1) * ti.cos(u2) + (random.random() - random.random()) / 5.0
        y = ti.cos(u1) * ti.sin(u2) + (random.random() - random.random()) / 5.0
        z = ti.sin(u1) + (random.random() - random.random()) / 5.0
        p = ratio * ti.Vector([x, y, z])
        c = ti.Vector(color_vec)
        pos[i] = p
        color[i] = c


def main():
    ec = init()
    electron(pos, color, n, 0.5, [0.8, 1, 1])
    electron(pos_2, color_2, 10 * n, 1.0, [1, 1, 1])

    window = ti.ui.Window("A Tiny World : Atom", (800, 800))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()

    camera.position(1.0, -10.0, 5.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(15)

    angle = 0
    speed = 8e-2

    count = 0
    while window.running:
        if count >= 500:
            window.running = False
        angle = rotate(angle, speed)
        x = ti.sin(angle) * 10.0
        y = ti.cos(angle) * 10.0
        camera.position(x, y, 5.0)
        scene.set_camera(camera)
        ec.display(scene)
        scene.particles(pos, radius=0.005, per_vertex_color=color)
        scene.particles(pos_2, radius=0.003, per_vertex_color=color_2)
        scene.point_light(pos=(x, y, 5.0), color=(0.5, 0.5, 0.5))
        canvas.scene(scene)
        window.write_image("./frames/frame{:03d}.png".format(count))
        count += 1
        window.show()


if __name__ == "__main__":
    main()
