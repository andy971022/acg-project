import taichi as ti
import random
import math
from cloud import Neutron, Proton, Electron, ElectronCloud

ti.init(arch=ti.gpu)
pi = 3.141592653
n = 10 # The num of the electron


def init():

    electronCloud = ElectronCloud()

    proton_1 = Proton(0.05)
    proton_1.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.09, 0.05, 0.03]))
    electronCloud.add_proton(proton_1)
    proton_2 = Proton(0.05)
    proton_2.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.01, 0.02, 0.05]))
    electronCloud.add_proton(proton_2)
    proton_3 = Proton(0.05)
    proton_3.initialize(ti.Vector([1, 0, 0]), ti.Vector([0.05, -0.07, 0.02]))
    electronCloud.add_proton(proton_3)

    neutron_1 = Proton(0.065)
    neutron_1.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.15, 0.05, 0.03]))
    electronCloud.add_neutron(neutron_1)
    neutron_2 = Proton(0.065)
    neutron_2.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.01, 0.09, 0.05]))
    electronCloud.add_neutron(neutron_2)
    neutron_3 = Proton(0.065)
    neutron_3.initialize(ti.Vector([0, 1, 0]), ti.Vector([0.09, -0.04, 0.09]))
    electronCloud.add_neutron(neutron_3)

    cloud = []
    for i in range(n):
        cloud.append(Electron(0.01))

        u1 = ti.acos(2 * random.random() - 1) - pi/2
        u2 = 2 * pi * random.random()
        x = ti.cos(u1)*ti.cos(u2)
        y = ti.cos(u1)*ti.sin(u2)
        z = ti.sin(u1)

        pos = ti.Vector([x, y, z])
        vel = ti.Vector([0, 0, 0])
        #Now, I can't determine the initial velocity of each electron
        cloud[i].initialize(ti.Vector([0, 1, 1]), pos, vel)
        electronCloud.add_electron(cloud[i])
    # electronCloud.add_electron()

    return electronCloud


def main():
     ec = init()

     window = ti.ui.Window('Solar System', (800, 800), vsync=True)
     canvas = window.get_canvas()
     scene = ti.ui.Scene()
     camera = ti.ui.make_camera()

     camera.position(0.0, -10.0, 5.0)
     camera.lookat(0.0, 0.0, 0.0)
     camera.fov(20)


     while window.running:
        # ec.update(0.01)
        scene.set_camera(camera)
        ec.display(scene)
        scene.point_light(pos=(0.0, -5.0, 5.0), color=(0.5, 0.5, 0.5))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()