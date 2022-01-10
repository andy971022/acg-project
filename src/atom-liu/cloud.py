import taichi as ti
import random
n = 10
pi = 3.141592653


class Atom:
    def __init__(self, radius, dim=3):
        self.radius = radius
        self.dim = dim
        self.color = ti.Vector.field(dim, ti.f32, shape=1)
        self.pos = ti.Vector.field(dim, ti.f32, shape=1)

    def display(self, scene):
        scene.particles(self.pos, self.radius, per_vertex_color=self.color)


@ti.data_oriented
class Proton(Atom):
    @ti.kernel
    def initialize(self, color: ti.template(), pos: ti.template()):
        self.color[0] = color
        self.pos[0] = pos


@ti.data_oriented
class Neutron(Atom):
    @ti.kernel
    def initialize(self, color: ti.template(), pos: ti.template()):
        self.color[0] = color
        self.pos[0] = pos


@ti.data_oriented
class Electron(Atom):
    def __init__(self, radius, dim=3):
        super().__init__(radius)
        self.vel = ti.Vector.field(dim, ti.f32, shape=1)

    @ti.kernel
    def initialize(self, color: ti.template(), pos: ti.template(),
                   vel: ti.template()):
        self.color[0] = color
        self.pos[0] = pos
        self.vel[0] = vel

    @ti.kernel
    def update(self, dt: ti.f32):
        self.pos[0] += self.vel[0] * dt
        sqr_sum = self.pos[0].norm_sqr()
        acc = -2.959e-4 * self.pos[0] / sqr_sum ** (3. / 2)
        self.vel[0] += acc * dt


@ti.data_oriented
class ElectronCloud:
    def __init__(self):
        self.protons = []
        self.neutrons = []
        self.electrons = []
        self.step = 0
        self.time = 0.0

    def add_proton(self, proton):
        self.protons.append(proton)

    def add_neutron(self, neutron):
        self.neutrons.append(neutron)

    def add_electron(self, electron):
        self.electrons.append(electron)

    # def add_electron(self):
    #     for i in range(n):
    #         self.electrons.append(Electron(0.01))
    #         u1 = ti.acos(2 * random.random() - 1) - pi / 2
    #         u2 = 2 * pi * random.random()
    #         x = ti.cos(u1) * ti.cos(u2)
    #         y = ti.cos(u1) * ti.sin(u2)
    #         z = ti.sin(u1)
    #         vel = ti.Vector([0, 0, 0])
    #         pos = ti.Vector([x, y, z])
    #         self.electrons[i].initialize(ti.Vector([0, 1, 1]), pos, vel)

    def update(self, dt=1):
        for i, p in enumerate(self.electrons):
            p.update(dt)

    def display(self, scene):
        for i in self.protons:
            i.display(scene)
        for j in self.neutrons:
            j.display(scene)
        for k in self.electrons:
            k.display(scene)