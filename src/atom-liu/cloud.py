import taichi as ti


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
    def initialize(self, color: ti.template(), pos: ti.template(), vel: ti.template()):
        self.color[0] = color
        self.pos[0] = pos
        self.vel[0] = vel


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

    def display(self, scene):
        for i in self.protons:
            i.display(scene)
        for j in self.neutrons:
            j.display(scene)
        for k in self.electrons:
            k.display(scene)
