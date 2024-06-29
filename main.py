import time
import random
import numpy as np
import numba.cuda
from numba import cuda
import pygame
from pygame.locals import OPENGL, DOUBLEBUF
from OpenGL.GL import *
from OpenGL.GLU import *
from math import pi, sin, cos, asin, acos
from dataclasses import dataclass

from physics import calc_densities, calc_density_gradients, calc_pressure_force, update_positions 


@dataclass
class Particle:
    x: float
    y: float

particle_dtype = np.dtype([
    ('position', np.float32, (2,)),
    ('force', np.float32, (2,)),
    ('velocity', np.float32, (2,)),
    ('acceleration', np.float32, (2,)),
    ('density', np.float32),
    ('density_gradient', np.float32, (2,)),
    ('pressure', np.float32),
    ('pressure_force', np.float32, (2,)),
    ('mass', np.float32)
])

def draw_circle(x: float, y: float, radius: float):
    num_segments = 36

    glPushMatrix()
    glTranslate(x, y, 0)

    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)
    for i in range(num_segments + 1):
        angle = 2 * pi * i / num_segments
        glVertex3f(cos(angle) * radius, sin(angle) * radius, 0)
    glEnd()

    glPopMatrix()


def generate_particles(n: int):
    dim = int(np.sqrt(n))
    spacing = min(SCREEN_RESOLUTION) / (dim + 1) * 2
    for i in range(dim):
        for j in range(dim):
            x = (
                -SCREEN_RESOLUTION[0] + spacing * (i + 1) + random.uniform(-25, 25)
            ) / SCREEN_RESOLUTION[0]
            y = (
                -SCREEN_RESOLUTION[0] + spacing * (j + 1) + random.uniform(-25, 25)
            ) / SCREEN_RESOLUTION[1]
            particles.append(Particle(x, y))


particles = []
SCREEN_RESOLUTION = (800, 800)


def main():
    pygame.init()
    pygame.display.set_mode(SCREEN_RESOLUTION, DOUBLEBUF | OPENGL)

    n_particles = 400 
    generate_particles(n_particles)
    particles_array = np.zeros(len(particles), dtype=particle_dtype)

    # Fill the structured array with data from the particles list
    for i, p in enumerate(particles):
        particles_array[i]['position'] = [p.x, p.y]
        particles_array[i]['force'] = [0, 0]
        particles_array[i]['velocity'] = [0, 0]
        particles_array[i]['acceleration'] = [0, 0]
        particles_array[i]['density'] = 0
        particles_array[i]['density_gradient'] = [0, 0]
        particles_array[i]['pressure'] = 0
        particles_array[i]['pressure_force'] = [0, 0]
        particles_array[i]['mass'] = 1

    d_particles_array = cuda.to_device(particles_array)

    h = 0.15

    threads_per_block = 256
    blocks_per_grid = (particles_array.shape[0] + threads_per_block - 1) // threads_per_block

    normalized_width = 1.0
    normalized_height = 1.0

    radius = 0.01
    d_time = 1 / 60
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                quit()

        calc_densities[threads_per_block, blocks_per_grid](d_particles_array, h)
        calc_density_gradients[threads_per_block, blocks_per_grid](d_particles_array, h)
        calc_pressure_force[threads_per_block, blocks_per_grid](d_particles_array, h)
        update_positions[threads_per_block, blocks_per_grid](d_particles_array, h,normalized_width, normalized_height)

        particles_array = d_particles_array.copy_to_host()
        #print(particles_array['density'])

        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        for particle in particles_array:
            draw_circle(particle['position'][0], particle['position'][1], radius)

        pygame.display.flip()


if __name__ == "__main__":
    main()
