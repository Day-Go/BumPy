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

from physics import smoothing_kernel

@dataclass
class Particle:
    x: float 
    y: float
    velocity: float = 0
    acceleration: float = 0
    density: float = 0
    mass: float = 1

def draw_circle(x: float, y: float, radius: float):
    num_segments = 36  # Reduced for performance, adjust as necessary

    glPushMatrix()
    glTranslate(x, y, 0)

    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)  # Center point
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
            x = (-SCREEN_RESOLUTION[0] + spacing * (i + 1) + random.uniform(-25, 25)) / SCREEN_RESOLUTION[0]
            y = (-SCREEN_RESOLUTION[0] + spacing * (j + 1) + random.uniform(-25, 25)) / SCREEN_RESOLUTION[1]
            particles.append(Particle(x, y))

particles = []
SCREEN_RESOLUTION = (800, 800)
def main():
    pygame.init()
    pygame.display.set_mode(SCREEN_RESOLUTION, DOUBLEBUF|OPENGL)

    n_particles = 300
    generate_particles(n_particles)
    positions = np.array([[p.x, p.y] for p in particles], dtype=np.float32)
    masses = np.array([p.mass for p in particles], dtype=np.float32)
    densities = np.array([p.density for p in particles], dtype=np.float32)
    field_values = np.random.rand(n_particles).astype(np.float32)  # Assuming field_values still needs random initialization
    outputs = np.zeros(n_particles, dtype=np.float32)
    h = 0.1

    d_positions = cuda.to_device(positions)
    d_masses = cuda.to_device(masses)
    d_densities = cuda.to_device(densities)
    d_field_values = cuda.to_device(field_values)
    d_outputs = cuda.to_device(outputs)

    threads_per_block = 256
    blocks_per_grid = (positions.shape[0] + threads_per_block - 1) // threads_per_block

    radius = 0.01
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        for particle in particles:
            draw_circle(particle.x, particle.y, radius)

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    main()
