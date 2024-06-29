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
import dearpygui.dearpygui as dpg

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

def generate_particles(n: int, spacing: float, stochasticity: float):
    particles = []
    dim = int(np.sqrt(n))

    # Calculate the total width and height of the particle grid
    total_width = spacing * (dim - 1)
    total_height = spacing * (dim - 1)

    # Calculate offsets to center the grid
    offset_x = (SCREEN_RESOLUTION[0] - total_width) / 2
    offset_y = (SCREEN_RESOLUTION[1] - total_height) / 2

    for i in range(dim):
        for j in range(dim):
            # Calculate base position
            x = offset_x + spacing * i
            y = offset_y + spacing * j

            # Add stochasticity
            x += random.uniform(-stochasticity, stochasticity)
            y += random.uniform(-stochasticity, stochasticity)

            # Normalize to [-1, 1] range
            x_norm = (x / SCREEN_RESOLUTION[0]) * 2 - 1
            y_norm = (y / SCREEN_RESOLUTION[1]) * 2 - 1

            particles.append(Particle(x_norm, y_norm))

    return particles

SCREEN_RESOLUTION = (800, 800)
simulation_running = False
particles = []
particles_array = None
d_particles_array = None

def start_simulation():
    global particles, particles_array, d_particles_array, simulation_running
    
    n_particles = dpg.get_value("num_particles")
    particle_radius = dpg.get_value("particle_radius")
    particle_spacing = dpg.get_value("particle_spacing")
    stochasticity = dpg.get_value("stochasticity")
    
    particles = generate_particles(n_particles, particle_spacing, stochasticity)
    particles_array = np.zeros(len(particles), dtype=particle_dtype)

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
    simulation_running = True

def pause_simulation():
    global simulation_running
    simulation_running = not simulation_running

TARGET_DENSITY = 8
PRESSURE_MULTIPLIER = 10
def main():
    global particles_array, d_particles_array, simulation_running, TARGET_DENSITY, PRESSURE_MULTIPLIER

    dpg.create_context()

    with dpg.window(label="Simulation Controls", autosize=True):
        dpg.add_slider_int(label="Number of Particles", default_value=200, min_value=10, max_value=1000, tag="num_particles")
        dpg.add_slider_float(label="Particle Radius", default_value=0.01, min_value=0.001, max_value=0.1, tag="particle_radius")
        dpg.add_slider_float(label="Particle Spacing", default_value=50, min_value=10, max_value=100, tag="particle_spacing")
        dpg.add_slider_float(label="Stochasticity", default_value=0, min_value=0, max_value=50, tag="stochasticity")
        dpg.add_slider_float(label="Smoothing Radius (h)", default_value=0.25, min_value=0.01, max_value=1, tag="smoothing_radius")
        dpg.add_slider_float(label="Target Density", default_value=12, min_value=1, max_value=50, tag="target_density")
        dpg.add_slider_float(label="Pressure Multiplier", default_value=50, min_value=0.1, max_value=500, tag="pressure_multiplier")
        dpg.add_button(label="Start Simulation", callback=start_simulation)
        dpg.add_button(label="Pause/Resume", callback=pause_simulation)

    dpg.create_viewport(title="Fluid Simulation Controls", width=600, height=400)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    pygame.display.set_mode(SCREEN_RESOLUTION, DOUBLEBUF | OPENGL)

    threads_per_block = 256
    blocks_per_grid = 0

    normalized_width = 1.0
    normalized_height = 1.0

    while True:
        dpg.render_dearpygui_frame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                dpg.destroy_context()
                return
        if simulation_running and particles_array is not None:
            blocks_per_grid = (particles_array.shape[0] + threads_per_block - 1) // threads_per_block
            h = dpg.get_value("smoothing_radius")
            TARGET_DENSITY = dpg.get_value("target_density")
            PRESSURE_MULTIPLIER = dpg.get_value("pressure_multiplier")

            calc_densities[threads_per_block, blocks_per_grid](d_particles_array, h)
            calc_density_gradients[threads_per_block, blocks_per_grid](d_particles_array, h)
            calc_pressure_force[threads_per_block, blocks_per_grid](d_particles_array, h, TARGET_DENSITY, PRESSURE_MULTIPLIER)
            update_positions[threads_per_block, blocks_per_grid](d_particles_array, h, normalized_width, normalized_height)

            particles_array = d_particles_array.copy_to_host()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for particle in particles_array:
                draw_circle(particle['position'][0], particle['position'][1], dpg.get_value("particle_radius"))

            pygame.display.flip()

if __name__ == "__main__":
    main()
