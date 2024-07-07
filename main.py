import time
import numpy as np
from numba import cuda
import pygame
from pygame.locals import OPENGL, DOUBLEBUF
from OpenGL.GL import *
from OpenGL.GLU import *
import dearpygui.dearpygui as dpg

from physics import calc_densities, calc_density_gradients, calc_pressure_force, update_positions, extract_positions
from graphics import set_colour, draw_circle, draw_rectangle
from simulation import generate_particles
from grid import Grid

SCREEN_RESOLUTION = (800, 800)
simulation_running = False
particle_array = None
d_particle_array = None

RECTANGLE_WIDTH = 0.4
RECTANGLE_HEIGHT = 0.1
rectangle_angle = 0
ROTATION_SPEED = 3  # radians per second

def start_simulation():
    global particle_array, d_particle_array, simulation_running, rectangle_angle

    n_particles = dpg.get_value("num_particles")
    particle_spacing = dpg.get_value("particle_spacing")
    stochasticity = dpg.get_value("stochasticity")

    particle_array = generate_particles(n_particles, SCREEN_RESOLUTION, particle_spacing, stochasticity)
    particle_array = particle_array[particle_array['mass'] != 0]
    d_particle_array = cuda.to_device(particle_array)
    print(particle_array)

    simulation_running = True

def pause_simulation():
    global simulation_running
    simulation_running = not simulation_running

TARGET_DENSITY = 8
PRESSURE_MULTIPLIER = 10
def main():
    global particle_array, d_particle_array, simulation_running, TARGET_DENSITY, PRESSURE_MULTIPLIER, rectangle_angle

    dpg.create_context()

    with dpg.window(label="Simulation Controls", autosize=True):
        dpg.add_slider_int(label="Number of Particles", default_value=200, min_value=10, max_value=10000, tag="num_particles")
        dpg.add_slider_float(label="Particle Radius", default_value=0.01, min_value=0.001, max_value=0.1, tag="particle_radius")
        dpg.add_slider_float(label="Particle Spacing", default_value=50, min_value=1, max_value=100, tag="particle_spacing")
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

    normalized_width = 1.0
    normalized_height = 1.0
    
    grid = Grid(2, 2, dpg.get_value("smoothing_radius"))
    while True:
        dpg.render_dearpygui_frame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                dpg.destroy_context()
                return
        if simulation_running and particle_array is not None:
            threads_per_block = 512 
            blocks_per_grid = (particle_array.shape[0] + threads_per_block - 1) // threads_per_block

            h = dpg.get_value("smoothing_radius")
            TARGET_DENSITY = dpg.get_value("target_density")
            PRESSURE_MULTIPLIER = dpg.get_value("pressure_multiplier")

            d_positions = cuda.device_array((particle_array.shape[0], 2), dtype=np.float32)

            sorted_particle_indices = grid.update_grid(d_particle_array)
            print(sorted_particle_indices)

            calc_densities[blocks_per_grid, threads_per_block](d_particle_array, sorted_particle_indices, grid.grid_start, grid.grid_end, h)
            cuda.synchronize()
            time.sleep(2)

            particle_array = d_particle_array.copy_to_host()
            print(particle_array)

            calc_density_gradients[blocks_per_grid, threads_per_block](d_particle_array, sorted_particle_indices, grid.grid_start, grid.grid_end, h)
            cuda.synchronize()

            particle_array = d_particle_array.copy_to_host()
            print(particle_array)
            calc_pressure_force[blocks_per_grid, threads_per_block](d_particle_array, sorted_particle_indices, grid.grid_start, grid.grid_end, h, TARGET_DENSITY, PRESSURE_MULTIPLIER)
            cuda.synchronize()

            particle_array = d_particle_array.copy_to_host()
            print(particle_array)
            update_positions[threads_per_block, blocks_per_grid](d_particle_array, 1/60, normalized_width, normalized_height)

            particle_array = d_particle_array.copy_to_host()
            print(particle_array)
            grid.update_grid(d_particle_array)
            extract_positions[blocks_per_grid, threads_per_block](d_particle_array, d_positions)

            # Copy only the positions back to the host
            positions = d_positions.copy_to_host()
            
            particle_array = d_particle_array.copy_to_host()
            print(particle_array)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            set_colour(1, 1, 1)
            for position in positions:
                draw_circle(position[0], position[1], dpg.get_value("particle_radius"))
            #set_colour(0, 1, 0)
            #rectangle_angle += ROTATION_SPEED * 1/60  # Assuming 60 FPS
            #draw_rectangle(0, 0, RECTANGLE_WIDTH, RECTANGLE_HEIGHT, rectangle_angle)

            pygame.display.flip()
            break

if __name__ == "__main__":
    main()
