import time
import numpy as np
import numba.cuda
from numba import cuda
import pygame
from pygame.locals import OPENGL, DOUBLEBUF 
from OpenGL.GL import * 
from OpenGL.GLU import *
from math import pi, sin, cos, asin, acos
from dataclasses import dataclass

from physics import smoothing_kernel, smoothing_kernel_numpy

@dataclass
class Particle:
    x: float 
    y: float
    mass: float = 1

def draw_circle(x: float, y: float, radius: float):
    num_segments = 360

    glPushMatrix()
    glTranslate(x, y, 0)

    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)
    for i in range(num_segments + 1):
        angle = 2 * pi * i / num_segments
        glVertex3f(cos(angle) * radius, sin(angle) * radius, 0)
    glEnd()

    glPopMatrix()


def main():
    pygame.init()
    display = (800, 800)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    paricle = Particle(0, 0)
    gluPerspective(90, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    radius = 1
    theta = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                quit()

        x = cos(theta)
        y = sin(theta) 

        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        draw_circle(x, y, radius)
        pygame.display.flip()
        theta = (theta + 0.1) % (2 * pi)
        pygame.time.wait(10)


if __name__ == "__main__":
    main()
