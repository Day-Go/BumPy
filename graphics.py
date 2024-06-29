from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np

def create_shader_program():
    vertex_shader = """
    #version 330
    in vec2 position;
    uniform float particle_radius;
    out vec2 v_texcoord;
    
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = (position + 1.0) / 2.0;
        gl_PointSize = particle_radius * 2.0 * 800.0; // Adjust for screen size
    }
    """

    fragment_shader = """
    #version 330
    in vec2 v_texcoord;
    uniform sampler2D density_texture;
    out vec4 fragColor;

    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5)
            discard;
        fragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black color
    }
    """
    
    vertex_shader_obj = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    fragment_shader_obj = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader_obj, fragment_shader_obj)
    
    return shader_program

def setup_vao():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    return vao

def create_density_texture(width, height):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, None)
    return texture

def update_density_texture(texture, density_data, width, height):
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, density_data)

def render_density_field(shader_program, vao, texture):
    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glBindTexture(GL_TEXTURE_2D, texture)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

def render_particles(shader_program, vao, particles_array, particle_radius):
    glUseProgram(shader_program)
    glBindVertexArray(vao)
    
    # Set the particle radius uniform
    radius_location = glGetUniformLocation(shader_program, "particle_radius")
    glUniform1f(radius_location, particle_radius)
    
    # Draw the particles
    glDrawArrays(GL_POINTS, 0, len(particles_array))
