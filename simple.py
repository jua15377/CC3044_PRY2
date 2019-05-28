import pygame
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import glm
import pyassimp
import numpy
import math

# pygame

pygame.init()
pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
pygame.key.set_repeat(1, 10)


glClearColor(1, 0.18, 0.18, 1.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)

# shaders

vertex_shader = """
#version 460
layout (location = 0) in vec4 position;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 texcoords;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 color;
uniform vec4 light;
out vec4 vertexColor;
out vec2 vertexTexcoords;
void main()
{
    float intensity = dot(normal, normalize(light - position));
    vec4 incline = vec4(position.x+position.y*0.1,position.yz,1);
    gl_Position = projection * view * model * incline;
    vertexColor = color * intensity;
    vertexTexcoords = texcoords;
}
"""

fragment_shader = """
#version 460
layout (location = 0) out vec4 diffuseColor;
in vec4 vertexColor;
in vec2 vertexTexcoords;
uniform sampler2D tex;
void main()
{
    diffuseColor = vertexColor * texture(tex, vertexTexcoords);
}
"""

shader = shaders.compileProgram(
    shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
    shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
)
glUseProgram(shader)


# matrixes
model = glm.mat4(1)
view = glm.mat4(1)
light = glm.vec3(-10, 20, 30)
continuos_ligth = True

projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000.0)

glViewport(0, 0, 800, 800)


# scene = pyassimp.load('models/OBJ/spider.obj')
# scene = pyassimp.load('models/OBJ/10076_pisa_tower_v1_diffuse.obj')
scene = pyassimp.load('10076_pisa_tower_v1_max2009_it0.obj')


def glize(node):
    model = node.transformation.astype(numpy.float32)

    for mesh in node.meshes:
        material = dict(mesh.material.properties.items())
        texture = material['file']


        texture_surface = pygame.image.load(texture)
        texture_data = pygame.image.tostring(texture_surface,"RGB",1)
        width = texture_surface.get_width()
        height = texture_surface.get_height()
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        vertex_data = numpy.hstack((
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32)
        ))

        faces = numpy.hstack(
            numpy.array(mesh.faces, dtype=numpy.int32)
        )

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 9 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)


        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

        glUniformMatrix4fv(
            glGetUniformLocation(shader, "model"), 1 , GL_FALSE,
            model
        )
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "view"), 1 , GL_FALSE,
            glm.value_ptr(view)
        )
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "projection"), 1 , GL_FALSE,
            glm.value_ptr(projection)
        )

        diffuse = mesh.material.properties["diffuse"]

        glUniform4f(
            glGetUniformLocation(shader, "color"),
            *diffuse,
            1
        )
        if continuos_ligth:
            glUniform4f(
                glGetUniformLocation(shader, "light"),
                int(camera.x), int(camera.y), int(camera.z), 1
            )
        else:
            glUniform4f(
                glGetUniformLocation(shader, "light"),
                int(light.x), int(light.y), int(light.z), 1
            )

        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)


    for child in node.children:
        glize(child)


camera = glm.vec3(0, 5, 15)
camera_speed = 0.1
zoom_speed = 1
view_vec = glm.vec3(0, 5, 0)
radio = camera.z
rotation = 0
y_move = False
def process_input():
    global rotation, radio, continuos_ligth, y_move
    # print('radio', radio)
    # print(camera.z)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            return True
        if event.type == pygame.KEYDOWN:
            # z sen
            # x cos
            if event.key == pygame.K_LEFT:
                rotation += camera_speed
                camera.x = math.sin(rotation) * radio
                camera.z = math.cos(rotation) * radio
            if event.key == pygame.K_RIGHT:
                rotation -= camera_speed
                camera.x = math.sin(rotation) * radio
                camera.z = math.cos(rotation) * radio
            if event.key == pygame.K_UP:
                if camera.z >= 5: camera.z -= zoom_speed
            if event.key == pygame.K_DOWN:
                if camera.z < 20: camera.z += zoom_speed
            if event.key == pygame.K_r:
                print('radio set')
                if 3 < camera.z < 20:
                    radrio = camera.z
            #for light
            if event.key == pygame.K_l:
                continuos_ligth = not continuos_ligth
            if event.key == pygame.K_y:
                y_move = not y_move
            # move on y axis
            if event.key == pygame.K_UP and y_move:
                camera.y += camera_speed
                view_vec.y += camera_speed
                if camera.y >= 16:
                    camera.y = 16
            if event.key == pygame.K_DOWN and y_move:
                camera.y -= camera_speed
                view_vec.y -= camera_speed
                if camera.y <= -2:
                    camera.y = -2
    # print(camera.x, camera.y,  camera.z,)
    return False


done = False
while not done:
    glClearColor(0.8,0.8,0.8,0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    view = glm.lookAt(camera, view_vec, glm.vec3(0, 1, 0))
    glDisable(GL_BLEND)
    glize(scene.rootnode)

    done = process_input()
    clock.tick(60)
    pygame.display.flip()
