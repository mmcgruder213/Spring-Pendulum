# Example file showing a circle moving on screen
import pygame
import math
import numpy as np
import random

# pygame setup
pygame.init()
info = pygame.display.Info()
WIDTH = info.current_w
HEIGHT = info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN |  pygame.DOUBLEBUF)
pygame.display.set_caption("Spring Pendulum Simulation")
clock = pygame.time.Clock()
running = True
font = pygame.font.Font(None, 36)

#Physics parameters
GRAVITY = 8000
SPRING = 250
AIR_RES = 0
SPRING_NAT_LEN = 100
MAX_LENGTH_VEL = 1500
MAX_ANGLE_VEL = 1000
DT = 0
PENDULUM_COUNT = 3
#Surface for drawing pendulum path
dot_surface = pygame.Surface((WIDTH, HEIGHT))
dot_surface.fill("black")
show_dots = False
prev_dot = (0, 0)

class Pendulum:
    def __init__(self):
        self.pivot = [WIDTH/2, 30]
        self.config = [2*SPRING_NAT_LEN, -np.pi/2 + np.pi*random.random()] #Length, angle
        self.velocity = [0.0, 15 - 30*random.random()] #Length, angle
    
    def draw(self, screen):
        try:
            coords = (self.get_length()*math.sin(self.config[1]), self.get_length()*math.cos(self.config[1]))
        except Exception as e:
            print(self.config[1])
        pivot_vector = pygame.Vector2(self.pivot[0], self.pivot[1])
        coord_vector = pygame.Vector2(self.pivot[0] + coords[0], self.pivot[1] + coords[1])
        pygame.draw.line(screen, "green", pivot_vector, coord_vector)
        pygame.draw.circle(screen, "red", coord_vector, 20)

    def get_length(self):
        return max(self.config[0], 1e-9)

    def get_speed(self):
        return max(math.sqrt(self.velocity[0]**2 + (self.get_length()*self.velocity[1])**2), 0.01)

    def get_vector(self):
        return np.array([-SPRING*(self.get_length() - SPRING_NAT_LEN), -GRAVITY, -AIR_RES * self.velocity[0]**2, -AIR_RES * (self.get_length() * self.velocity[1])**2]).T

class Playground:
    def __init__(self, count):
        self.count = count
        self.total_energy = np.array([0,0,0,0])
        self.pendulums = [Pendulum() for _ in range(count)]
        for i in range(1, count):
            coords = (self.pendulums[i-1].get_length()*math.sin(self.pendulums[i-1].config[1]), self.pendulums[i-1].get_length()*math.cos(self.pendulums[i-1].config[1]))
            self.pendulums[i].pivot[0] = coords[0] + self.pendulums[i-1].pivot[0]
            self.pendulums[i].pivot[1] = coords[1] + self.pendulums[i-1].pivot[1]
        coords = (self.pendulums[self.count-1].get_length()*math.sin(self.pendulums[self.count-1].config[1]), self.pendulums[self.count-1].get_length()*math.cos(self.pendulums[self.count-1].config[1]))
        global prev_dot
        prev_dot = pygame.Vector2(coords[0] + self.pendulums[self.count-1].pivot[0], coords[1] + self.pendulums[self.count-1].pivot[1])
        

    def draw(self, screen):
        global prev_dot
        self.pendulums[0].draw(screen)
        coords = (self.pendulums[0].get_length()*math.sin(self.pendulums[0].config[1]), self.pendulums[0].get_length()*math.cos(self.pendulums[0].config[1]))
        dot_coords = pygame.Vector2(self.pendulums[0].pivot[0] + coords[0], self.pendulums[0].pivot[1] + coords[1])
        if self.count == 1 and show_dots:
                pygame.draw.line(dot_surface, "white", prev_dot, dot_coords)
        prev_dot = dot_coords if self.count == 1 else prev_dot
        for i in range(1, self.count):
            coords = (self.pendulums[i-1].get_length()*math.sin(self.pendulums[i-1].config[1]), self.pendulums[i-1].get_length()*math.cos(self.pendulums[i-1].config[1]))
            self.pendulums[i].pivot[0] = coords[0] + self.pendulums[i-1].pivot[0]
            self.pendulums[i].pivot[1] = coords[1] + self.pendulums[i-1].pivot[1]
            self.pendulums[i].draw(screen)
            coords = (self.pendulums[i].get_length()*math.sin(self.pendulums[i].config[1]), self.pendulums[i].get_length()*math.cos(self.pendulums[i].config[1]))
            dot_coords = pygame.Vector2(self.pendulums[i].pivot[0] + coords[0], self.pendulums[i].pivot[1] + coords[1])
            if i == self.count-1 and show_dots:
                pygame.draw.line(dot_surface, "white", prev_dot, dot_coords)
            prev_dot = dot_coords if i == self.count-1 else prev_dot

    def compute_transform(self, i, j):
        return np.array([[np.cos(self.pendulums[j].config[1] - self.pendulums[i].config[1]), -(self.count - i) * np.cos(self.pendulums[i].config[1]), np.sign(self.pendulums[i].velocity[0]), 0],
                         [np.sin(self.pendulums[j].config[1] - self.pendulums[i].config[1]), (self.count - i) * np.sin(self.pendulums[i].config[1]), 0, np.sign(self.pendulums[i].velocity[1])]])
    
    def compute_correction(self, i):
        pend = self.pendulums[i]
        return np.array([pend.get_length()*(pend.velocity[1]**2), -2 * pend.velocity[0] * pend.velocity[1]]).T
    
    def get_accel(self, i):
        #Use 4th order Runge-Kutta to update velocity
        current_vector = self.pendulums[i].get_vector()
        if i == 0:
            if self.count > 1:
                below_vector = self.pendulums[i+1].get_vector()
                accel1 = self.compute_transform(i, i) @ current_vector - self.compute_transform(i, i + 1) @ below_vector
            else:
                accel1 = self.compute_transform(i, i) @ current_vector
        elif i < self.count-1:
            below_vector, above_vector = self.pendulums[i+1].get_vector(), self.pendulums[i-1].get_vector()
            accel1 = -self.compute_transform(i, i-1) @ above_vector + 2 * self.compute_transform(i, i) @ current_vector - self.compute_transform(i, i + 1) @ below_vector
        else:
            above_vector = self.pendulums[i-1].get_vector()
            accel1 = -self.compute_transform(i, i-1) @ above_vector + 2 * self.compute_transform(i, i) @ current_vector
        accel1 = accel1 + self.compute_correction(i)
        accel1 = accel1.T
        accel1[1] = accel1[1] / self.pendulums[i].get_length()
        return accel1
        
    def rotation(self, i):
        L = self.pendulums[i].get_length()
        return np.array([[math.sin(self.pendulums[i].config[1]), math.cos(self.pendulums[i].config[1])], [L * math.cos(self.pendulums[i].config[1]), -L * math.sin(self.pendulums[i].config[1])]])   

    def update(self):
        self.total_energy = np.array([0.0,0.0,0.0,0.0])
        new_configs = []
        for i in range(self.count):
            y = np.array([self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1]])
            derivatives = np.hstack((y[2:4], self.get_accel(i)))
            k1 = DT * derivatives
            y += k1/2
            self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1] = y

            y = np.array([self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1]])
            derivatives = np.hstack((y[2:4], self.get_accel(i)))
            k2 = DT * derivatives
            y += k2/2 - k1/2
            self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1] = y

            y = np.array([self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1]])
            derivatives = np.hstack((y[2:4], self.get_accel(i)))
            k3 = DT * derivatives
            y += k3 - k2/2
            self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1] = y

            y = np.array([self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1]])
            derivatives = np.hstack((y[2:4], self.get_accel(i)))
            k4 = DT * derivatives

            y -= k3
            self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1] = y
            y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
            y_new[2] = max(-MAX_LENGTH_VEL, min(MAX_LENGTH_VEL, y_new[2]))
            y_new[3] = max(-MAX_ANGLE_VEL, min(MAX_ANGLE_VEL, y_new[3])) 
            new_configs.append(y_new)
        x = np.array([0.0,0.0])
        for i in range(self.count):
            #Calculate energy
            x += x + np.array([self.pendulums[i].velocity[0], self.pendulums[i].velocity[1]]) @ self.rotation(i)
            kinetic = 0.5 * (x[0]**2 + x[1]**2)
            spring = 0.5 * SPRING * (self.pendulums[i].get_length() - SPRING_NAT_LEN)**2
            gravity = -GRAVITY * (self.pendulums[i].pivot[1] + self.pendulums[i].get_length()*math.cos(self.pendulums[i].config[1]))
            total = kinetic + spring + gravity
            self.total_energy += np.array([total, kinetic, spring, gravity])
            self.pendulums[i].config[0], self.pendulums[i].config[1], self.pendulums[i].velocity[0], self.pendulums[i].velocity[1] = new_configs[i]

        #Debug info
        text_surface1 = font.render(f"Energy: {self.total_energy[0]:.2f}", True, "white")
        text_rect1 = text_surface1.get_rect(topleft=(0, 0))
        screen.blit(text_surface1, text_rect1)
        text_surface2 = font.render(f"Kinetic Energy: {self.total_energy[1]:.2f}", True, "white")
        text_rect2 = text_surface2.get_rect(topleft=(0, 30))
        screen.blit(text_surface2, text_rect2)
        text_surface3 = font.render(f"Spring Energy: {self.total_energy[2]:.2f}", True, "white")
        text_rect3 = text_surface3.get_rect(topleft=(0, 60))
        screen.blit(text_surface3, text_rect3)
        text_surface4 = font.render(f"Gravitational: {self.total_energy[3]:.2f}", True, "white")
        text_rect4 = text_surface4.get_rect(topleft=(0, 90))
        screen.blit(text_surface4, text_rect4)


playground = Playground(PENDULUM_COUNT)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.KEYDOWN):
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dot_surface.fill("black")
            show_dots = not show_dots

    screen.fill("black")

    if show_dots:
        screen.blit(dot_surface, (0,0))
    playground.draw(screen)
    playground.update()
    DT = clock.tick(2000) / 1000
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.

pygame.quit()