import pygame
import numpy as np
from pygame.locals import K_ESCAPE

class EgoController:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        self.display = pygame.display.set_mode((250, 100))
        pygame.display.set_caption("Ego Vehicle Status")
        self.last_collision_actor = "None"
        self.collision_count = 0

    def update_status(self, ego, elapsed_time):
        velocity = ego.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_limit = ego.get_speed_limit()
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.display.fill((30, 30, 30))
        self.display.blit(self.font.render(f"Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255)), (10, 10))
        self.display.blit(self.font.render(f"Speed Limit: {speed_limit:.1f} km/h", True, (200, 200, 0)), (10, 27))
        self.display.blit(self.font.render(f"Collisions: {self.collision_count}", True, (255, 100, 100)), (10, 44))
        self.display.blit(self.font.render(f"Last Collision: {self.last_collision_actor}", True, (255, 150, 0)), (10, 61))
        self.display.blit(self.font.render(f"Run Time: {hours:02}:{minutes:02}:{seconds:02}", True, (0, 200, 255)), (10, 78))
        pygame.display.flip()
    
    def log_collision(self, event):
        self.collision_count += 1
        self.last_collision_actor = event.other_actor.type_id

    def handle_vehicle_control(self, ego):
        keys = pygame.key.get_pressed()
        control = ego.get_control()
        control.throttle = 1.0 if keys[pygame.K_UP] else (1.0 if keys[pygame.K_DOWN] else 0.0)
        control.reverse = keys[pygame.K_DOWN]
        control.steer = -0.1 if keys[pygame.K_LEFT] else 0.1 if keys[pygame.K_RIGHT] else 0.0
        control.brake = 1.0 if keys[pygame.K_SPACE] else 0.0
        control.hand_brake = False
        ego.apply_control(control)

    def check_exit_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                return True
        return False
