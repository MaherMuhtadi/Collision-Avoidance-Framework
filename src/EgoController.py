import pygame
import numpy as np
from pygame.locals import K_ESCAPE

class EgoController:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        self.display = pygame.display.set_mode((250, 115))
        pygame.display.set_caption("Ego Vehicle Status")
        self.last_collision_actor = "None"
        self.collision_count = 0
        self.current_steer = 0.0  # Store last steering value for smoothing
        self.steer_rate = 0.05    # How fast to adjust steering

    def update_status(self, ego, elapsed_time):
        control = ego.get_control()
        velocity = ego.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_limit = ego.get_speed_limit()
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Determine motion and assign color
        if control.reverse:
            speed_kmh = -speed_kmh
            motion = "Reversing"
            motion_color = (255, 140, 0)  # Dark orange
        elif control.brake:
            motion = "Braking"
            motion_color = (255, 50, 50)    # Red
        elif control.throttle:
            motion = "Accelerating"
            motion_color = (0, 255, 0)    # Green
        else:
            motion = "Coasting"
            motion_color = (180, 180, 180)  # Gray

        # Steering direction
        steer = control.steer
        if steer < -0.05:
            steer_direction = "Left"
        elif steer > 0.05:
            steer_direction = "Right"
        else:
            steer_direction = "Straight"

        self.display.fill((30, 30, 30))  # Background

        # Draw status lines
        self.display.blit(self.font.render(f"Speed: {speed_kmh:.1f} km/h ({motion})", True, motion_color), (10, 10))
        self.display.blit(self.font.render(f"Speed Limit: {speed_limit:.1f} km/h", True, (255, 215, 0)), (10, 27))  # Gold
        self.display.blit(self.font.render(f"Steering: {steer:.2f} ({steer_direction})", True, (135, 206, 250)), (10, 44))  # Sky blue
        self.display.blit(self.font.render(f"Collisions: {self.collision_count}", True, (255, 80, 80)), (10, 61))  # Red
        self.display.blit(self.font.render(f"Last Collision: {self.last_collision_actor}", True, (255, 165, 0)), (10, 78))  # Orange
        self.display.blit(self.font.render(f"Run Time: {hours:02}:{minutes:02}:{seconds:02}", True, (144, 238, 144)), (10, 95))  # Light green

        pygame.display.flip()
    
    def log_collision(self, event):
        self.collision_count += 1
        self.last_collision_actor = event.other_actor.type_id

    def handle_vehicle_control(self, ego):
        keys = pygame.key.get_pressed()
        control = ego.get_control()

        # Current speed in m/s
        velocity = ego.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        speed_limit_kmh = max(60.0, ego.get_speed_limit())  # Ensure minimum speed limit of 60 km/h
        speed_limit = speed_limit_kmh / 3.6

        # Throttle and brake control
        if keys[pygame.K_DOWN]:
            if speed > 0 and not control.reverse:
                # Still moving forward — brake to stop
                control.throttle = 0.0
                control.brake = 1.0
                control.reverse = False
            else:
                # Vehicle is stopped — reverse
                control.brake = 0.0
                control.throttle = 1.0
                control.reverse = True
        elif keys[pygame.K_UP] and speed < speed_limit - 0.5:
            control.throttle = 1.0
            control.brake = 0.0
            control.reverse = False
        else:
            control.throttle = 0.0
            control.brake = 0.2 if speed > speed_limit + 0.5 else 0.0
            control.reverse = False

        # Brake control
        control.brake = 1.0 if keys[pygame.K_SPACE] else control.brake
        control.hand_brake = False

        # Steering sensitivity based on speed
        max_steer = 0.3
        min_steer = 0.05
        max_speed = 25.0 # ~90 km/h
        sensitivity = max(min_steer, max_steer * (1 - (speed / max_speed)))

        target_steer = 0.0
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            target_steer = -sensitivity
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            target_steer = sensitivity

        # Smooth steering change
        delta = target_steer - self.current_steer
        delta = np.clip(delta, -self.steer_rate, self.steer_rate)
        self.current_steer += delta
        control.steer = self.current_steer

        ego.apply_control(control)

    def check_exit_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                return True
        return False
