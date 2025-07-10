import pygame
import numpy as np
from pygame.locals import K_ESCAPE

class EgoController:
    """Keyboard controller for the ego vehicle.

    Two modes are supported:
    1. **Manual** – If *traffic_manager* is *None*, we generate throttle, brake, and
       steering commands ourselves using ← / → / ↑ / ↓ and SPACE keys.
    2. **Traffic‑Manager Autopilot** – If *traffic_manager* is provided we switch
       the ego vehicle to autopilot (so Carla’s Traffic Manager handles all
       low‑level control) **but** we still listen to ← / → keys and force a lane
       change whenever the requested lane exists.
    """

    def __init__(self, ego, traffic_manager=None):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        self.display = pygame.display.set_mode((250, 115))
        pygame.display.set_caption("Ego Vehicle Status")

        self.ego = ego
        self.tm = traffic_manager
        if self.tm is not None:
            self.ego.set_autopilot(True, self.tm.get_port())
            self.tm.ignore_lights_percentage(self.ego, 100.0)
            self.tm.ignore_signs_percentage(self.ego, 100.0)
            self.tm.vehicle_percentage_speed_difference(self.ego, -30) # Makes ego 30% faster

        self.last_collision_actor = "None"
        self.collision_count = 0
        self.current_steer = 0.0  # For smooth manual steering
        self.steer_rate = 0.05

    def update_status(self, elapsed_time):
        """Small on‑screen HUD for displaying vehicle status."""
        control = self.ego.get_control()
        velocity = self.ego.get_velocity()
        speed_kmh = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
        speed_limit = self.ego.get_speed_limit()
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Determine motion and assign color
        if control.reverse:
            speed_kmh = -speed_kmh
            motion, colour = "Reversing", (255, 140, 0)
        elif control.brake > 0.1:
            motion, colour = "Braking", (255, 50, 50)
        elif control.throttle > 0.1:
            motion, colour = "Accelerating", (0, 255, 0)
        else:
            motion, colour = "Coasting", (180, 180, 180)

        steer_dir = ("Left" if control.steer < -0.05 else
                     "Right" if control.steer > 0.05 else
                     "Straight")

        self.display.fill((30, 30, 30))
        self.display.blit(self.font.render(f"Speed: {speed_kmh:5.1f} km/h ({motion})", True, colour), (10, 10))
        self.display.blit(self.font.render(f"Speed Limit: {speed_limit:5.1f} km/h", True, (255, 215, 0)), (10, 27))
        self.display.blit(self.font.render(f"Steering: {control.steer:+.2f} ({steer_dir})", True, (135, 206, 250)), (10, 44))
        self.display.blit(self.font.render(f"Collisions: {self.collision_count}", True, (255, 80, 80)), (10, 61))
        self.display.blit(self.font.render(f"Last Collision: {self.last_collision_actor}", True, (255, 165, 0)), (10, 78))
        self.display.blit(self.font.render(f"Run Time: {hours:02}:{minutes:02}:{seconds:02}", True, (144, 238, 144)), (10, 95))
        pygame.display.flip()

    def log_collision(self, event):
        """Callback for collision events."""
        self.collision_count += 1
        self.last_collision_actor = event.other_actor.type_id

    def handle_vehicle_control(self):
        """Dispatch to manual or TM‑autopilot control.

        * Always listen for ESC / window close to allow clean exit (see
          *check_exit_event*).
        * When under autopilot we only react to ← → keys.
        """
        if self.tm is None:
            self._manual_control()
        else:
            self._autopilot_lane_change()

    def check_exit_event(self):
        """Check for exit events (ESC key or window close)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                return True
        return False

    def _autopilot_lane_change(self):
        """Request lane change if left or right arrow key is pressed."""
        pass

    def _manual_control(self):
        """Manual control of the ego vehicle using keyboard input."""
        keys = pygame.key.get_pressed()
        control = self.ego.get_control()

        # Throttle, brake, reverse logic
        vel = self.ego.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y, vel.z])
        speed_limit = max(60.0, self.ego.get_speed_limit()) / 3.6

        if keys[pygame.K_DOWN]:
            if speed > 0.1 and not control.reverse:
                control.throttle = 0.0
                control.brake = 1.0
                control.reverse = False
            else:
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

        control.brake = 1.0 if keys[pygame.K_SPACE] else control.brake
        control.hand_brake = False

        # Steering logic
        max_steer, min_steer, max_speed = 0.3, 0.05, 25.0  # 25 m/s ≈ 90 km/h
        sensitivity = max(min_steer, max_steer * (1 - speed / max_speed))
        target = 0.0
        if keys[pygame.K_LEFT] ^ keys[pygame.K_RIGHT]:
            target = -sensitivity if keys[pygame.K_LEFT] else sensitivity
        delta = np.clip(target - self.current_steer, -self.steer_rate, self.steer_rate)
        self.current_steer += delta
        control.steer = self.current_steer

        self.ego.apply_control(control)
