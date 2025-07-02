import os
import carla
import random
import time
import numpy as np
import pickle
import signal
import pygame
import subprocess
import psutil
from dotenv import load_dotenv

load_dotenv()

process = None  # Global variable to hold the CARLA server process
running = True # Global flag to control the simulation loop

# Signal handler to gracefully stop the simulation
def signal_handler(sig, frame):
    global running
    print("\nInterrupt received. Stopping simulation...")
    running = False

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def kill_zombie_carla_processes():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ''
            if any(keyword in cmd for keyword in ['carla', 'carlaue4', 'unrealengine', 'ue4']):
                print(f"Terminating CARLA process {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    print(f"Force killing CARLA process {proc.pid}...")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print("All zombie CARLA processes terminated.")

def main():
    global running
    ego = camera = lidar = imu_sensor = gnss_sensor = collision_sensor = None
    npc_vehicles = []
    frame_data = {} # Dictionary to store frame data for logging
    collision_count = 0

    # Connect to the CARLA server
    print("Connecting to CARLA server...")
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            client.load_world("Town06")
            world = client.get_world()
            print(f"Connected to CARLA server. Loaded {world.get_map().name}. Running simulation...")
            break
        except Exception:
            time.sleep(1)
    else:
        print("CARLA server connection timed out.")
        return

    # Set the world to synchronous mode for precise control    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0  # 30 FPS
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Initialize traffic manager
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.global_percentage_speed_difference(30) # Makes all vehicles 30% slower

    # Spawn the ego vehicle (Tesla Model 3)
    spawn_point = random.choice(spawn_points)
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    ego = world.spawn_actor(vehicle_bp, spawn_point)
    ego.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(ego, -30.0)  # Makes ego 30% faster
    traffic_manager.distance_to_leading_vehicle(ego, 1.0)
    traffic_manager.ignore_lights_percentage(ego, 100.0) # Ignore traffic lights to reduce uneventful frames
    traffic_manager.ignore_signs_percentage(ego, 100.0) # Ignore stop signs to reduce uneventful frames

    # Attach RGB camera sensor to vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('sensor_tick', str(1.0 / 30.0))
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)

    # Attach LiDAR sensor to vehicle
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('sensor_tick', str(1.0 / 30.0))
    lidar_bp.set_attribute('range', '50')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.5)), attach_to=ego)

    # Attach IMU sensor to vehicle
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1.0 / 30.0))
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
    
    # Attach GNSS sensor to vehicle
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(1.0 / 30.0))
    gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=ego)
    
    # Attach collision sensor to vehicle
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)

    # Set up spectator to follow the ego vehicle in third-person view
    spectator = world.get_spectator()
    def update_spectator():
        transform = ego.get_transform()
        forward_vector = transform.get_forward_vector()
        location = transform.location - forward_vector * 10 + carla.Location(z=5)
        rotation = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(location, rotation))

    # Callback functions to process and store sensor data
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4)).copy()
        frame_data.setdefault(image.frame, {})['camera_data'] = array[:, :, :3]

    def process_lidar(point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        frame_data.setdefault(point_cloud.frame, {})['lidar_data'] = data

    def process_imu(imu):
        frame_data.setdefault(imu.frame, {})['imu'] = {
            'acc': [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z],
            'gyro': [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
        }

    def process_gnss(gnss):
        frame_data.setdefault(gnss.frame, {})['gnss'] = {
            'lat': gnss.latitude,
            'lon': gnss.longitude,
            'alt': gnss.altitude
        }

    def process_collision(event):
        nonlocal collision_count
        collision_count += 1
        frame_data.setdefault(event.frame, {})['collision'] = {
            'other_actor': event.other_actor.type_id,
            'impact_point': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
        }

    # Register callbacks with sensors
    camera.listen(process_image)
    lidar.listen(process_lidar)
    imu_sensor.listen(process_imu)
    gnss_sensor.listen(process_gnss)
    collision_sensor.listen(process_collision)

    # Spawn 80 NPC vehicles
    random.shuffle(spawn_points)
    for _, spawn_point in enumerate(spawn_points[:80]):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(npc, 100.0)
            traffic_manager.ignore_signs_percentage(npc, 100.0)
            npc_vehicles.append(npc)
    
    # Initialize pygame window for status overlay
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 12)
    status_display = pygame.display.set_mode((250, 80))
    pygame.display.set_caption("Ego Vehicle Status")

    def update_status(vehicle, time):
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_limit = vehicle.get_speed_limit()
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        status_display.fill((30, 30, 30))
        speed_text = font.render(f"Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255))
        limit_text = font.render(f"Speed Limit: {speed_limit:.1f} km/h", True, (200, 200, 0))
        collision_text = font.render(f"Collisions: {collision_count}", True, (255, 100, 100))
        runtime_text = font.render(f"Run Time: {hours:02}:{minutes:02}:{seconds:02}", True, (0, 200, 255))
        
        status_display.blit(speed_text, (10, 10))
        status_display.blit(limit_text, (10, 27))
        status_display.blit(collision_text, (10, 44))
        status_display.blit(runtime_text, (10, 61))
        pygame.display.flip()

    start_sim_time = time.time() # Start time of simulation

    # Main simulation loop
    while running:
        try:
            elapsed_time = int(time.time() - start_sim_time)
            world.tick()
            update_spectator()
            update_status(ego, elapsed_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Status window closed. Stopping simulation...")
                    running = False

        except RuntimeError as e:
            print("Runtime error:", e)
            running = False

    # Cleanup: stop sensors and save data
    try:
        for sensor in [camera, lidar, imu_sensor, gnss_sensor, collision_sensor]:
            if sensor is not None:
                sensor.stop()
        time.sleep(1.0)
        with open('Logs/raw_frame_data.pkl', 'wb') as f:
            pickle.dump(frame_data, f)
        print("Raw frame data saved for post-processing.")
    except Exception as e:
        print("Error during cleanup:", e)

# Entry point for the simulation
if __name__ == '__main__':
    try:
        # Spawn the child process
        kill_zombie_carla_processes()
        print("Starting CARLA server...")
        process = subprocess.Popen(
            [os.environ["CARLA_PATH"], "--quality=low"],
            shell=False,
            preexec_fn=os.setsid  # Starts a new process group on Unix 
        )
        main()
    finally:
        print("Terminating CARLA server...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
        except Exception as e:
            print("Error while terminating CARLA:", e)
        kill_zombie_carla_processes()
        print("Simulation completed.")
