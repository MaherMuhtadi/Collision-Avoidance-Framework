import os
import carla
import random
import time
import numpy as np
import json
import signal
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

def save_data(frame_data):
    valid_frame_data = {}
    dropped_frames = []
    total_frames_seen = set()

    print("Processing frame data...")
    for frame_id, data in frame_data.items():
        total_frames_seen.add(frame_id)
        if all(k in data for k in ["camera_data", "lidar_data", "imu", "gnss"]):
            cam_path = f'Camera/camera_{frame_id:06d}.png'
            data['camera_data'].save_to_disk(cam_path)
            data['camera_path'] = cam_path
            del data['camera_data']

            lidar_path = f'Lidar/lidar_{frame_id:06d}.csv'
            np.savetxt(lidar_path, data['lidar_data'], delimiter=',')
            data['lidar_path'] = lidar_path
            del data['lidar_data']

            valid_frame_data[frame_id] = data
        else:
            dropped_frames.append(frame_id)

    total_frames = len(total_frames_seen)
    stored_frames = len(valid_frame_data)
    dropped_count = len(dropped_frames)

    with open('Logs/simulation_log.json', 'w') as f:
        json.dump(valid_frame_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("Frame data log saved.")

    with open('Logs/frame_summary.json', 'w') as f:
        json.dump({
            "total_frames_played": total_frames,
            "stored_frames": stored_frames,
            "dropped_frames": dropped_count,
            "dropped_frame_ids": sorted(dropped_frames)
        }, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print("Frame drop summary saved.")

def main():
    global running
    ego = camera = lidar = imu_sensor = gnss_sensor = collision_sensor = None
    npc_vehicles = []
    pedestrians = []
    walker_controllers = []
    frame_data = {} # Dictionary to store frame data for logging

    # Connect to the CARLA server
    print("Connecting to CARLA server...")
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            world = client.get_world()
            print("Connected to CARLA server. Running simulation...")
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
    traffic_manager.set_global_distance_to_leading_vehicle(0.0)

    # Spawn the ego vehicle (Tesla Model 3) with collision evasion settings
    spawn_point = random.choice(spawn_points)
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    ego = world.spawn_actor(vehicle_bp, spawn_point)
    ego.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.distance_to_leading_vehicle(ego, 5.0)
    traffic_manager.ignore_lights_percentage(ego, 100.0) # Ignore traffic lights to reduce uneventful frames
    traffic_manager.ignore_signs_percentage(ego, 100.0) # Ignore stop signs to reduce uneventful frames
    traffic_manager.ignore_vehicles_percentage(ego, 0.0)
    traffic_manager.ignore_walkers_percentage(ego, 0.0)

    # Create output directories if they don't exist
    os.makedirs('Camera', exist_ok=True)
    os.makedirs('Lidar', exist_ok=True)
    os.makedirs('Logs', exist_ok=True)

    # Attach RGB camera sensor to vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('sensor_tick', str(1.0 / 30.0))  # 30 Hz
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)

    # Attach LiDAR sensor to vehicle
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('sensor_tick', str(1.0 / 30.0))  # 30 Hz
    lidar_bp.set_attribute('range', '50')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.5)), attach_to=ego)

    # Attach IMU sensor to vehicle
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1.0 / 30.0))  # 30 Hz
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
    
    # Attach GNSS sensor to vehicle
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(1.0 / 30.0))  # 30 Hz
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
        frame_data.setdefault(image.frame, {})['camera_data'] = image

    def process_lidar(point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
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
        frame = event.frame
        frame_data.setdefault(frame, {})['collision'] = {
            'other_actor': event.other_actor.type_id,
            'impact_point': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
        }

    # Register callbacks with sensors
    camera.listen(process_image)
    lidar.listen(process_lidar)
    imu_sensor.listen(process_imu)
    gnss_sensor.listen(process_gnss)
    collision_sensor.listen(process_collision)

    # Spawn 50 reckless NPC vehicles
    random.shuffle(spawn_points)
    for i, spawn_point in enumerate(spawn_points[:50]):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.distance_to_leading_vehicle(npc, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(npc, -30.0)
            traffic_manager.ignore_lights_percentage(npc, 100.0)
            traffic_manager.ignore_signs_percentage(npc, 100.0)
            traffic_manager.ignore_vehicles_percentage(npc, 100.0)
            traffic_manager.ignore_walkers_percentage(npc, 100.0)
            direction = random.choice([True, False])  # True = right, False = left
            traffic_manager.force_lane_change(npc, direction)
            npc_vehicles.append(npc)

    # Spawn 30 pedestrians to further increase collision chances
    for _ in range(30):
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        controller_bp = blueprint_library.find('controller.ai.walker')
        loc = random.choice(spawn_points).location
        loc.z += 1
        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
        if walker:
            pedestrians.append(walker)
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            walker_controllers.append(controller)
            controller.start()
            # Increase pedestrian riskiness
            num_movements = random.randint(3, 6)
            for _ in range(num_movements):
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                    controller.set_max_speed(2.0 + random.random() * 2.0)  # Walk fast

    # Main simulation loop
    while running:
        try:
            world.tick()
            update_spectator()
        except RuntimeError as e:
            print("Runtime error during simulation:", e)
            running = False
    
    # Cleanup: stop sensors and save data
    try:
        for sensor in [camera, lidar, imu_sensor, gnss_sensor, collision_sensor]:
            if sensor is not None:
                sensor.stop()
        time.sleep(1.0)
        save_data(frame_data)
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
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Gracefully terminate CARLA
            process.wait()
        except Exception as e:
            print("Error while terminating CARLA:", e)
        kill_zombie_carla_processes()
        print("Simulation completed.")