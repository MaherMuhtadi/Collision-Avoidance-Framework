import os
import carla
import random
import time
import numpy as np
import shelve
import signal
import subprocess
import psutil
import bisect
import json
from dotenv import load_dotenv
from EgoController import EgoController

load_dotenv()

fps = 30
window = 20 # Data collection window limited to last 20 sec
frame_buffer = [] # Buffers the allowed window of frames
seen_frames = []
lidar_settings_path = "lidar_settings.json"
process = None # Global variable to hold the CARLA server process
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

def next_save_file(base_dir="SensorData", prefix="run_"):
    os.makedirs(base_dir, exist_ok=True)
    nums = []
    for fname in os.listdir(base_dir):
        base, _ = os.path.splitext(fname)
        if base.startswith(prefix):
            s = base[len(prefix):]
            if s.isdigit():
                nums.append(int(s))
    n = max(nums) + 1 if nums else 1
    return os.path.join(base_dir, f"{prefix}{n}")

def save_lidar_settings(lidar):
    try:
        attrs = lidar.attributes if lidar is not None else {}
        channels = int(attrs.get('channels'))
        upper_fov = float(attrs.get('upper_fov'))
        lower_fov = float(attrs.get('lower_fov'))
        max_range = float(attrs.get('range'))
        rot_freq = float(attrs.get('rotation_frequency'))
        pps = int(attrs.get('points_per_second'))

        kv = {
            'CHANNELS': str(channels),
            'ROT_FREQ': str(rot_freq),
            'PPS': str(pps),
            'FOV_UP': f"{upper_fov}",
            'FOV_DOWN': f"{lower_fov}",
            'MAX_RANGE': f"{max_range}",
        }
        with open(lidar_settings_path, 'w') as f:
            json.dump(kv, f, indent=2)

    except Exception as e:
        print("Warning:", e)

def main():
    global running
    ego = camera = lidar = imu_sensor = collision_sensor = None
    npc_vehicles = []
    save_file = next_save_file()
    frame_data = shelve.open(save_file, writeback=True)

    def slide_buffer(frame_id):
        if len(frame_buffer) == window*fps:
            old_id = str(frame_buffer.pop(0))
            if old_id in frame_data:
                del frame_data[old_id]
        bisect.insort(frame_buffer, int(frame_id))

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
    settings.fixed_delta_seconds = 1.0 / fps  # 30 FPS
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
    ec = EgoController(ego=ego, traffic_manager=None)
        
    # Attach RGB camera sensor to vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('sensor_tick', str(1.0 / fps))
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)

    # Attach LiDAR sensor to vehicle
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('sensor_tick', str(1.0 / fps))
    lidar_bp.set_attribute('range', '50')
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=2.5)), attach_to=ego)
    save_lidar_settings(lidar)

    # Attach IMU sensor to vehicle
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1.0 / fps))
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
    
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
    def log_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4)).copy()
        frame_data.setdefault(str(image.frame), {})['camera_data'] = array[:, :, :3]
        if str(image.frame) not in seen_frames:
            seen_frames.append(str(image.frame))
            slide_buffer(str(image.frame))

    def log_lidar(point_cloud):
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        frame_data.setdefault(str(point_cloud.frame), {})['lidar_data'] = data
        if str(point_cloud.frame) not in seen_frames:
            seen_frames.append(str(point_cloud.frame))
            slide_buffer(str(point_cloud.frame))

    def log_imu(imu):
        frame_data.setdefault(str(imu.frame), {})['imu'] = {
            'acc': [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z],
            'gyro': [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
        }
        if str(imu.frame) not in seen_frames:
            seen_frames.append(str(imu.frame))
            slide_buffer(str(imu.frame))
    
    def log_collision(event):
        global running
        ec.track_collision(event)
        frame_data.setdefault(str(event.frame), {})['collision'] = 1
        if str(event.frame) not in seen_frames:
            seen_frames.append(str(event.frame))
            slide_buffer(str(event.frame))
        print(f"Collision detected at frame {event.frame}. Stopping simulation...")
        running = False
    
    # def log_steering():
    #     control = ego.get_control()
    #     frame_id = ego.get_world().get_snapshot().frame
    #     frame_data.setdefault(str(frame_id), {})['steer'] = float(control.steer)

    # Register callbacks with sensors
    camera.listen(log_image)
    lidar.listen(log_lidar)
    imu_sensor.listen(log_imu)
    collision_sensor.listen(log_collision)

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

    start_sim_time = time.time() # Start time of simulation

    # Main simulation loop
    while running:
        try:
            elapsed_time = int(time.time() - start_sim_time)
            if elapsed_time >= 1800:
                print("Simulation time limit reached (30 min). Stopping simulation...")
                running = False
            world.tick()
            ec.handle_vehicle_control()
            # log_actions()
            update_spectator()
            ec.update_status(elapsed_time)
            frame_data.sync()

            if ec.check_exit_event():
                print("Stopping simulation...")
                running = False

        except RuntimeError as e:
            print("Runtime error:", e)
            running = False

    # Cleanup: stop sensors and save data
    try:
        for sensor in [camera, lidar, imu_sensor, collision_sensor]:
            if sensor is not None:
                sensor.stop()
        time.sleep(1.0)
        frame_data.close()
        print(f"Saved sensor readings to {save_file}.")
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
