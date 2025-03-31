import bpy
import math
import mathutils
import random
import json
import os
import time


# Enable GPUs function
def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus

enable_gpus("CUDA")


# Configuration
output_path = "/home/visione/Projects/BlenderScenarios/Sat/Dataset/Orbit_VR_256_dynlight/VIS_Test"    # Set your desired output path
image_format = 'PNG'                                                                            # File format (e.g., 'PNG', 'JPEG')
resolution = (256, 256)                                                                         # Resolution of the output images
num_frames = 30                                                                                # Number of frames for the satellite's orbit (one frame per degree)
camera_distance = 12                                                                            # Distance of the camera from the satellite
json_output_path = os.path.join(output_path, "transforms.json")                                 # JSON output file
cam_flag = ['V', 'R']


# Set render settings
bpy.ops.wm.open_mainfile(filepath="/home/visione/Projects/BlenderScenarios/Sat/cloudsat1.blend")

scene = bpy.context.scene
scene.render.image_settings.file_format = image_format
scene.render.resolution_x, scene.render.resolution_y = resolution


# Access objects
sat = bpy.data.objects["Pivot-Layer_0"]
camera = bpy.data.objects['Camera']
sun = bpy.data.objects["Light"]


# Set initial orientations
camera.rotation_mode = 'QUATERNION'
sun.rotation_mode = 'QUATERNION'


# Calculate camera_angle_x (horizontal FOV)
focal_length = camera.data.lens
sensor_width = camera.data.sensor_width
camera_angle_x = 2 * math.atan(sensor_width / (2 * focal_length))


# NeRF dataset JSON structure
nerf_data = {
    "camera_angle_x": camera_angle_x,
    "frames": []
}

# Initialize object pose to a random configuration in origin
random_quaternion = mathutils.Quaternion((random.uniform(-1, 1),
                                          random.uniform(-1, 1),
                                          random.uniform(-1, 1),
                                          random.uniform(-1, 1)))
random_quaternion.normalize()
obj2world = random_quaternion.to_matrix().to_4x4()

# For each orbit
for n_orbit, orbit in enumerate(cam_flag):

    # Animation loop
    for frame in range(num_frames):


        # Discretize angle according to number of frames required
        angle = math.radians(frame * (360 / num_frames))
        print(f"Frame {frame + n_orbit * num_frames}/{num_frames * 2}: Processing...")


        # V-Bar Camera-2-Obj Transform
        vcam2obj = mathutils.Matrix([
            [1,               0,                0,                                  0],
            [0, math.cos(angle), -math.sin(angle), -camera_distance * math.sin(angle)],
            [0, math.sin(angle),  math.cos(angle),  camera_distance * math.cos(angle)],
            [0,               0,                0,                                  1]
        ])


        # Static Transform for pointing
        rcam2rcam = mathutils.Matrix([
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 0],
            [ 0, 0, 0, 1]
        ])


        # R-Bar Camera-2-Obj Transform
        rcam2obj = mathutils.Matrix([
            [math.cos(angle), -math.sin(angle), 0, camera_distance * math.cos(angle)],
            [math.sin(angle),  math.cos(angle), 0, camera_distance * math.sin(angle)],
            [              0,                0, 1,                                 0],
            [              0,                0, 0,                                 1]
        ])


        # Select camera
        if orbit == 'V':
            cam2obj = obj2world @ vcam2obj
        elif orbit == 'R':
            cam2obj = obj2world @ rcam2obj @ rcam2rcam
        else:
            print(f"No available camera with code {cam_flag}")
            exit(5)


        # Define satellite pose
        t_sat = cam2obj.translation
        quat_sat = cam2obj.to_3x3().to_quaternion()


        # Define camera pose
        t_cam = cam2obj.translation
        quat_cam = cam2obj.to_3x3().to_quaternion()


        # Assign pose to camera object
        camera.location = (
            t_cam[0],
            t_cam[1],
            t_cam[2]
        )
        camera.rotation_quaternion = quat_cam

        # Modify light direction accordingly
        quat_sun = mathutils.Quaternion((random.uniform(-1, 1),
                                         random.uniform(-1, 1),
                                         random.uniform(-1, 1),
                                         random.uniform(-1, 1)))
        quat_sun.normalize()
        sun.rotation_quaternion = quat_sun
        sun2world = quat_sun.to_matrix().to_4x4()

        # Set output file path
        image_filename = f"{n_orbit * num_frames + frame:03d}.png"
        scene.render.filepath = os.path.join(output_path, image_filename)


        # Render and save image
        print(f"Rendering {image_filename}...")
        bpy.ops.render.render(write_still=True)
        time.sleep(1)


        # Add frame data to NeRF dataset
        nerf_data["frames"].append({
            "file_path": image_filename,
            "transform_matrix": [[cam2obj[row][col] for col in range(4)] for row in range(4)],
            "light_direction": [[sun2world[row][col] for col in range(4)] for row in range(4)]
        })


# Save the JSON file
os.makedirs(output_path, exist_ok=True)
with open(json_output_path, "w") as json_file:
    json.dump(nerf_data, json_file, indent=4)


print(f"Dataset saved to {output_path}")
bpy.ops.wm.save_mainfile()

