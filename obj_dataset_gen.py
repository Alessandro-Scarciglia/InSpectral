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
image_format = 'PNG'                                                                                      # File format (e.g., 'PNG', 'JPEG')
resolution = (256, 256)                                                                                   # Resolution of the output images
num_frames = 15                                                                                           # Number of frames for the satellite's orbit (one frame per degree)
camera_distance = 12                                                                                      # Distance of the camera from the satellite
json_output_path = os.path.join(output_path, "transforms.json")                                           # JSON output file
cam_flag = ['V', 'R']


# Set render settings
bpy.ops.wm.open_mainfile(filepath="/home/visione/Projects/BlenderScenarios/Sat/cloudsat1.blend")

scene = bpy.context.scene
scene.render.image_settings.file_format = image_format
scene.render.resolution_x, scene.render.resolution_y = resolution


# Access objects
sat = bpy.data.objects["Layer_0"]
camera = bpy.data.objects['Camera']
sun = bpy.data.objects["Light"]

# Set Object Index for Masking
sat.pass_index = 1
bpy.context.view_layer.use_pass_object_index = True

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

# # Initialize object pose to a random configuration in origin
# random_quaternion = mathutils.Quaternion((random.uniform(-1, 1),
#                                           random.uniform(-1, 1),
#                                           random.uniform(-1, 1),
#                                           random.uniform(-1, 1)))
# random_quaternion.normalize()
# obj2world = random_quaternion.to_matrix().to_4x4()

# For each orbit
for n_orbit, orbit in enumerate(cam_flag):

    # Animation loop
    for frame in range(num_frames):


        # Discretize angle according to number of frames required
        angle = math.radians(frame * (360 / num_frames))
        print(f"Frame {frame + n_orbit * num_frames}/{num_frames * 2}: Processing...")

        # Initialize object pose to a random configuration in origin
        random_quaternion = mathutils.Quaternion((random.uniform(-1, 1),
                                                random.uniform(-1, 1),
                                                random.uniform(-1, 1),
                                                random.uniform(-1, 1)))
        random_quaternion.normalize()
        obj2world = random_quaternion.to_matrix().to_4x4()


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

        base_filename = f"{n_orbit * num_frames + frame:03d}"
        scene.render.filepath = os.path.join(output_path, f"{base_filename}.png")

        bpy.context.view_layer.use_pass_z = True
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        for node in nodes:
            nodes.remove(node)

        render_layers = nodes.new(type="CompositorNodeRLayers")
        render_layers.location = (-400, 0)

        map_range = nodes.new(type="CompositorNodeMapRange")
        map_range.inputs[1].default_value = 8.0
        map_range.inputs[2].default_value = 16.0
        map_range.inputs[3].default_value = 1.0
        map_range.inputs[4].default_value = 0.0
        map_range.location = (0, -100)

        depth_output = nodes.new(type="CompositorNodeOutputFile")
        depth_output.base_path = output_path
        depth_output.file_slots[0].path = f"{base_filename}_d.png"
        depth_output.location = (300, -100)

        links.new(render_layers.outputs["Depth"], map_range.inputs[0])
        links.new(map_range.outputs[0], depth_output.inputs[0])

        id_mask = nodes.new(type="CompositorNodeIDMask")
        id_mask.index = 1
        id_mask.location = (0, -200)

        mask_output = nodes.new(type="CompositorNodeOutputFile")
        mask_output.base_path = output_path
        mask_output.file_slots[0].path = f"{base_filename}_m.png"
        mask_output.location = (300, -200)

        links.new(render_layers.outputs["IndexOB"], id_mask.inputs["ID value"])
        links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])

        bpy.ops.render.render(write_still=True)
        time.sleep(1)

        os.rename(src=os.path.join(output_path, f"{base_filename}_d.png" + "0001.png"),
                  dst=os.path.join(output_path, f"{base_filename}_d.png"))
        
        os.rename(src=os.path.join(output_path, f"{base_filename}_m.png" + "0001.png"),
                  dst=os.path.join(output_path, f"{base_filename}_m.png"))

        nerf_data["frames"].append({
            "file_path": f"{base_filename}.png",
            "depth_path": f"{base_filename}_d.png",
            "mask_path": f"{base_filename}_m.png",
            "transform_matrix": [[cam2obj[row][col] for col in range(4)] for row in range(4)],
            "light_direction": [[sun2world[row][col] for col in range(4)] for row in range(4)]
        })


# Save the JSON file
os.makedirs(output_path, exist_ok=True)
with open(json_output_path, "w") as json_file:
    json.dump(nerf_data, json_file, indent=4)


print(f"Dataset saved to {output_path}")
bpy.ops.wm.save_mainfile()

