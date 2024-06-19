import bpy
import shutil
from math import radians as rad
import os
import sys
import yaml

repo_name = 'advancing-spectrum-anomaly-detection'

filepath = bpy.context.space_data.text.filepath
module_path = filepath[:filepath.find(repo_name)+len(repo_name)]
sys.path.append(os.path.abspath(module_path))
sys.path.append(os.path.join(os.path.abspath(module_path), 'blender-python'))

from blender_utils import AnchorPoint, EdgeLengths, init_materials
from blender_utils import add_light, add_camera, clear_scene, create_description_file
from blender_utils import create_cuboid_mesh, create_obstacle, link_objects_to_view_layer

clear_scene()

materials = init_materials(os.path.join(module_path, 'blender-python', 'textures'))

object_collection = [] # collect all objects (room + obstacles)

# Load scene attributes from yaml file
scene_attributes_path = os.path.join(module_path, 'blender-python', 'conf', 'scene_attributes.yaml')
with open(scene_attributes_path, 'r') as f:
    scene_attributes = yaml.load(f, Loader=yaml.FullLoader)

print(scene_attributes)


# define room
l,b,w = scene_attributes['room']['size']
room_size = EdgeLengths(l, b, w)
room_anchor_point = AnchorPoint(0, 0, 0)
room_mesh = create_cuboid_mesh(room_anchor_point, room_size, False, 'room')
room_object = bpy.data.objects.new('room_object', room_mesh)
print(materials['concrete'])
room_object.data.materials.append(materials['concrete'])
object_collection.append(room_object)

# define obstacles
obstacle_counter = 0

for obst_name in scene_attributes['obstacles']:
    x,y,z = scene_attributes['obstacles'][obst_name]['anchor_point']
    l,b,w = scene_attributes['obstacles'][obst_name]['edge_length']
    obstacle_counter, obstacle_object = create_obstacle(obstacle_counter,
                                                        AnchorPoint(x,y,z),
                                                        EdgeLengths(l,b,w),
                                                        materials,
                                                        scene_attributes['obstacles'][obst_name]['type'])
    object_collection.append(obstacle_object)

link_objects_to_view_layer(object_collection)

# add some light
light_counter = 0
light_counter = add_light(light_counter, (0.33*room_size.x, 0.67*room_size.y, 14), 4000)
light_counter = add_light(light_counter, (0.67*room_size.x, 0.33*room_size.y, 14), 4000)

# add a camera
cam_location = (80, 80, 118)
cam_rotation_euler = (rad(34), rad(0), rad(-227))
add_camera(cam_location, cam_rotation_euler)

# First, get path where this script is located
scenes_folder = os.path.join(os.path.abspath(module_path), 'scenes')

# find which number can be used for the next scene
scene_number = 0
while os.path.isdir(os.path.join(scenes_folder, f'scene{scene_number}')):
    scene_number += 1

output_path = os.path.join(scenes_folder, f'scene{scene_number}')
# Create the directory
os.mkdir(output_path)
filename = 'scene.xml'

bpy.ops.export_scene.mitsuba(filepath=os.path.join(output_path,filename), export_ids=True, axis_forward='Y', axis_up='Z')

create_description_file(output_path, room_size, object_collection=object_collection)

#copying config file
copy_file = 'scene_attributes.yaml'
file_name_dst = 'scene_attributes.yaml'
copy_src = os.path.join(module_path, 'blender-python', 'conf', copy_file)
copy_dst = os.path.join(output_path, file_name_dst)

dest = shutil.copyfile(copy_src, copy_dst)