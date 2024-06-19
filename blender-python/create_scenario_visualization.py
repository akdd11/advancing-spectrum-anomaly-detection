import bpy
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
from blender_utils import add_light, add_camera, clear_scene
from blender_utils import create_cuboid_mesh, create_obstacle, link_objects_to_view_layer
from blender_utils import add_radio_device


draw_jammer = True

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

# add the sensing units
for obst_name in scene_attributes['obstacles']:
    # the sensing units are placed in the middle of the long side of the obstacle
    x,y,z = scene_attributes['obstacles'][obst_name]['anchor_point']
    l,b,w = scene_attributes['obstacles'][obst_name]['edge_length']
    add_radio_device('SU', (x, y+b/2, 1.5))
    add_radio_device('SU', (x+l, y+b/2, 1.5))

for tx in scene_attributes['transmitters']:
    x,y,z = scene_attributes['transmitters'][tx]['anchor_point']
    add_radio_device('TX', (x, y, z))

for bs in scene_attributes['base_stations']:
    x,y,z = scene_attributes['base_stations'][bs]['anchor_point']
    add_radio_device('BS', (x, y, z))

if draw_jammer:
    for jam in scene_attributes['jammers']:
        x,y,z = scene_attributes['jammers'][jam]['anchor_point']
        add_radio_device('JAM', (x, y, z))

# add some light
light_counter = 0
light_counter = add_light(light_counter, (0.33*room_size.x, 0.33*room_size.y, 14), 3000)
light_counter = add_light(light_counter, (0.33*room_size.x, 0.67*room_size.y, 14), 3000)
light_counter = add_light(light_counter, (0.67*room_size.x, 0.33*room_size.y, 14), 3000)
light_counter = add_light(light_counter, (0.67*room_size.x, 0.67*room_size.y, 14), 3000)
light_counter = add_light(light_counter, (0.5*room_size.x, -0.25*room_size.y, 10), 400)

# add a camera
cam_location = (20, -60, 70)
cam_rotation_euler = (rad(49), rad(0), rad(0))
add_camera(cam_location, cam_rotation_euler)