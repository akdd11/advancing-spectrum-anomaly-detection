import os

import bpy


# Classes -----------------------------------------------------------------------------------------

class Something3D():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

class AnchorPoint(Something3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

class EdgeLengths(Something3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)


# Functions ---------------------------------------------------------------------------------------

def add_light(light_counter, location, energy):
    light_data = bpy.data.lights.new(f'light{light_counter}', type='POINT')
    light = bpy.data.objects.new(f'light{light_counter}', light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    bpy.context.collection.objects[f'light{light_counter}'].data.energy = energy

    return light_counter + 1

def add_camera(location, rotation_euler):
    """Add a camera to the scene.
    
    location : tuple
        The location of the camera (in meters).
    rotation_euler : tuple
        The rotation of the camera (in radians).
    """

    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = rotation_euler


def clear_scene():
    """Clear the current Blender scene."""
    scene = bpy.context.scene

    for c in scene.collection.children:
        scene.collection.children.unlink(c)


def create_cuboid_mesh(anchor_point: AnchorPoint, edge_lengths: EdgeLengths, create_top_face: bool, name: str):
    """Create a cuboid mesh.
    
    Args:
        anchor_point :  AnchorPoint
            The anchor point of the cuboid.
        edge_lengths : EdgeLengths
            The edge lengths of the cuboid.
        create_top_face : bool
            Whether to create the top face of the cuboid.
        name : str
            The name of the mesh.
    """

    vertices = [(anchor_point.x, anchor_point.y, anchor_point.z),
                (anchor_point.x+edge_lengths.x, anchor_point.y, anchor_point.z),
                (anchor_point.x, anchor_point.y+edge_lengths.y, anchor_point.z),
                (anchor_point.x+edge_lengths.x, anchor_point.y+edge_lengths.y, anchor_point.z),
                (anchor_point.x, anchor_point.y, anchor_point.z+edge_lengths.z),
                (anchor_point.x+edge_lengths.x, anchor_point.y, anchor_point.z+edge_lengths.z),
                (anchor_point.x, anchor_point.y+edge_lengths.y, anchor_point.z+edge_lengths.z),
                (anchor_point.x+edge_lengths.x, anchor_point.y+edge_lengths.y, anchor_point.z+edge_lengths.z)]

    edges = []
    faces = [(0,1,3,2), (0,1,5,4), (1,3,7,5), (3,7,6,2), (0,2,6,4)]

    if create_top_face:
        faces = faces + [(4,5,7,6)]

    mesh = bpy.data.meshes.new(name+'mesh')
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()

    return mesh


def create_obstacle(obstacle_counter, anchor_point, edge_lengths, materials, material):
    """Add an obstacle to the scene.
    
    obstacle_counter : int
        The number of obstacles in the scene.
    anchor_point : AnchorPoint
        The anchor point of the obstacle.
    edge_lengths : EdgeLengths
        The edge lengths of the obstacle.
    material : str
        The material of the obstacle.
    """
    name = f'obstacle{obstacle_counter}'
    obstacle_mesh = create_cuboid_mesh(anchor_point, edge_lengths, True, name)
    obstacle_object = bpy.data.objects.new(f'{name}_object', obstacle_mesh)
    obstacle_object.data.materials.append(materials[material])
    return (obstacle_counter + 1, obstacle_object)


def create_description_file(output_path, room_size, **kwargs):
    """Create a description file for the scenario.
    
    output_path : str
        The path to the output directory.
    room_size : EdgeLengths
        The edge lengths of the room.
    """

    description_file =  open(os.path.join(output_path, 'description.txt'), 'w')
    if 'object_collection' in kwargs:
        object_collection = kwargs['object_collection']
        print('Scenario with obstacles\n', file=description_file)
        print(f'Room size: {room_size}', file=description_file)
        print(f'Number of obstacles: {len(object_collection)-1}', file=description_file)
    else:
        print('Scenario without obstacles\n', file=description_file)
        print(f'Room size: {room_size}', file=description_file)
    
    description_file.close()


def init_materials(texture_path, initialize_textures=False):
    """The materials are initialized and assigned with textures for
    making nice visualizations.
    
    texture_path : str
        The path to the texture folder.
    initialize_textures : bool
        Whether to initialize the textures. The textures are not part of the
        repository and need to be downloaded manually. Please refer to the README for
        more information.
    """

    materials = {}
    materials['concrete'] = bpy.data.materials.new(name='itu_concrete')
    materials['metal'] = bpy.data.materials.new(name='itu_metal')

    # Concrete ------------------------------------------------------------------------------------

    if initialize_textures:

        # Adjust textures of the materials
        materials['concrete'].use_nodes = True
        nodes_concrete = materials['concrete'].node_tree.nodes

        # Clear existing nodes
        for node in nodes_concrete:
            nodes_concrete.remove(node)

        # Add Principled BSDF shader
        bsdf_shader_concrete = nodes_concrete.new(type='ShaderNodeBsdfPrincipled')

        # Add Image Texture node
        texture_nodes_concrete = {k: nodes_concrete.new(type='ShaderNodeTexImage') for k in ['color', 'metalness', 'roughness']}

        # Set the image path
        texture_path_concrete = os.path.join(texture_path, 'ConcretePoured')
        texture_nodes_concrete['color'].image = bpy.data.images.load(os.path.join(texture_path_concrete,
                                                                                'ConcretePoured001_COL_2K_METALNESS.png'))
        texture_nodes_concrete['metalness'].image = bpy.data.images.load(os.path.join(texture_path_concrete,
                                                                                    'ConcretePoured001_METALNESS_2K_METALNESS.png'))
        texture_nodes_concrete['roughness'].image = bpy.data.images.load(os.path.join(texture_path_concrete,
                                                                                    'ConcretePoured001_ROUGHNESS_2K_METALNESS.png'))

        # Link nodes: texture color to Principled BSDF base color
        materials['concrete'].node_tree.links.new(texture_nodes_concrete['color'].outputs['Color'],
                                                bsdf_shader_concrete.inputs['Base Color'])
        materials['concrete'].node_tree.links.new(texture_nodes_concrete['metalness'].outputs['Color'],
                                                bsdf_shader_concrete.inputs['Metallic'])
        materials['concrete'].node_tree.links.new(texture_nodes_concrete['roughness'].outputs['Color'], 
                                                bsdf_shader_concrete.inputs['Roughness'])
        
        # Add Material Output node
        material_output_concrete = materials['concrete'].node_tree.nodes.new(type='ShaderNodeOutputMaterial')

        # Link BSDF output to Surface input
        materials['concrete'].node_tree.links.new(bsdf_shader_concrete.outputs['BSDF'], material_output_concrete.inputs['Surface'])
        
        # Metal ---------------------------------------------------------------------------------------

        # Adjust textures of the materials
        materials['metal'].use_nodes = True
        nodes_metal = materials['metal'].node_tree.nodes

        # Clear existing nodes
        for node in nodes_metal:
            nodes_metal.remove(node)

        # Add Principled BSDF shader
        bsdf_shader_metal = nodes_metal.new(type='ShaderNodeBsdfPrincipled')

        # Add Image Texture nodes
        texture_nodes_metal = {k: nodes_metal.new(type='ShaderNodeTexImage') for k in ['color', 'metalness', 'roughness']}

        # Set the image path
        texture_path_metal = os.path.join(texture_path, 'MetalGalvanizedSteelWorn')
        texture_nodes_metal['color'].image = bpy.data.images.load(os.path.join(texture_path_metal,
                                                                            'MetalGalvanizedSteelWorn001_COL_2K_METALNESS.jpg'))
        texture_nodes_metal['metalness'].image = bpy.data.images.load(os.path.join(texture_path_metal,
                                                                                'MetalGalvanizedSteelWorn001_METALNESS_2K_METALNESS.jpg'))
        texture_nodes_metal['roughness'].image = bpy.data.images.load(os.path.join(texture_path_metal,
                                                                                'MetalGalvanizedSteelWorn001_ROUGHNESS_2K_METALNESS.jpg'))

        # Link nodes: texture color to Principled BSDF base color
        materials['metal'].node_tree.links.new(texture_nodes_metal['color'].outputs['Color'],
                                                bsdf_shader_metal.inputs['Base Color'])
        materials['metal'].node_tree.links.new(texture_nodes_metal['metalness'].outputs['Color'],
                                                bsdf_shader_metal.inputs['Metallic'])
        materials['metal'].node_tree.links.new(texture_nodes_metal['roughness'].outputs['Color'], 
                                                bsdf_shader_metal.inputs['Roughness'])
        
        # Add Material Output node
        material_output_metal = materials['metal'].node_tree.nodes.new(type='ShaderNodeOutputMaterial')

        # Link BSDF output to Surface input
        materials['metal'].node_tree.links.new(bsdf_shader_metal.outputs['BSDF'], material_output_metal.inputs['Surface'])

    return materials


def add_radio_device(type, location, diameter=1.5):
    """
    Add a radio device represented by a sphere to the scene.
    
    type : str
        The type of the radio device (SU, TX, JAM, BS).
    location : tuple
        The location of the radio device (in meters).
    diameter : float
        The diameter of the sphere (in meters).
    """

    if type not in ['SU', 'TX', 'JAM', 'BS']:
        raise ValueError("Invalid type. Choose from 'SU', 'TX', 'JAM'.")

    colors = {'SU': (247/255, 199/255, 9/255, 1),
              'TX': (127/255, 255/255, 91/255, 1),
              'JAM': (255/255, 55/255, 55/255, 1),
              'BS': (0/255, 127/255, 255/255, 1)}

    # Add UV Sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=diameter/2, location=location)

    # Get the newly added sphere object
    sphere_obj = bpy.context.object

    # Set the sphere's material color
    if sphere_obj.data.materials:
        # If there's already a material, use it
        sphere_material = sphere_obj.data.materials[0]
    else:
        # Check if the material already exists
        sphere_material = bpy.data.materials.get(f'{type}_Material')

        if sphere_material is None:
            # If it doesn't exist, create a new material
            sphere_material = bpy.data.materials.new(name=f'{type}_Material')
        sphere_obj.data.materials.append(sphere_material)

    # Set material color
    sphere_material.use_nodes = True
    nodes = sphere_material.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs['Base Color'].default_value = colors[type]
    else:
        print("Error: Principled BSDF node not found.")


def link_objects_to_view_layer(object_collection):
    view_layer = bpy.context.view_layer

    for obj in object_collection:
        view_layer.active_layer_collection.collection.objects.link(obj)