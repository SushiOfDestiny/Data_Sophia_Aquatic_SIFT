import bpy

def point_cloud(ob_name, coords, edges=[], faces=[]):
    """Create point cloud object based on given coordinates and name.

    Keyword arguments:
    ob_name -- new object name
    coords -- float triplets eg: [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)]
    """

    # Create new mesh and a new object
    me = bpy.data.meshes.new(ob_name + "Mesh")
    ob = bpy.data.objects.new(ob_name, me)

    # Make a mesh from a list of vertices/edges/faces
    me.from_pydata(coords, edges, faces)

    # Display name and update the mesh
    ob.show_name = True
    ob.show_texture_space = True
    me.update()
    return ob

def draw_points(pt_list, ob_name):
    '''Draw points from a list of points into the current scene'''
    # Create the object
    pc = point_cloud(ob_name, pt_list)

    # Link object to the active collection
    bpy.context.collection.objects.link(pc)

def draw_points_in_camera_frame(pt_list, ob_name, cam):
    '''Draw points from a list of points given by their coordinates in the camera frame into the current scene'''
    pt_list_global = [cam.matrix_world @ pt for pt in pt_list]
    draw_points(pt_list_global, ob_name)

# distance to canvas = f/d, where f is the camera's focal length and d is the sensor size
# f = 0.05
# d = 0.036
# D = f/d