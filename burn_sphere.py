from pxr import Usd, UsdGeom
from pathlib import Path
import omni.usd
import omni.replicator.core as rep
import pathlib
import warp as wp
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

stage = omni.usd.get_context().get_stage()
plastic = rep.create.sphere()
rep.create.light(light_type='dome')
cube = rep.create.cube(visible=False, semantics=[('class', 'cube')])
texture_path = "C:\\Users\\ryan\\Documents\\black_circle.png" 
num_samples = 1000

# generste n unit vectors to use for ray casting
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return  vec.reshape(npoints,3)

# constructs a convex hull of the prim's points and stores the hull as a mesh on the gpu
def init_hull_on_gpu(prim):
    hull = ConvexHull(prim.GetAttribute("points").Get())
    points = wp.array(hull.points, dtype=wp.vec3f, device="cuda")
    indices = wp.array(hull.simplices.reshape(hull.simplices.size), dtype=wp.int32, device="cuda")
    return wp.Mesh(points, indices)
    
# https://nvidia.github.io/warp/modules/runtime.html#warp.Mesh.id
@wp.kernel
def raycast(mesh: wp.uint64,
            ray_dirs: wp.array(dtype=wp.vec3f),
            ray_hit: wp.array(dtype=wp.vec3f),
            proj_rots: wp.array(dtype=wp.quat)):

    tid = wp.tid()
    ray_origin = wp.vec3(0.0, 0.0, 0.0)
    # even though the samples are normalized, theyre not normalized enough apparently
    ray_dir = wp.normalize(ray_dirs[tid])

    t = float(0.0)      # hit distance along ray
    u = float(0.0)      # hit face barycentric u
    v = float(0.0)      # hit face barycentric v
    sign = float(0.0)   # hit face sign
    n = wp.vec3()       # hit face normal
    f = int(0)          # hit face index

    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_origin, ray_dir, 100.0, t, u, v, sign, n, f):
      ray_hit[tid] = ray_dir*t # position where ray intersects face
      normal = n*(-1.0*wp.sign(sign)) # normal of the hit face, using sign to adjust for bug
      
      # check if normal is parallel to projection material
      axis = wp.cross(vec3(-1.0, 0.0, 0.0), normal)
      if (axis == vec3()):
          axis = vec3(0.0, 1.0, 0.0)

      # build quaternion rotation from projection_material initial direction to the current faces normal
      proj_rots[tid] = wp.quat_from_axis_angle(axis, wp.acos(wp.dot(vec3(-1.0, 0.0, 0.0), normal)))

# load samples and mesh on gpu for generating projection directions and positions
hull = init_hull_on_gpu(stage.GetPrimAtPath(plastic.get_outputs()['prims'][0].GetString() + "/Sphere"))
ray_dirs = wp.array(sample_spherical(num_samples), dtype=wp.vec3f, device="cuda")
ray_hits = wp.zeros(shape=num_samples, dtype=wp.vec3f, device="cuda")
proj_rots = wp.zeros(shape=num_samples, dtype=wp.quat, device="cuda")

# calculate positions and rotation for each instance of the projection materialr
wp.launch(kernel=raycast, dim=num_samples, inputs=[hull.id, ray_dirs, ray_hits, proj_rots], device="cuda")

# synchronize data and convert it for replicator
positions = [tuple(i.tolist()) for i in ray_hits.numpy()]
rotations = [tuple(R.from_quat(i).as_euler("XYZ", degrees=True).tolist()) for i in proj_rots.numpy()]
sem = [('class', 'shape')]

# Randomizer for scattering
def place_projection(positions, rotations):
    proxies = rep.get.prims(semantics=[('class', 'cube')])
    with proxies:
        rep.modify.pose(
            position=rep.distribution.sequence(positions),
            rotation=rep.distribution.sequence(rotations),
            scale=rep.distribution.uniform((.1, .1, .1), (.7, .7,.7)))
    return proxies.node
rep.randomizer.register(place_projection)

# Create the projection with the plane as the target
with plastic:
    proj1 = rep.create.projection_material(cube, sem)

# Modify the cube position, and update the projection
with rep.trigger.on_frame(num_frames=num_samples):
    rep.randomizer.place_projection(positions, rotations)
    with proj1:
        rep.modify.projection_material(diffuse=texture_path)


