import open3d as o3d
import numpy as np

# -----------------------
# STEP 1: Load and visualize 
# -----------------------
mesh = o3d.io.read_triangle_mesh("091_W_Aya_30K.obj")  # замените на свой файл
mesh.compute_vertex_normals()

print("=== Step 1: Original Model ===")
print("Vertices:", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))
print("Has colors:", mesh.has_vertex_colors())
print("Has normals:", mesh.has_vertex_normals())

o3d.visualization.draw_geometries([mesh], window_name="Step 1: Original Mesh")

# -----------------------
# STEP 2: Convert to point cloud
# -----------------------
pcd = mesh.sample_points_poisson_disk(number_of_points=80000)
print("\n=== Step 2: Point Cloud ===")
print("Points:", len(pcd.points))
print("Has colors:", pcd.has_colors())

o3d.visualization.draw_geometries([pcd], window_name="Step 2: Point Cloud")

# -----------------------
# STEP 3: Surface reconstruction (Poisson)
# -----------------------
print("\n=== Step 3: Poisson Reconstruction ===")
pcd.estimate_normals(fast_normal_computation=True)
mesh_recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
bbox = pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center())
mesh_crop = mesh_recon.crop(bbox)
mesh_crop.compute_vertex_normals()

print("Vertices:", len(mesh_crop.vertices))
print("Triangles:", len(mesh_crop.triangles))
print("Has colors:", mesh_crop.has_vertex_colors())

o3d.visualization.draw_geometries([mesh_crop], window_name="Step 3: Reconstructed Mesh")

# -----------------------
# STEP 4: Voxelization
# -----------------------
print("\n=== Step 4: Voxelization ===")
voxel_size = 0.05
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

print("Voxels:", len(voxel_grid.get_voxels()))
o3d.visualization.draw_geometries([voxel_grid], window_name="Step 4: Voxel Grid")

# -----------------------
# STEP 5: Add cutting plane
# -----------------------
print("\n=== Step 5: Add Plane ===")
bbox = mesh_crop.get_axis_aligned_bounding_box()
center = bbox.get_center()
plane_size = max(bbox.get_extent()) * 1.5

plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=0.02, depth=plane_size)
plane.translate([center[0]-plane_size/2, center[1]-0.01, center[2]-plane_size/2])
plane.paint_uniform_color([0.8, 0.2, 0.2])

o3d.visualization.draw_geometries([mesh_crop, plane], window_name="Step 5: Plane through Model")
print("Plane placed at Y =", center[1])

# -----------------------
# STEP 6: Clipping
# -----------------------
print("\n=== Step 6: Clipping ===")
vertices = np.asarray(mesh_crop.vertices)
mask = vertices[:, 1] < center[1]  # оставляем только нижнюю половину
clipped_vertices = vertices[mask]

clipped_mesh = o3d.geometry.TriangleMesh()
clipped_mesh.vertices = o3d.utility.Vector3dVector(clipped_vertices)

# Для наглядности можно пересоздать треугольники через Poisson, но проще показать как облако точек:
clipped_pcd = o3d.geometry.PointCloud()
clipped_pcd.points = o3d.utility.Vector3dVector(clipped_vertices)
clipped_mesh.compute_vertex_normals()
clipped_mesh.paint_uniform_color([0.3, 0.7, 0.2])

o3d.visualization.draw_geometries([clipped_mesh, plane], window_name="Step 6: Clipped Mesh")
print("Original vertices:", len(vertices))
print("Remaining vertices after clipping:", len(clipped_vertices))

# -----------------------
# STEP 7: Coloring and Extremes
# -----------------------
print("\n=== Step 7: Coloring and Extremes ===")
points = np.asarray(clipped_pcd.points)
if len(points) > 0:
    # Градиент по Z
    z_values = points[:, 2]
    min_z, max_z = z_values.min(), z_values.max()
    normalized = (z_values - min_z) / (max_z - min_z)
    colors = np.stack([normalized, np.zeros_like(normalized), 1 - normalized], axis=1)
    clipped_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Экстремальные точки
    min_point = points[np.argmin(z_values)]
    max_point = points[np.argmax(z_values)]
    print(f"Min Z point: {min_point}")
    print(f"Max Z point: {max_point}")

    # Сферы для экстремумов
    sphere_min = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_min.paint_uniform_color([1, 0, 0])
    sphere_min.translate(min_point)

    sphere_max = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_max.paint_uniform_color([0, 0, 1])
    sphere_max.translate(max_point)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries([clipped_pcd, sphere_min, sphere_max, frame],
                                      window_name="Step 7: Gradient + Extremes")
else:
    print("No points remaining after clipping.")

print("\n✅ DONE: All 7 steps completed successfully!")
