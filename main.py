import open3d as o3d
import numpy as np
import copy
import os

def print_model_info(model, model_name, check_color=True, check_normals=True, check_triangles=True, check_intersections=True):
    """Utility function to print model information"""
    print(f"\n=== {model_name} Information ===")
    # vertices / points
    if hasattr(model, 'vertices'):
        print(f"Number of vertices: {len(model.vertices)}")
    elif hasattr(model, 'points'):
        print(f"Number of vertices: {len(model.points)}")
    # triangles
    if check_triangles and hasattr(model, 'triangles'):
        print(f"Number of triangles: {len(model.triangles)}")
    # color
    if check_color:
        has_color = False
        if hasattr(model, 'vertex_colors') and len(model.vertex_colors) > 0:
            has_color = True
        elif hasattr(model, 'colors') and len(model.colors) > 0:
            has_color = True
        print(f"Has color: {'Yes' if has_color else 'No'}")
    # normals
    if check_normals:
        has_normals = False
        if hasattr(model, 'vertex_normals') and len(model.vertex_normals) > 0:
            has_normals = True
        elif hasattr(model, 'normals') and len(model.normals) > 0:
            has_normals = True
        print(f"Has normals: {'Yes' if has_normals else 'No'}")
    # intersections / manifoldness (proxy)
    if check_intersections and hasattr(model, 'is_edge_manifold'):
        try:
            edge_manifold = model.is_edge_manifold()
            vertex_manifold = model.is_vertex_manifold()
            print(f"Edge manifold: {edge_manifold}, Vertex manifold: {vertex_manifold}")
        except Exception:
            # older open3d versions may not have these methods
            pass

def load_model_alternative(model_path):
    """Alternative method to load problematic OBJ files"""
    print("Trying alternative loading methods...")
    # Method 1: Try loading as point cloud first
    try:
        pcd = o3d.io.read_point_cloud(model_path)
        if len(pcd.points) > 0:
            print("Successfully loaded as point cloud using read_point_cloud()")
            return pcd, True
    except Exception as e:
        print(f"Method 1 failed: {e}")
    # Method 2: Try reading file manually and creating point cloud (OBJ vertex parsing)
    try:
        vertices = []
        with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if vertices:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
            print(f"Successfully loaded {len(vertices)} vertices from OBJ file (manual parse)")
            return pcd, True
    except Exception as e:
        print(f"Method 2 failed: {e}")
    return None, False

def point_cloud_to_mesh(pcd, depth=9, remove_density_quantile=0.01):
    """Convert point cloud to mesh using Poisson reconstruction"""
    print("Converting point cloud to mesh using Poisson reconstruction...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    # Remove low density vertices (artifacts)
    try:
        densities = np.asarray(densities)
        threshold = np.quantile(densities, remove_density_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    except Exception:
        pass
    mesh.compute_vertex_normals()
    return mesh

def main():
    print("Starting Assignment #5 - 3D Processing with Open3D")
    print("=" * 60)
    model_path = "Sting-Sword-lowpoly.obj"  # <- замените на ваш уникальный файл

    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found!")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            print(f"  - {file}")
        return

    # -----------------------
    # TASK 1: Loading & Visual
    # -----------------------
    print("\nTASK 1: Loading and Visualization")
    mesh = None
    try:
        mesh = o3d.io.read_triangle_mesh(model_path)
        if mesh is None or len(mesh.vertices) == 0:
            print("read_triangle_mesh returned empty mesh")
            mesh = None
        else:
            print("Loaded as triangle mesh using read_triangle_mesh()")
    except Exception as e:
        print(f"read_triangle_mesh failed: {e}")
        mesh = None

    # If triangle mesh failed, try reading as point cloud and reconstruct later
    pcd = None
    if mesh is None:
        pcd, success = load_model_alternative(model_path)
        if not success:
            print("Error: could not load model by any method.")
            return
        # we will reconstruct later; but show the point cloud now
        print_model_info(pcd, "Original Point Cloud (from file)", check_triangles=False)
        o3d.visualization.draw_geometries([pcd], window_name="Task1: Original Point Cloud", width=800, height=600)
    else:
        # Ensure normals exist
        if len(mesh.vertex_normals) == 0:
            mesh.compute_vertex_normals()
        print_model_info(mesh, "Original Mesh")
        o3d.visualization.draw_geometries([mesh], window_name="Task1: Original Mesh", width=800, height=600)

    # -----------------------
    # TASK 2: To Point Cloud
    # -----------------------
    print("\nTASK 2: Conversion to Point Cloud")
    # First, try to read point cloud directly (requirement)
    pcd_from_file = None
    try:
        pcd_from_file = o3d.io.read_point_cloud(model_path)
        if pcd_from_file is not None and len(pcd_from_file.points) > 0:
            print("read_point_cloud() returned a non-empty cloud (used per assignment requirement).")
            point_cloud = pcd_from_file
        else:
            print("read_point_cloud() returned empty - will sample from mesh as fallback.")
            pcd_from_file = None
    except Exception as e:
        print(f"read_point_cloud failed: {e}")
        pcd_from_file = None

    if pcd_from_file is None:
        # fallback: sample points from mesh
        if mesh is None:
            # we have pcd already from load_model_alternative
            point_cloud = pcd
        else:
            number_of_points = min(5000, max(1000, len(mesh.vertices) * 2))
            point_cloud = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    # ensure has color or set uniform
    if len(point_cloud.colors) == 0:
        point_cloud.paint_uniform_color([0.1, 0.5, 0.8])
    print_model_info(point_cloud, "Point Cloud (Task 2)", check_triangles=False, check_normals=False)
    o3d.visualization.draw_geometries([point_cloud], window_name="Task2: Point Cloud", width=800, height=600)

    # -----------------------
    # TASK 3: Surface Reconstruction
    # -----------------------
    print("\nTASK 3: Surface Reconstruction from Point Cloud")
    # Use Poisson reconstruction
    mesh_reconstructed = point_cloud_to_mesh(point_cloud, depth=8)
    # Crop using bounding box of point cloud to remove far artifacts
    bbox = point_cloud.get_axis_aligned_bounding_box()
    mesh_reconstructed = mesh_reconstructed.crop(bbox)
    mesh_reconstructed.compute_vertex_normals()
    mesh_reconstructed.paint_uniform_color([0.3, 0.7, 0.2])
    print_model_info(mesh_reconstructed, "Reconstructed Mesh (Task 3)")
    o3d.visualization.draw_geometries([mesh_reconstructed], window_name="Task3: Reconstructed Mesh", width=800, height=600)

    # -----------------------
    # TASK 4: Voxelization
    # -----------------------
    print("\nTASK 4: Voxelization")
    bbox = mesh_reconstructed.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    voxel_size = max(bbox_size) / 20.0
    # create voxel grid from point cloud (assignment asks from point cloud)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
    print(f"Voxel size: {voxel_size:.4f}")
    print(f"Number of voxels: {len(voxel_grid.get_voxels())}")
    # VoxelGrid doesn't have colors per voxel in O3D stable API; print presence of vertex colors from source
    has_color = len(point_cloud.colors) > 0
    print(f"Voxel color presence (in source point cloud): {'Yes' if has_color else 'No'}")
    o3d.visualization.draw_geometries([voxel_grid], window_name="Task4: Voxel Grid", width=800, height=600)

    # -----------------------
    # TASK 5: Adding a Plane
    # -----------------------
    print("\nTASK 5: Adding a Plane")
    bbox = mesh_reconstructed.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    plane_size = max(bbox_size) * 1.5
    plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=0.01, depth=plane_size)
    # position the plane next to object (slightly below)
    plane_center = bbox.get_center()
    plane.translate([plane_center[0] - plane_size/2.0, bbox.get_min_bound()[1] - 0.05, plane_center[2] - plane_size/2.0])
    plane.paint_uniform_color([0.8, 0.2, 0.2])
    print(f"Plane placed at Y = {bbox.get_min_bound()[1] - 0.05:.3f}, plane size ~ {plane_size:.3f}")
    o3d.visualization.draw_geometries([mesh_reconstructed, plane], window_name="Task5: Mesh with Plane", width=900, height=600)

    # -----------------------
    # TASK 6: Surface Clipping
    # -----------------------
    print("\nTASK 6: Surface Clipping")
    # We'll use a vertical plane x = plane_x (plane through center X of bounding box)
    center = point_cloud.get_axis_aligned_bounding_box().get_center()
    plane_x = center[0]  # vertical plane through center
    print(f"Clipping plane: keep points with X < {plane_x:.6f} (left side)")

    # Clip point cloud first
    points = np.asarray(point_cloud.points)
    mask_points = points[:, 0] < plane_x
    clipped_points = points[mask_points]
    clipped_pcd = o3d.geometry.PointCloud()
    clipped_pcd.points = o3d.utility.Vector3dVector(clipped_points)
    if len(point_cloud.colors) > 0:
        original_colors = np.asarray(point_cloud.colors)
        clipped_colors = original_colors[mask_points]
        clipped_pcd.colors = o3d.utility.Vector3dVector(clipped_colors)
    else:
        clipped_pcd.paint_uniform_color([0.1, 0.5, 0.8])

    # Now clip mesh (remove vertices on the right side)
    mesh_clipped = copy.deepcopy(mesh_reconstructed)
    verts = np.asarray(mesh_clipped.vertices)
    vertices_to_remove_mask = verts[:, 0] >= plane_x  # remove >= plane_x (right side)
    if vertices_to_remove_mask.shape[0] > 0:
        mesh_clipped.remove_vertices_by_mask(vertices_to_remove_mask)
    mesh_clipped.compute_vertex_normals()

    # Print results
    print(f"Original point count: {len(points)}")
    print(f"Remaining points after clipping (cloud): {len(clipped_points)}")
    # Triangles and normals from clipped mesh
    triangles_count_after = len(mesh_clipped.triangles) if hasattr(mesh_clipped, 'triangles') else 0
    print(f"Triangles after clipping (mesh): {triangles_count_after}")
    has_color_mesh_clipped = (len(mesh_clipped.vertex_colors) > 0) if hasattr(mesh_clipped, 'vertex_colors') else False
    print(f"Has color (clipped mesh): {'Yes' if has_color_mesh_clipped else 'No'}")
    has_normals_clipped = (len(mesh_clipped.vertex_normals) > 0) if hasattr(mesh_clipped, 'vertex_normals') else False
    print(f"Has normals (clipped mesh): {'Yes' if has_normals_clipped else 'No'}")
    # manifoldness as proxy to intersections
    try:
        print(f"Clipped mesh: edge_manifold={mesh_clipped.is_edge_manifold()}, vertex_manifold={mesh_clipped.is_vertex_manifold()}")
    except Exception:
        pass

    print_model_info(clipped_pcd, "Clipped Point Cloud", check_triangles=False, check_normals=False)
    print_model_info(mesh_clipped, "Clipped Mesh (Task 6)")
    o3d.visualization.draw_geometries([clipped_pcd, mesh_clipped], window_name="Task6: Clipped Model", width=900, height=600)

    # -----------------------
    # TASK 7: Color & Extremes
    # -----------------------
    print("\nTASK 7: Working with Color and Extremes")
    colored_pcd = copy.deepcopy(clipped_pcd) if len(clipped_pcd.points) > 0 else copy.deepcopy(point_cloud)
    pts = np.asarray(colored_pcd.points)
    if len(pts) == 0:
        print("No points available to color. Skipping Task 7.")
    else:
        # choose axis (Y axis by default)
        axis = 1  # 0->X,1->Y,2->Z
        coords = pts[:, axis]
        cmin, cmax = coords.min(), coords.max()
        if cmax > cmin:
            normalized = (coords - cmin) / (cmax - cmin)
        else:
            normalized = np.zeros_like(coords)
        colors = np.zeros((len(pts), 3))
        colors[:, 0] = normalized         # red channel
        colors[:, 2] = 1.0 - normalized   # blue channel
        colors[:, 1] = 0.2
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        # find extrema along chosen axis
        min_idx = int(np.argmin(pts[:, axis]))
        max_idx = int(np.argmax(pts[:, axis]))
        min_pt = pts[min_idx]
        max_pt = pts[max_idx]
        print(f"Extrema along axis {axis} -> min: {min_pt}, max: {max_pt}")

        # highlight with spheres
        dims = np.abs(np.max(pts, axis=0) - np.min(pts, axis=0))
        sphere_radius = max(dims) * 0.03 if max(dims) > 0 else 0.01
        s_min = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s_min.translate(min_pt)
        s_min.paint_uniform_color([1.0, 1.0, 0.0])
        s_max = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s_max.translate(max_pt)
        s_max.paint_uniform_color([1.0, 0.0, 1.0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=sphere_radius * 3)

        o3d.visualization.draw_geometries([colored_pcd, s_min, s_max, frame], window_name="Task7: Gradient and Extremes", width=900, height=600)

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print("1. Model loaded (triangle mesh or point cloud)")
    print("2. Converted/ensured point cloud using read_point_cloud() or sampling")
    print("3. Surface reconstructed (Poisson) and cropped")
    print("4. Voxel grid created")
    print("5. Plane added to scene")
    print("6. Clipping performed for point cloud and mesh")
    print("7. Gradient colors applied and extremes highlighted")
    print("Assignment steps completed. Prepare to run the script live for defense.")

if __name__ == "__main__":
    main()
