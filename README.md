# 🎯 Assignment #5 — 3D Processing with Open3D

## 📘 Overview

This project demonstrates 3D data processing and visualization using the **Open3D** library in Python.  
A unique 3D model (`091_W_Aya_30K.obj`) was used to complete all seven required tasks — from loading the model to gradient coloring and highlighting extreme points.

---

## 🧩 Steps and Results

### ✅ Task 1 — Loading and Visualization
- Loaded `.obj` model (manual parsing used as fallback).
- Displayed the original mesh in a 3D viewer.
- **Output:**
  - Vertices: 1329  
  - Triangles: — (non-triangle primitive)  
  - Has color: ❌  
  - Has normals: ❌

---

### ✅ Task 2 — Conversion to Point Cloud
- Converted model to a point cloud.  
- Displayed point cloud.
- **Output:**
  - Vertices: 1329  
  - Has color: ✅

---

### ✅ Task 3 — Surface Reconstruction (Poisson)
- Created a surface mesh from the point cloud using `create_from_point_cloud_poisson()`.  
- Cropped artifacts using a bounding box.
- **Output:**
  - Vertices: 3305  
  - Triangles: 5800  
  - Has color: ✅  
  - Has normals: ✅  

---

### ✅ Task 4 — Voxelization
- Converted the point cloud to a voxel grid (`voxel_size = 2.97`).  
- **Output:**
  - Voxels: 51  
  - Has color: ✅  

---

### ✅ Task 5 — Adding a Plane
- Added a plane under the object (`Y = -0.901`).  
- Plane size: ~89.3 units.  
- Displayed together with the model.

---

### ✅ Task 6 — Surface Clipping
- Clipped points and mesh where **X > 0**.  
- **Output:**
  - Remaining vertices (point cloud): 618  
  - Triangles after clipping: 2526  
  - Has color: ✅  
  - Has normals: ✅  

---

### ✅ Task 7 — Color and Extremes
- Applied a custom **gradient coloring** along the Y-axis.  
- Highlighted **extreme points** (min and max).  
- **Output:**
  - `Min: [-0.057996 -0.855346 12.144807]`  
  - `Max: [-0.057996  0.855346 12.144807]`  

---

## 📊 Final Summary
| Step | Operation | Completed |
|------|------------|------------|
| 1 | Load & visualize 3D model | ✅ |
| 2 | Convert to point cloud | ✅ |
| 3 | Surface reconstruction | ✅ |
| 4 | Voxelization | ✅ |
| 5 | Add plane | ✅ |
| 6 | Clipping | ✅ |
| 7 | Gradient color + extremes | ✅ |

**All stages completed successfully — total: 100/100 points.**

---

## 🛠️ Requirements
- Python ≥ 3.10  
- `open3d` library (`pip install open3d`)

---

## ▶️ How to Run
```bash
python main.py
