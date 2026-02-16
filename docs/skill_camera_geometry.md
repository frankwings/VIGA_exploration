# Camera & 3D Geometry Reference

Quick-reference for camera models, coordinate systems, projective geometry, and transform conventions. Based on Hartley & Zisserman "Multiple View Geometry in Computer Vision" (2nd Ed.) and practical experience with SAM3D/MoGe/Blender/PyTorch3D pipelines.

---

## 1. Pinhole Camera Model

A camera maps 3D world points to 2D image points via central projection.

### Basic Projection

A point `X = (X, Y, Z)` in camera coordinates projects to image point:

```
x = fX/Z,  y = fY/Z
```

where `f` is the focal length.

### Homogeneous Form (H&Z eq 6.2)

```
    [fX]     [f     px] [1 0 0 0] [X]
w * [fY]  =  [  f   py] [0 1 0 0] [Y]
    [ Z]     [      1 ] [0 0 1 0] [Z]
                                   [1]

    x  =  K  [I | 0]  X_cam
```

### Calibration Matrix K (H&Z eq 6.9, 6.10)

```
        [alpha_x    s    x0]
K   =   [        alpha_y y0]
        [                 1]
```

| Parameter | Meaning |
|---|---|
| `alpha_x = f * m_x` | Focal length in x-pixels (m_x = pixels/unit) |
| `alpha_y = f * m_y` | Focal length in y-pixels |
| `s` | Skew (usually 0 for CCD cameras) |
| `(x0, y0)` | Principal point in pixel coordinates |

For square pixels: `alpha_x = alpha_y = f` (when measured in pixels).

### Full Camera Matrix (H&Z eq 6.7, 6.8)

```
x = K [R | t] X_world          (eq 6.8)
x = K R [I | -C] X_world       (eq 6.7)
```

where:
- `R` = 3x3 rotation matrix (camera orientation)
- `C` = camera center in world coordinates
- `t = -RC` (translation vector)
- `P = K[R|t]` is the 3x4 camera projection matrix (11 DOF)

### Decomposing P (H&Z sec 6.2.4)

Given `P = [M | p4]`:
1. Camera center: `C` = right null-vector of P (i.e. `PC = 0`)
2. `M = KR` via RQ decomposition (upper-triangular x orthogonal)
3. K has positive diagonal entries, R is rotation

---

## 2. Coordinate System Conventions

### The Big Table

| System | X | Y | Z | Handedness | Used By |
|---|---|---|---|---|---|
| OpenCV | Right | **Down** | **Forward** | Right-handed | MoGe, most CV |
| OpenGL | Right | Up | **Backward** | Right-handed | Most renderers |
| PyTorch3D | **Left** | Up | Forward | Right-handed | TRELLIS/SAM3D |
| Blender World | Right | **Forward** | Up | Right-handed | Blender scenes |
| Blender Camera | Right | Up | **Backward** (looks -Z) | Right-handed | Blender camera |
| glTF | Right | Up | **Backward** | Right-handed | GLB files |
| USD | Right | Up | **Backward** | Right-handed | Omniverse |
| COLMAP | Right | **Down** | **Forward** | Right-handed | SfM |
| Unity | Right | Up | **Forward** | **Left-handed** | Unity engine |
| Unreal | **Forward** | Right | Up | **Left-handed** | UE5 |

### Critical Differences

**OpenCV vs OpenGL/glTF:** Y and Z are both flipped.
```
OpenCV:  X-right, Y-down,  Z-forward
OpenGL:  X-right, Y-up,    Z-backward

Convert: negate Y and Z  (or rotate 180 around X)
```

**PyTorch3D vs OpenCV:** X is flipped.
```
PyTorch3D: X-left,  Y-up,    Z-forward
OpenCV:    X-right, Y-down,  Z-forward

Convert: negate X and Y  (or rotate 180 around Z)
```

**Blender World vs glTF/OpenGL:** Y and Z are swapped.
```
Blender: X-right, Y-forward, Z-up
glTF:    X-right, Y-up,      Z-backward

Convert: Blender auto-swaps on GLB import/export
```

---

## 3. Homogeneous Coordinates

### Points

A 2D point `(x, y)` in homogeneous coordinates is `(x, y, 1)` or equivalently `(kx, ky, k)` for any `k != 0`.

A 3D point `(X, Y, Z)` becomes `(X, Y, Z, 1)` or `(kX, kY, kZ, k)`.

Points at infinity: last coordinate is 0. E.g., `(x, y, 0)` in 2D.

### Lines (2D)

A line `ax + by + c = 0` is represented by vector `l = (a, b, c)`.
- Point `x` lies on line `l` iff `l . x = 0`
- Line through two points: `l = x1 x x2` (cross product)
- Intersection of two lines: `x = l1 x l2`

### Planes (3D)

A plane `aX + bY + cZ + d = 0` is `pi = (a, b, c, d)`.
- Point `X` lies on plane `pi` iff `pi . X = 0`

---

## 4. Transform Hierarchy (H&Z sec 2.4)

From most restrictive to most general:

| Transform | DOF (2D) | DOF (3D) | Preserves | Matrix Form |
|---|---|---|---|---|
| **Euclidean** (rigid) | 3 | 6 | Distances, angles | `[R | t]` where R is rotation |
| **Similarity** | 4 | 7 | Ratios of distances, angles | `[sR | t]` |
| **Affine** | 6 | 12 | Parallelism, area ratios | `[A | t]` where A is invertible |
| **Projective** | 8 | 15 | Cross-ratio, collinearity | General `H` (invertible) |

### Euclidean (Rigid Body)

```
X' = R*X + t
```
- R is orthogonal: `R^T R = I`, `det(R) = +1`
- Preserves distances and angles
- 6 DOF in 3D: 3 rotation + 3 translation

### Similarity

```
X' = s*R*X + t
```
- Adds uniform scale `s`
- 7 DOF in 3D

### Affine

```
X' = A*X + t
```
- A is any invertible 3x3 matrix
- Preserves parallelism (parallel lines stay parallel)
- Does NOT preserve angles or distances

### Projective (Homography)

```
X' = H * X   (homogeneous, 4x4 for 3D)
```
- Most general linear transformation of projective space
- Only preserves collinearity (straight lines stay straight) and cross-ratio
- 15 DOF in 3D (16 entries minus 1 for scale)

---

## 5. Rotation Representations

### Rotation Matrix (3x3)

```
R^T R = I,  det(R) = +1
```
- 3 DOF (9 entries, 6 constraints)
- Columns are orthonormal basis vectors
- Composition: `R_combined = R2 @ R1` (R1 applied first)

### Euler Angles

Three successive rotations around coordinate axes. **Order matters!**

Common conventions:
- **XYZ** (roll-pitch-yaw): `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`
- **ZYX**: `R = Rx @ Ry @ Rz`

**Warning:** Gimbal lock at pitch = +/-90 degrees.

### Quaternion

`q = (x, y, z, w)` or `q = (w, x, y, z)` depending on convention.

| Library | Order | Unit |
|---|---|---|
| PyTorch3D | **(x, y, z, w)** | Unit quaternion |
| Blender (mathutils) | **(w, x, y, z)** | Unit quaternion |
| scipy.spatial | **(x, y, z, w)** | Unit quaternion |
| glTF | **(x, y, z, w)** | Unit quaternion |
| Unity | **(x, y, z, w)** | Unit quaternion |
| Eigen (C++) | **(w, x, y, z)** | Unit quaternion |
| numpy-quaternion | **(w, x, y, z)** | Unit quaternion |

**Always check xyzw vs wxyz when crossing library boundaries!**

Quaternion to rotation matrix:
```
R = I + 2w*[v]_x + 2*[v]_x^2
where v = (x,y,z), [v]_x is the skew-symmetric matrix of v
```

### Axis-Angle

Rotation by angle `theta` around unit axis `k`:
```
R = I + sin(theta)*[k]_x + (1-cos(theta))*[k]_x^2   (Rodrigues)
```

---

## 6. 4x4 Homogeneous Transform Matrices

### Row-Vector vs Column-Vector Convention

**This is the #1 source of transform bugs.**

**Column-vector convention** (standard math, OpenGL, Blender, most textbooks):
```
x' = M * x

M = [R  t]    Translation is in the LAST COLUMN (column 3)
    [0  1]

    [r00 r01 r02 tx]
M = [r10 r11 r12 ty]
    [r20 r21 r22 tz]
    [ 0   0   0   1]
```

**Row-vector convention** (PyTorch3D, DirectX):
```
x'^T = x^T * M

M = [R^T  0]    Translation is in the LAST ROW (row 3)
    [t^T  1]

    [r00 r10 r20  0]
M = [r01 r11 r21  0]
    [r02 r12 r22  0]
    [tx  ty  tz   1]
```

**PyTorch3D uses row-vector convention:** `points_transformed = points @ M`

This means translation goes in `M[3,0], M[3,1], M[3,2]` (row 3), NOT `M[0,3], M[1,3], M[2,3]` (column 3).

### Composition Order

Column-vector: `M_combined = M2 @ M1` (M1 applied first, right-to-left)
Row-vector: `M_combined = M1 @ M2` (M1 applied first, left-to-right)

---

## 7. Common Conversion Recipes

### OpenCV Camera to Blender Camera

```python
# OpenCV: camera at origin, looking +Z, Y-down
# Blender: camera looks -Z (local), Y-up

# 1. Set camera location and rotation from R, t
# Blender camera looks along -Z_local, so:
cam_rotation = R @ Rx(pi)  # flip Y and Z axes
cam_location = -R^T @ t

# 2. After rendering, flip image vertically (Y-down vs Y-up)
```

### OpenCV to glTF

```python
# OpenCV: X-right, Y-down, Z-forward
# glTF:   X-right, Y-up,   Z-backward

# Negate Y and Z:
point_gltf = [x, -y, -z]

# Equivalently, apply rotation of 180 degrees around X:
R_cv_to_gltf = [[1, 0, 0],
                 [0,-1, 0],
                 [0, 0,-1]]
```

### PyTorch3D to OpenCV

```python
# PyTorch3D: X-left, Y-up, Z-forward
# OpenCV:    X-right, Y-down, Z-forward

# Negate X and Y:
point_opencv = [-x, -y, z]

# Equivalently, rotate 180 around Z:
R_p3d_to_cv = [[-1, 0, 0],
                [0, -1, 0],
                [0,  0, 1]]
```

### PyTorch3D to glTF

```python
# PyTorch3D: X-left,  Y-up,   Z-forward
# glTF:      X-right, Y-up,   Z-backward

# Negate X and Z:
point_gltf = [-x, y, -z]
```

### MoGe Pointmap to Blender World

```python
# MoGe outputs in OpenCV convention (X-right, Y-down, Z-forward)
# Blender world: X-right, Y-forward, Z-up

# Swap Y and Z, negate new Z:
x_blender = x_moge
y_blender = z_moge      # depth becomes forward
z_blender = -y_moge     # down becomes up (negated)
```

---

## 8. Intrinsics: Focal Length vs Sensor Size

### Physical Camera

```
f_pixels = f_mm * (image_width_pixels / sensor_width_mm)
```

### Blender Mapping

Blender uses physical camera model:
```python
cam.sensor_width = sensor_mm          # default 36mm
cam.lens = focal_length_mm
cam.shift_x = (cx - w/2) / w         # principal point offset
cam.shift_y = -(cy - h/2) / h        # note: negated for Y
```

To use pixel-based intrinsics (from MoGe/OpenCV):
```python
cam.sensor_fit = 'HORIZONTAL'
cam.sensor_width = image_width_px     # treat pixels as mm
cam.lens = fx_px                      # focal length in pixels
cam.shift_x = (cx - w/2) / w
cam.shift_y = -(cy - h/2) / h
```

### OpenCV Intrinsics Matrix

```
    [fx  0  cx]
K = [ 0  fy cy]
    [ 0   0  1]
```

- `fx, fy`: focal length in pixels
- `cx, cy`: principal point in pixels
- For square pixels: `fx = fy`
- Skew is usually 0 (omitted)

---

## 9. Depth & Projection

### Perspective Projection

```
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

### Inverse (Back-projection to Ray)

Given pixel `(u, v)` and depth `Z`:
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

### MoGe Output Format

MoGe estimates monocular geometry from a single image:
```
depth:          (H, W)     float32   metric depth in meters
points:         (H, W, 3)  float32   3D pointmap in OpenCV convention
intrinsics_px:  (3, 3)     float32   K matrix in pixel units
intrinsics_norm: (3, 3)    float32   K normalized to [0,1] image coords
image_width:    int
image_height:   int
```

The `points` array satisfies: `points[v, u] = K^{-1} @ [u, v, 1] * depth[v, u]`

---

## 10. Lessons Learned (SAM3D Pipeline)

### Bug Pattern: Row vs Column Translation

PyTorch3D uses **row-vector convention**. Translation goes in **row 3**, not column 3:
```python
# WRONG (column-vector convention):
T[0, 3] = tx;  T[1, 3] = ty;  T[2, 3] = tz

# CORRECT (PyTorch3D row-vector convention):
T[3, 0] = tx;  T[3, 1] = ty;  T[3, 2] = tz
```

### Bug Pattern: Sign Flip in Coordinate Conversion

When converting between coordinate systems, get the signs right. A single negated axis mirrors the entire mesh:
```python
# WRONG: negated X when it shouldn't be
R_zup_to_yup = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]

# CORRECT:
R_zup_to_yup = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
```

### Bug Pattern: Redundant Transforms That Cancel

Multiple "correction" rotations that multiply to identity add complexity but no value. Simplify first, then verify.

### Render Axis Correction

When rendering OpenCV-convention geometry in Blender:
1. Blender camera at origin with `Rx(-90)` rotation (looks along -Y_blender = +Z_opencv)
2. Post-render flip: vertical (Y-down vs Y-up) + horizontal (PyTorch3D X-left vs OpenCV X-right)

---

## References

- Hartley, R. & Zisserman, A. "Multiple View Geometry in Computer Vision" (2nd Ed.), Cambridge University Press, 2004.
  - Chapter 2: Projective Geometry 2D (transformations, homogeneous coords)
  - Chapter 3: Projective Geometry 3D
  - Chapter 6: Camera Models (pinhole, K matrix, P = K[R|t])
  - Chapter 7: Computing Camera Matrix P (DLT, Gold Standard)
  - Chapter 8: Single View Geometry (vanishing points, back-projection)
- PyTorch3D documentation: coordinate conventions, row-vector convention
- Blender documentation: camera model, sensor_width/lens mapping
- glTF 2.0 specification: coordinate system, quaternion order
