# sfm

Incremental structure-from-motion, a clean reimplementation of the original pipeline
(`SfM.cpp`, `ROP.cpp`, `F_Matching.cpp` in `~/Dev/my_repos/SfM`) with the same incremental math.

| Stage | Module | Original |
|---|---|---|
| 1. Read images + cameras (YAML) | `config`, `pipeline` | `Load_img`, `.prj` |
| 2. SIFT | `features` | `SIFT_operator` |
| 3. FLANN matching + ROP (5-pt E in RANSAC, decomposition, Sampson refinement) | `matching`, `two_view` | `Flann_matching`, `Initial_estimate_hybrid` |
| 4. Incremental extrinsics + window BA | `incremental_mapper`, `rotation_averaging`, `resection`, `bundle_adjustment` | `SfM_incremental_pba`, `Single_Rotation_Averaging_RANSAC`, `Translation_Estimation_trimming`, `Edge_link_trimming`, `pba_driver` |
| 5. Feature tracking + final BA | `tracks`, `bundle_adjustment` | `Tracking_all`, `Remove_outliers`, `pba_driver` |

## Incremental SfM

1. **Seed**: the image with the most verified pairs, paired with its strongest neighbour
   (median triangulation angle ≥ 3°, homography/essential inlier ratio ≤ 0.8). The first
   image is the origin, the second sits at the ROP, and the baseline is 1.
2. **Registration**: for each unregistered image connected to the model:
   - **≥ 2 registered neighbours**: each ROP gives a rotation candidate `R_k = R_ki · R_i`.
     *Single rotation averaging with RANSAC* applies each candidate to the unit axes and solves
     the rotation in closed form (Horn's quaternion method, top eigenvector of a 4×4 matrix).
     It samples 2 candidates per hypothesis, keeps the largest consensus and refits on the
     inliers. Then the **position with the rotation fixed** comes from linear least squares
     on the 2D–3D correspondences through the inlier neighbours, using least trimmed squares
     while the RMS is above 10 px. The image is accepted if the RMS is below 5 px.
   - **Fallback, single neighbour**: the rotation comes from that one ROP, the position from
     the same solver, and the acceptance threshold is 20 px.
   - Images that fail are skipped until a newly registered image overlaps them.
3. New points are triangulated against every registered neighbour.
4. **Window BA** (Ceres, Schur): the last `window` images are optimized every `interval`
   registrations, and right after a poor registration. Older cameras that see the same points
   stay fixed. A full BA runs every `global_interval` registrations.

Conventions: `x_cam = R·X + t` (world → camera), OpenCV camera frame, OpenCV pinhole model
with `(k1, k2, p1, p2, k3)`. The ROP of a pair `(i, j)` is `x_j = R_ji·x_i + t_ji`, with `|t| = 1`.

### Differences from the original

- Translation estimation uses the maintained map points, which are triangulated from
  registered images and refined by window BA, instead of re-triangulating pairwise matches
  for every candidate. The linear model and the trimming are the same.
- A motion-only pose refinement follows the linear solve (`resection.refine_pose`; set it to
  `false` to match the old behaviour).
- The rotation RANSAC tries every candidate pair when there are few candidates, and its
  inlier test is an angle in degrees. The old `srand(time(0))` inside the loop, which
  repeated the same sample, is gone.
- The single-connection fallback no longer registers an image whose score is above its
  threshold.
- The ROP makes no nadir or 2-point assumption: it is a 5-point essential matrix plus
  nonlinear refinement.

## Build

Dependencies: OpenCV (core, imgproc, imgcodecs, features2d, flann, calib3d), Eigen3, Ceres, yaml-cpp, glog, and GTest (optional, for tests).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./sfm_tests
```

Configuring also writes `compile_commands.json` and symlinks it to the repo root, so clangd
(LunarVim, VS Code, ...) resolves includes and go-to-definition without extra setup.

SIFT backend: `cv::SIFT` on OpenCV ≥ 4.4, otherwise `xfeatures2d`, otherwise the bundled
implementation (`src/features/sift_impl.cpp`). CMake prints which one it chose.

## Run

```bash
./sfm_main ../config/example.yaml [-v 1]
```

Only `io` and `cameras` are required in the config; see `config/example.yaml` for every
parameter and its default. Features and ROPs are cached in `<output_dir>/cache`, and the
cache is invalidated automatically when the parameters that produced it change.

Outputs in `output_dir`:
- `colmap/`: COLMAP text model (open it with `colmap gui` → File → Import model)
- `points.ply`: coloured points plus camera centers (red)
- `poses.txt`: `name qw qx qy qz tx ty tz cx cy cz`

## Synthetic end-to-end check

```bash
python3 tools/render_synthetic.py /tmp/render --views 20   # textured box world, known poses, lens distortion
./build/sfm_main /tmp/render/config.yaml
python3 tools/evaluate_poses.py /tmp/render/gt_poses.txt /tmp/render/out/poses.txt
```

Reference result: 20/20 registered, mean rotation error 0.012°, mean position error 1.2 mm
on an 8 m scene, final reprojection error 0.29 px, about 13 s on 20 threads.

## Layout

```
include/sfm/    public headers, one per stage
src/            implementation (cost_functions.h holds the Ceres residuals)
apps/           sfm_main
tests/          gtest: geometry units + synthetic end-to-end incremental SfM
tools/          renderer and pose evaluation scripts
config/         example.yaml
```
