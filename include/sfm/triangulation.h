/*******************************************************************************
* @file    triangulation.h
* @brief   Linear multi-view intersection and the checks used to accept points.
*******************************************************************************/

#ifndef SFM_TRIANGULATION_H_
#define SFM_TRIANGULATION_H_

#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct TriangulationParams
{
  double max_reprojection_error_px = 4.0;
  double min_triangulation_angle_deg = 1.5;
};

/// Linear (DLT) intersection from normalized observations. Needs >= 2 views.
bool TriangulateDLT(const std::vector<Pose> &poses, const std::vector<Eigen::Vector2d> &normalized, Eigen::Vector3d *X);

/// Largest pairwise ray angle at X, in degrees.
double MaxTriangulationAngleDeg(const std::vector<Eigen::Vector3d> &centers, const Eigen::Vector3d &X);

/// Reprojection error in pixels, measured in the undistorted image (normalized error * mean focal).
/// Returns a huge value if X is behind the camera.
double ReprojectionErrorPx(const Camera &camera, const Pose &pose, const Eigen::Vector3d &X, const Eigen::Vector2d &normalized);

/// Triangulates a track, dropping the worst observation until all pass the checks.
/// `keep` (output) flags the observations that support the returned point.
bool TriangulateRobust(const std::vector<const Camera *> &cameras, const std::vector<Pose> &poses,
                       const std::vector<Eigen::Vector2d> &normalized, const TriangulationParams &params, Eigen::Vector3d *X,
                       std::vector<bool> *keep);

} // namespace sfm

#endif // SFM_TRIANGULATION_H_
