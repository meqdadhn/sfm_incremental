/*******************************************************************************
* @file    resection.h
* @brief   Position of a new image given its rotation (after Translation_Estimation_trimming).
*
* With R fixed, each 2D-3D correspondence (u, v) <-> X gives two equations
* linear in t (x_cam = R X + t):
*     t1 - u t3 = u (RX)_3 - (RX)_1
*     t2 - v t3 = v (RX)_3 - (RX)_2
* solved in least squares. If the RMS reprojection error is above a trigger,
* least trimmed squares repeatedly drops the worst residuals and re-solves.
* An optional motion-only refinement of (R, t) on the reprojection error follows.
*******************************************************************************/

#ifndef SFM_RESECTION_H_
#define SFM_RESECTION_H_

#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct ResectionParams
{
  int min_correspondences = 10;
  double trim_trigger_px = 10.0;   ///< start trimming when the RMS error exceeds this
  double trim_ratio = 0.95;        ///< fraction kept per trimming step
  double max_error_px = 4.0;       ///< final inlier threshold
  bool refine_pose = true;         ///< motion-only refinement of (R, t) on the inliers
  double refine_loss_px = 2.0;
};

struct ResectionResult
{
  Pose pose;
  std::vector<bool> inliers;
  int num_inliers = 0;
  double score_px = 0.0; ///< RMS reprojection error over inliers
};

/// Least-squares translation for a known rotation from normalized observations.
bool SolveTranslationKnownRotation(const Eigen::Matrix3d &R, const std::vector<Eigen::Vector3d> &points,
                                   const std::vector<Eigen::Vector2d> &normalized, const std::vector<bool> &mask, Eigen::Vector3d *t);

/// Full robust resection with a known rotation.
bool ResectKnownRotation(const Camera &camera, const Eigen::Matrix3d &R, const std::vector<Eigen::Vector3d> &points,
                         const std::vector<Eigen::Vector2d> &normalized, const std::vector<Eigen::Vector2d> &pixels,
                         const ResectionParams &params, ResectionResult *result);

/// Motion-only bundle adjustment of one pose with fixed points and intrinsics (pixel residuals).
void RefinePose(const Camera &camera, const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector2d> &pixels,
                double loss_px, Pose *pose);

} // namespace sfm

#endif // SFM_RESECTION_H_
