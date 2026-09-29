/*******************************************************************************
* @file    two_view.h
* @brief   Relative orientation (ROP) of an image pair.
*
* 5-point essential matrix in RANSAC, decomposition with cheirality check,
* then nonlinear refinement of (R, t) on the Sampson error with t on the unit sphere.
* No assumption on the viewing geometry (nadir, forward, ...).
*******************************************************************************/

#ifndef SFM_TWO_VIEW_H_
#define SFM_TWO_VIEW_H_

#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct TwoViewParams
{
  double ransac_threshold_px = 1.5; ///< epipolar inlier threshold, converted to normalized units by the mean focal
  double confidence = 0.999;
  int min_inliers = 30;
  bool refine = true;               ///< Sampson-error refinement after RANSAC
  double refine_loss_px = 1.0;      ///< Cauchy loss scale for the refinement
  double homography_threshold_px = 3.0;
};

/// Estimates the ROP between two images from their normalized coordinates and putative matches.
/// Returns false if the pair is not geometrically verified.
bool EstimateTwoViewGeometry(const Camera &camera1, const Camera &camera2, const std::vector<Eigen::Vector2d> &normalized1,
                             const std::vector<Eigen::Vector2d> &normalized2, const std::vector<FeatureMatch> &matches,
                             const TwoViewParams &params, TwoViewGeometry *geometry);

/// Sampson distance of a normalized correspondence under E, in normalized units.
double SampsonError(const Eigen::Matrix3d &E, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2);

} // namespace sfm

#endif // SFM_TWO_VIEW_H_
