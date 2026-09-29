/*******************************************************************************
* @file    rotation_averaging.h
* @brief   Single rotation averaging with RANSAC (after Single_Rotation_Averaging_RANSAC).
*
* An unregistered image k connected to registered images i gets one rotation
* candidate per connection, R_k^(i) = R_ki * R_i. Each candidate is applied to
* the unit axes n_j, giving correspondences n_j -> R_k^(i) n_j, and the rotation
* relating all of them is found in closed form with Horn's quaternion method
* (the eigenvector of the 4x4 matrix N with the largest eigenvalue).
* RANSAC draws two candidates per hypothesis, keeps the largest consensus set
* and refits on all inliers.
*******************************************************************************/

#ifndef SFM_ROTATION_AVERAGING_H_
#define SFM_ROTATION_AVERAGING_H_

#include <vector>

#include <Eigen/Core>

namespace sfm
{
struct RotationAveragingParams
{
  double inlier_threshold_deg = 2.0; ///< max angle between a candidate and the average
  int max_iterations = 100;
  double confidence = 0.99;
  int min_inliers = 2;
};

struct RotationAveragingResult
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  std::vector<bool> inliers;
  int num_inliers = 0;
  double score = 0.0; ///< RMS of (R * n_j - R_c * n_j) over inlier candidates and axes, as in the original code
};

/// Horn's closed-form average of the candidates flagged in `mask` (all if empty).
Eigen::Matrix3d AverageRotationsQuaternion(const std::vector<Eigen::Matrix3d> &candidates, const std::vector<bool> &mask = {});

/// RANSAC single rotation averaging. Needs >= 2 candidates; returns false without min_inliers consensus.
bool SingleRotationAveragingRansac(const std::vector<Eigen::Matrix3d> &candidates, const RotationAveragingParams &params,
                                   RotationAveragingResult *result);

} // namespace sfm

#endif // SFM_ROTATION_AVERAGING_H_
