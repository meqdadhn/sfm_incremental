/*******************************************************************************
* @file    bundle_adjustment.h
* @brief   Sparse bundle adjustment with Ceres (Schur complement on the points).
*
* Used for the window BA during incremental registration and for the final BA.
* Poses are parameterized as angle-axis + camera center. Gauge: if fewer than
* two fixed cameras are in the problem, the first gauge image is held fixed and
* the dominant baseline coordinate of the second is frozen to fix the scale.
*******************************************************************************/

#ifndef SFM_BUNDLE_ADJUSTMENT_H_
#define SFM_BUNDLE_ADJUSTMENT_H_

#include <string>
#include <vector>

#include "sfm/reconstruction.h"

namespace sfm
{
struct BundleAdjustmentParams
{
  int max_iterations = 50;
  std::string loss = "huber";      ///< "huber", "cauchy" or "none"
  double loss_scale_px = 2.0;
  bool refine_intrinsics = false;  ///< focal + distortion, shared per camera
  bool refine_principal_point = false;
  int num_threads = 0;             ///< 0 = hardware concurrency
  double function_tolerance = 1e-6;
  bool verbose = false;
};

struct BundleAdjustmentSetup
{
  std::vector<ImageId> variable_images; ///< poses to optimize; other registered images seeing the same points are held fixed
  ImageId gauge_image1 = kInvalidId;    ///< held fixed if the problem has no fixed cameras
  ImageId gauge_image2 = kInvalidId;    ///< one center coordinate frozen to fix the scale
};

struct BundleAdjustmentSummary
{
  bool success = false;
  int num_images = 0;
  int num_points = 0;
  int num_residuals = 0;
  int iterations = 0;
  double initial_rms_px = 0.0;
  double final_rms_px = 0.0;
  double time_s = 0.0;

  std::string Brief() const;
};

BundleAdjustmentSummary RunBundleAdjustment(const BundleAdjustmentParams &params, const BundleAdjustmentSetup &setup,
                                            Reconstruction *reconstruction);

} // namespace sfm

#endif // SFM_BUNDLE_ADJUSTMENT_H_
