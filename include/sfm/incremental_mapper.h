/*******************************************************************************
* @file    incremental_mapper.h
* @brief   Incremental extrinsic estimation (after SfM_incremental_pba).
*
*  1. Seed: the image with the most verified connections, paired with its
*     strongest neighbour (subject to baseline / non-planarity gates). The seed
*     defines the coordinate system: first image at the origin, second at the ROP,
*     baseline length 1.
*  2. Registration loop over unregistered images connected to the model:
*     - >= 2 registered neighbours: single rotation averaging with RANSAC over
*       the candidates R_ki * R_i, then position from the 2D-3D correspondences
*       with the rotation fixed (linear LS + least trimmed squares).
*       Accepted if the RMS reprojection error is below accept_score_px.
*     - fallback, one neighbour: rotation from that ROP alone, same position
*       estimation, looser acceptance (accept_score_single_px).
*     Images that fail are skipped until a newly registered image connects to them.
*  3. New points are triangulated against every registered neighbour.
*  4. Window bundle adjustment over the most recent images every ba_interval
*     registrations, and immediately after a poor registration.
*******************************************************************************/

#ifndef SFM_INCREMENTAL_MAPPER_H_
#define SFM_INCREMENTAL_MAPPER_H_

#include <set>
#include <vector>

#include "sfm/bundle_adjustment.h"
#include "sfm/reconstruction.h"
#include "sfm/resection.h"
#include "sfm/rotation_averaging.h"
#include "sfm/triangulation.h"
#include "sfm/view_graph.h"

namespace sfm
{
struct IncrementalParams
{
  // Seed pair
  double seed_min_triangulation_angle_deg = 3.0;
  double seed_max_homography_ratio = 0.8;
  int seed_min_points = 100;
  int seed_max_trials = 20;

  // Registration
  RotationAveragingParams rotation;
  ResectionParams resection;
  double accept_score_px = 5.0;         ///< >= 2 connections
  double accept_score_single_px = 20.0; ///< single-connection fallback

  TriangulationParams triangulation;

  // Window bundle adjustment
  int ba_window = 20;          ///< number of most recent images optimized
  int ba_interval = 5;         ///< run window BA every N registrations
  int global_ba_interval = 50; ///< BA over all registered images every N registrations (0 = off)
  BundleAdjustmentParams ba;
};

class IncrementalMapper
{
public:
  IncrementalMapper(const IncrementalParams &params, const ViewGraph &view_graph, Reconstruction *reconstruction);

  /// Runs seed + registration loop. Returns false if no seed pair could be initialized.
  bool Run();

  ImageId SeedImage1() const { return seed1_; }
  ImageId SeedImage2() const { return seed2_; }
  const std::vector<ImageId> &RegistrationOrder() const { return order_; }

private:
  struct Candidate
  {
    ImageId id;
    std::vector<ImageId> registered_neighbors;
  };

  struct Correspondences
  {
    std::vector<int> features;
    std::vector<PointId> points;
  };

  bool Initialize();
  bool InitializeFromPair(ImageId a, ImageId b);
  bool RegisterNextImage();
  bool TryRegisterMulti(const Candidate &cand, Pose *pose, double *score, Correspondences *corr, std::vector<bool> *inliers);
  bool TryRegisterSingle(ImageId id, ImageId ref, Pose *pose, double *score, Correspondences *corr, std::vector<bool> *inliers);
  bool Resect(ImageId id, const Eigen::Matrix3d &R, const Correspondences &corr, Pose *pose, double *score, std::vector<bool> *inliers);
  Correspondences Collect2D3D(ImageId id, const std::vector<ImageId> &refs) const;
  void AddImage(ImageId id, const Pose &pose, const Correspondences &corr, const std::vector<bool> &inliers);
  int TriangulateImage(ImageId id);
  void WindowBundleAdjustment();
  void GlobalBundleAdjustment();
  std::vector<Candidate> Candidates() const;

  const IncrementalParams params_;
  const ViewGraph &view_graph_;
  Reconstruction *rec_;

  ImageId seed1_ = kInvalidId;
  ImageId seed2_ = kInvalidId;
  std::vector<ImageId> order_;
  std::set<ImageId> failed_;
  int since_window_ba_ = 0;
  int since_global_ba_ = 0;
};

} // namespace sfm

#endif // SFM_INCREMENTAL_MAPPER_H_
