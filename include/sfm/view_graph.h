/*******************************************************************************
* @file    view_graph.h
* @brief   Verified image pairs (ROPs) and their adjacency.
*******************************************************************************/

#ifndef SFM_VIEW_GRAPH_H_
#define SFM_VIEW_GRAPH_H_

#include <unordered_map>
#include <vector>

#include "sfm/types.h"

namespace sfm
{
class ViewGraph
{
public:
  explicit ViewGraph(int num_images = 0) : neighbors_(num_images) {}

  void Reset(int num_images);
  void AddPair(TwoViewGeometry geometry);

  int NumImages() const { return static_cast<int>(neighbors_.size()); }
  const std::vector<TwoViewGeometry> &Pairs() const { return pairs_; }
  const std::vector<ImageId> &Neighbors(ImageId id) const { return neighbors_[id]; }
  const TwoViewGeometry *Pair(ImageId a, ImageId b) const;
  int NumInliers(ImageId a, ImageId b) const;

  /// Relative pose taking camera `from` to camera `to`: x_to = R * x_from + t, |t| = 1.
  bool RelativePose(ImageId from, ImageId to, Eigen::Matrix3d *R, Eigen::Vector3d *t) const;

  /// Inlier matches oriented as (feature in a, feature in b).
  std::vector<FeatureMatch> Matches(ImageId a, ImageId b) const;

private:
  static uint64_t Key(ImageId a, ImageId b);

  std::vector<TwoViewGeometry> pairs_;
  std::unordered_map<uint64_t, int> index_;
  std::vector<std::vector<ImageId>> neighbors_;
};

} // namespace sfm

#endif // SFM_VIEW_GRAPH_H_
