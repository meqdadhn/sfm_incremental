#include "sfm/view_graph.h"

#include <algorithm>

namespace sfm
{
uint64_t ViewGraph::Key(ImageId a, ImageId b)
{
  if (a > b)
    std::swap(a, b);
  return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
}

void ViewGraph::Reset(int num_images)
{
  pairs_.clear();
  index_.clear();
  neighbors_.assign(num_images, {});
}

void ViewGraph::AddPair(TwoViewGeometry geometry)
{
  if (geometry.image1 > geometry.image2)
  {
    // Store with image1 < image2: invert the relative pose and swap the matches.
    std::swap(geometry.image1, geometry.image2);
    geometry.t = -geometry.R.transpose() * geometry.t;
    geometry.R.transposeInPlace();
    for (auto &m : geometry.inliers)
      std::swap(m.idx1, m.idx2);
  }
  index_[Key(geometry.image1, geometry.image2)] = static_cast<int>(pairs_.size());
  neighbors_[geometry.image1].push_back(geometry.image2);
  neighbors_[geometry.image2].push_back(geometry.image1);
  pairs_.push_back(std::move(geometry));
}

const TwoViewGeometry *ViewGraph::Pair(ImageId a, ImageId b) const
{
  const auto it = index_.find(Key(a, b));
  return it == index_.end() ? nullptr : &pairs_[it->second];
}

int ViewGraph::NumInliers(ImageId a, ImageId b) const
{
  const TwoViewGeometry *g = Pair(a, b);
  return g ? static_cast<int>(g->inliers.size()) : 0;
}

bool ViewGraph::RelativePose(ImageId from, ImageId to, Eigen::Matrix3d *R, Eigen::Vector3d *t) const
{
  const TwoViewGeometry *g = Pair(from, to);
  if (!g)
    return false;
  if (g->image1 == from)
  {
    *R = g->R;
    *t = g->t;
  }
  else
  {
    *R = g->R.transpose();
    *t = -g->R.transpose() * g->t;
  }
  return true;
}

std::vector<FeatureMatch> ViewGraph::Matches(ImageId a, ImageId b) const
{
  const TwoViewGeometry *g = Pair(a, b);
  if (!g)
    return {};
  std::vector<FeatureMatch> matches = g->inliers;
  if (g->image1 != a)
    for (auto &m : matches)
      std::swap(m.idx1, m.idx2);
  return matches;
}

} // namespace sfm
