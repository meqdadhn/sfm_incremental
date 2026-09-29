#include "sfm/matching.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <opencv2/flann.hpp>

namespace sfm
{
namespace
{
/// For each row of `query`, the index of its nearest neighbour in the index if it passes the ratio test, else -1.
std::vector<int> RatioTestNN(cv::flann::Index &index, const cv::Mat &query, const MatchingParams &params)
{
  std::vector<int> best(query.rows, -1);
  if (query.rows == 0)
    return best;
  cv::Mat indices, dists;
  index.knnSearch(query, indices, dists, 2, cv::flann::SearchParams(params.flann_checks));
  const float ratio2 = static_cast<float>(params.ratio * params.ratio); // FLANN returns squared L2
  for (int i = 0; i < query.rows; ++i)
  {
    if (dists.at<float>(i, 0) < ratio2 * dists.at<float>(i, 1))
      best[i] = indices.at<int>(i, 0);
  }
  return best;
}
} // namespace

std::vector<std::pair<ImageId, ImageId>> SelectPairs(const std::vector<Image> &images, const MatchingParams &params)
{
  const int num_images = static_cast<int>(images.size());
  std::vector<std::pair<ImageId, ImageId>> pairs;
  if (params.mode == "trajectory")
  {
    // Symmetric neighbourhood graph on the position priors (knn or radius), as in the original option 2.
    std::set<std::pair<ImageId, ImageId>> selected;
    for (int i = 0; i < num_images; ++i)
    {
      if (!images[i].has_prior)
      {
        for (int j = 0; j < num_images; ++j)
          if (j != i)
            selected.emplace(std::min(i, j), std::max(i, j));
        continue;
      }
      std::vector<std::pair<double, int>> dist;
      for (int j = 0; j < num_images; ++j)
        if (j != i && images[j].has_prior)
          dist.emplace_back((images[j].prior_position - images[i].prior_position).norm(), j);
      std::sort(dist.begin(), dist.end());
      size_t keep = 0;
      if (params.search == "knn")
        keep = std::min<size_t>(dist.size(), params.knn);
      else if (params.search == "radius")
        while (keep < dist.size() && keep < static_cast<size_t>(params.max_neighbors) && dist[keep].first <= params.radius)
          ++keep;
      else
        throw std::invalid_argument("Unknown trajectory search: " + params.search);
      for (size_t k = 0; k < keep; ++k)
        selected.emplace(std::min(i, dist[k].second), std::max(i, dist[k].second));
    }
    pairs.assign(selected.begin(), selected.end());
  }
  else if (params.mode == "exhaustive")
  {
    for (int i = 0; i < num_images; ++i)
      for (int j = i + 1; j < num_images; ++j)
        pairs.emplace_back(i, j);
  }
  else if (params.mode == "sequential")
  {
    for (int i = 0; i < num_images; ++i)
      for (int j = i + 1; j < std::min(num_images, i + 1 + params.sequential_overlap); ++j)
        pairs.emplace_back(i, j);
  }
  else
  {
    throw std::invalid_argument("Unknown matching mode: " + params.mode);
  }
  return pairs;
}

std::vector<FeatureMatch> MatchDescriptors(const cv::Mat &desc1, const cv::Mat &desc2, const MatchingParams &params)
{
  std::vector<FeatureMatch> matches;
  if (desc1.rows < 2 || desc2.rows < 2)
    return matches;

  const cv::flann::KDTreeIndexParams index_params(params.flann_trees);
  cv::flann::Index index2(desc2, index_params);
  const std::vector<int> nn12 = RatioTestNN(index2, desc1, params);

  std::vector<int> nn21;
  if (params.cross_check)
  {
    cv::flann::Index index1(desc1, index_params);
    nn21 = RatioTestNN(index1, desc2, params);
  }

  for (int i = 0; i < desc1.rows; ++i)
  {
    const int j = nn12[i];
    if (j < 0)
      continue;
    if (params.cross_check && nn21[j] != i)
      continue;
    matches.push_back({i, j});
  }
  return matches;
}

} // namespace sfm
