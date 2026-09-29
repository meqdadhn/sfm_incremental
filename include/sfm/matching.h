/*******************************************************************************
* @file    matching.h
* @brief   Image pair selection and FLANN descriptor matching.
*******************************************************************************/

#ifndef SFM_MATCHING_H_
#define SFM_MATCHING_H_

#include <string>
#include <utility>
#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct MatchingParams
{
  std::string mode = "exhaustive"; ///< "exhaustive", "sequential" or "trajectory"
  int sequential_overlap = 10;     ///< sequential mode: match each image with the next N images
  // Trajectory mode (needs position priors): match each image with its nearest images only.
  std::string search = "knn";      ///< "knn" or "radius"
  int knn = 20;                    ///< neighbours per image
  double radius = 10.0;            ///< search radius, in trajectory units (meters for EXIF GPS)
  int max_neighbors = 50;          ///< radius search: keep at most this many (closest) neighbours
  double ratio = 0.8;              ///< Lowe ratio test
  bool cross_check = true;         ///< keep only mutual nearest neighbours
  int flann_trees = 4;
  int flann_checks = 128;
  int min_matches = 30;            ///< pairs with fewer putative matches are dropped before geometry
};

/// Image pairs (i < j) to match, according to the mode. Trajectory mode uses Image::prior_position;
/// images without a prior are matched with every image.
std::vector<std::pair<ImageId, ImageId>> SelectPairs(const std::vector<Image> &images, const MatchingParams &params);

/// Ratio-test (+ optional cross-check) matching of two descriptor sets.
std::vector<FeatureMatch> MatchDescriptors(const cv::Mat &desc1, const cv::Mat &desc2, const MatchingParams &params);

} // namespace sfm

#endif // SFM_MATCHING_H_
