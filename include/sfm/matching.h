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
  std::string mode = "exhaustive"; ///< "exhaustive" or "sequential"
  int sequential_overlap = 10;     ///< sequential mode: match each image with the next N images
  double ratio = 0.8;              ///< Lowe ratio test
  bool cross_check = true;         ///< keep only mutual nearest neighbours
  int flann_trees = 4;
  int flann_checks = 128;
  int min_matches = 30;            ///< pairs with fewer putative matches are dropped before geometry
};

/// Image pairs (i < j) to match, according to the mode.
std::vector<std::pair<ImageId, ImageId>> SelectPairs(int num_images, const MatchingParams &params);

/// Ratio-test (+ optional cross-check) matching of two descriptor sets.
std::vector<FeatureMatch> MatchDescriptors(const cv::Mat &desc1, const cv::Mat &desc2, const MatchingParams &params);

} // namespace sfm

#endif // SFM_MATCHING_H_
