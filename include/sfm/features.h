/*******************************************************************************
* @file    features.h
* @brief   SIFT extraction and keypoint undistortion.
*
* The SIFT backend is chosen at configure time: cv::SIFT (OpenCV >= 4.4),
* cv::xfeatures2d::SIFT (contrib), or the bundled implementation in sift_impl.cpp.
*******************************************************************************/

#ifndef SFM_FEATURES_H_
#define SFM_FEATURES_H_

#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct SiftParams
{
  int max_features = 8000;         ///< keep the strongest N keypoints, <= 0 keeps all
  int octave_layers = 3;
  double contrast_threshold = 0.02; ///< DoG contrast threshold (OpenCV scale), lower gives more features
  double edge_threshold = 10.0;
  double sigma = 1.6;
  int max_image_dim = 3200;        ///< images are downscaled for extraction, keypoints are scaled back
  bool root_sift = true;           ///< L1-normalize + sqrt; improves matching with an L2 matcher
};

/// Runs SIFT on a grayscale 8-bit image. Descriptors are CV_32F, L2-normalized.
void ExtractSift(const cv::Mat &gray, const SiftParams &params, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

/// Bundled SIFT (Lowe 2004), used when OpenCV has none. Exposed for tests.
void ExtractSiftBundled(const cv::Mat &gray, const SiftParams &params, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

/// Fills Features::normalized with undistorted normalized coordinates of each keypoint.
void UndistortKeypoints(const Camera &camera, Features *features);

} // namespace sfm

#endif // SFM_FEATURES_H_
