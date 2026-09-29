#include "sfm/features.h"

#include <algorithm>
#include <numeric>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#if defined(SFM_SIFT_OPENCV)
#  include <opencv2/features2d.hpp>
#elif defined(SFM_SIFT_XFEATURES2D)
#  include <opencv2/xfeatures2d.hpp>
#endif

namespace sfm
{
namespace
{
void KeepStrongest(int max_features, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors)
{
  if (max_features <= 0 || static_cast<int>(keypoints->size()) <= max_features)
    return;
  std::vector<int> order(keypoints->size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + max_features, order.end(),
                    [&](int a, int b) { return (*keypoints)[a].response > (*keypoints)[b].response; });
  std::vector<cv::KeyPoint> kept(max_features);
  cv::Mat kept_desc(max_features, descriptors->cols, descriptors->type());
  for (int i = 0; i < max_features; ++i)
  {
    kept[i] = (*keypoints)[order[i]];
    descriptors->row(order[i]).copyTo(kept_desc.row(i));
  }
  *keypoints = std::move(kept);
  *descriptors = kept_desc;
}

void NormalizeDescriptors(bool root_sift, cv::Mat *descriptors)
{
  for (int i = 0; i < descriptors->rows; ++i)
  {
    cv::Mat row = descriptors->row(i);
    if (root_sift)
    {
      const double l1 = cv::norm(row, cv::NORM_L1);
      if (l1 > 0)
        row /= l1;
      cv::sqrt(row, row);
    }
    else
    {
      const double l2 = cv::norm(row, cv::NORM_L2);
      if (l2 > 0)
        row /= l2;
    }
  }
}
} // namespace

void ExtractSift(const cv::Mat &gray, const SiftParams &params, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors)
{
  CV_Assert(gray.type() == CV_8UC1);
  double scale = 1.0;
  cv::Mat input = gray;
  const int max_dim = std::max(gray.cols, gray.rows);
  if (params.max_image_dim > 0 && max_dim > params.max_image_dim)
  {
    scale = static_cast<double>(params.max_image_dim) / max_dim;
    cv::resize(gray, input, cv::Size(), scale, scale, cv::INTER_AREA);
  }

#if defined(SFM_SIFT_OPENCV) || defined(SFM_SIFT_XFEATURES2D)
#  if defined(SFM_SIFT_OPENCV)
  auto sift = cv::SIFT::create(0, params.octave_layers, params.contrast_threshold, params.edge_threshold, params.sigma);
#  else
  auto sift = cv::xfeatures2d::SIFT::create(0, params.octave_layers, params.contrast_threshold, params.edge_threshold, params.sigma);
#  endif
  sift->detectAndCompute(input, cv::noArray(), *keypoints, *descriptors);
  descriptors->convertTo(*descriptors, CV_32F);
#else
  ExtractSiftBundled(input, params, keypoints, descriptors);
#endif

  KeepStrongest(params.max_features, keypoints, descriptors);
  NormalizeDescriptors(params.root_sift, descriptors);

  if (scale != 1.0)
  {
    // Pixel centers: x_full + 0.5 = (x_small + 0.5) / scale
    for (auto &kp : *keypoints)
    {
      kp.pt.x = static_cast<float>((kp.pt.x + 0.5) / scale - 0.5);
      kp.pt.y = static_cast<float>((kp.pt.y + 0.5) / scale - 0.5);
      kp.size = static_cast<float>(kp.size / scale);
    }
  }
}

void UndistortKeypoints(const Camera &camera, Features *features)
{
  features->normalized.clear();
  if (features->keypoints.empty())
    return;
  std::vector<cv::Point2d> pixels(features->keypoints.size());
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = features->keypoints[i].pt;
  std::vector<cv::Point2d> undistorted;
  // No P matrix: output is in normalized image coordinates.
  cv::undistortPoints(pixels, undistorted, camera.K(), camera.DistCoeffs());
  features->normalized.resize(undistorted.size());
  for (size_t i = 0; i < undistorted.size(); ++i)
    features->normalized[i] = Eigen::Vector2d(undistorted[i].x, undistorted[i].y);
}

} // namespace sfm
