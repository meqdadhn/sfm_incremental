// SIFT + matching on a synthetic textured image and a rotated / scaled copy with known homography.

#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "sfm/features.h"
#include "sfm/matching.h"

namespace sfm
{
namespace
{
cv::Mat TexturedImage()
{
  cv::Mat img(480, 640, CV_8UC1, cv::Scalar(128));
  std::mt19937 rng(31);
  std::uniform_int_distribution<int> ux(0, 639), uy(0, 479), ur(4, 30), uc(0, 255);
  for (int i = 0; i < 400; ++i)
    cv::circle(img, {ux(rng), uy(rng)}, ur(rng), cv::Scalar(uc(rng)), -1, cv::LINE_AA);
  for (int i = 0; i < 150; ++i)
    cv::rectangle(img, cv::Rect(ux(rng), uy(rng), ur(rng), ur(rng)), cv::Scalar(uc(rng)), -1);
  cv::GaussianBlur(img, img, cv::Size(), 1.0);
  return img;
}

void CheckAgainstHomography(bool bundled)
{
  const cv::Mat img1 = TexturedImage();
  const cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(320, 240), 30.0, 0.8);
  cv::Mat img2;
  cv::warpAffine(img1, img2, M, img1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128));

  SiftParams params;
  std::vector<cv::KeyPoint> k1, k2;
  cv::Mat d1, d2;
  if (bundled)
  {
    ExtractSiftBundled(img1, params, &k1, &d1);
    ExtractSiftBundled(img2, params, &k2, &d2);
  }
  else
  {
    ExtractSift(img1, params, &k1, &d1);
    ExtractSift(img2, params, &k2, &d2);
  }
  ASSERT_GT(k1.size(), 300u);
  ASSERT_EQ(d1.cols, 128);

  const std::vector<FeatureMatch> matches = MatchDescriptors(d1, d2, MatchingParams());
  ASSERT_GT(matches.size(), 150u);
  int correct = 0;
  for (const FeatureMatch &m : matches)
  {
    const cv::Point2f p = k1[m.idx1].pt;
    const cv::Point2f q(static_cast<float>(M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2)),
                        static_cast<float>(M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2)));
    if (cv::norm(q - k2[m.idx2].pt) < 2.0)
      ++correct;
  }
  EXPECT_GT(static_cast<double>(correct) / matches.size(), 0.9) << correct << " / " << matches.size();
}

TEST(Sift, BundledIsRotationAndScaleInvariant) { CheckAgainstHomography(true); }
TEST(Sift, ConfiguredBackend) { CheckAgainstHomography(false); }

} // namespace
} // namespace sfm
