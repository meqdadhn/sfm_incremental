// Orthophoto of a textured ground plane seen by nadir cameras, compared with the true texture.

#include <filesystem>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "sfm/ortho.h"
#include "synthetic.h"

namespace sfm
{
namespace
{
constexpr double kPlaneZ = 10.0;    // map Z of the ground (cameras look along +Z)
constexpr double kHalfSize = 6.0;   // texture covers [-6, 6]^2 in map units
constexpr double kTexelsPerUnit = 100.0;

cv::Mat GroundTexture()
{
  const int size = static_cast<int>(2 * kHalfSize * kTexelsPerUnit);
  cv::Mat tex(size, size, CV_8UC3, cv::Scalar(120, 120, 120));
  std::mt19937 rng(41);
  std::uniform_int_distribution<int> up(0, size - 1), ur(5, 40), uc(0, 255);
  for (int i = 0; i < 1500; ++i)
    cv::circle(tex, {up(rng), up(rng)}, ur(rng), cv::Scalar(uc(rng), uc(rng), uc(rng)), -1, cv::LINE_AA);
  cv::GaussianBlur(tex, tex, cv::Size(), 1.5);
  return tex;
}

cv::Vec3d SampleTexture(const cv::Mat &tex, double X, double Y)
{
  cv::Mat px;
  cv::getRectSubPix(tex, cv::Size(1, 1), cv::Point2f(static_cast<float>((X + kHalfSize) * kTexelsPerUnit - 0.5),
                                                     static_cast<float>((Y + kHalfSize) * kTexelsPerUnit - 0.5)), px);
  const cv::Vec3b v = px.at<cv::Vec3b>(0, 0);
  return {double(v[0]), double(v[1]), double(v[2])};
}

/// Renders the plane from `pose`, including the camera's lens distortion.
cv::Mat RenderView(const Camera &cam, const Pose &pose, const cv::Mat &tex)
{
  std::vector<cv::Point2d> px;
  for (int r = 0; r < cam.height; ++r)
    for (int c = 0; c < cam.width; ++c)
      px.emplace_back(c, r);
  std::vector<cv::Point2d> und;
  cv::undistortPoints(px, und, cam.K(), cam.DistCoeffs());
  cv::Mat map_x(cam.height, cam.width, CV_32F), map_y(cam.height, cam.width, CV_32F);
  const Eigen::Vector3d C = pose.Center();
  for (int i = 0; i < static_cast<int>(und.size()); ++i)
  {
    const Eigen::Vector3d d = pose.R.transpose() * Eigen::Vector3d(und[i].x, und[i].y, 1.0);
    const double s = (kPlaneZ - C.z()) / d.z();
    const Eigen::Vector3d X = C + s * d;
    map_x.at<float>(i / cam.width, i % cam.width) = static_cast<float>((X.x() + kHalfSize) * kTexelsPerUnit - 0.5);
    map_y.at<float>(i / cam.width, i % cam.width) = static_cast<float>((X.y() + kHalfSize) * kTexelsPerUnit - 0.5);
  }
  cv::Mat img;
  cv::remap(tex, img, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return img;
}

TEST(Ortho, PlanarGroundMatchesTexture)
{
  const cv::Mat tex = GroundTexture();
  const std::string dir = (std::filesystem::path(testing::TempDir()) / "sfm_ortho_test").string();
  std::filesystem::create_directories(dir);

  Reconstruction rec;
  Camera cam = test::TestCamera();
  cam.width = 640;
  cam.height = 480;
  cam.fx = cam.fy = 500;
  cam.cx = 320;
  cam.cy = 240;
  rec.cameras[0] = cam;

  // 3 x 3 grid of nadir cameras with a few degrees of tilt.
  std::mt19937 rng(43);
  int id = 0;
  for (double y : {-2.0, 0.0, 2.0})
  {
    for (double x : {-2.0, 0.0, 2.0})
    {
      Image image;
      image.id = id;
      image.name = "nadir_" + std::to_string(id) + ".png";
      image.path = (std::filesystem::path(dir) / image.name).string();
      image.camera_id = 0;
      image.registered = true;
      image.pose = Pose::FromCenter(test::RandomRotation(rng, 3.0), Eigen::Vector3d(x, y, 0.0));
      ASSERT_TRUE(cv::imwrite(image.path, RenderView(cam, image.pose, tex)));
      rec.images.push_back(std::move(image));
      ++id;
    }
  }

  // Sparse ground points (the DEM source). Observations are placeholders; only X is used.
  const int n = 41;
  for (Image &image : rec.images)
  {
    image.features.keypoints.assign(n * n, cv::KeyPoint(0, 0, 1));
    image.point_ids.assign(n * n, kInvalidId);
  }
  for (int i = 0; i < n * n; ++i)
  {
    const Eigen::Vector3d X(-4.0 + 8.0 * (i % n) / (n - 1), -4.0 + 8.0 * (i / n) / (n - 1), kPlaneZ);
    rec.AddPoint(X, {{0, i}, {1, i}});
  }

  OrthoParams params;
  params.gsd = 0.02;
  params.bounds_percentile = 0.0;
  params.image_batch = 4; // exercise several batches
  Orthophoto ortho;
  ASSERT_TRUE(GenerateOrthophoto(rec, params, &ortho));
  ASSERT_EQ(ortho.image.cols, 400);
  ASSERT_EQ(ortho.image.rows, 400);

  double err = 0.0;
  int covered = 0;
  for (int r = 0; r < ortho.image.rows; ++r)
  {
    for (int c = 0; c < ortho.image.cols; ++c)
    {
      const cv::Vec4b v = ortho.image.at<cv::Vec4b>(r, c);
      if (v[3] == 0)
        continue;
      ++covered;
      const cv::Point2d P = ortho.PixelToMap(c, r);
      const cv::Vec3d gt = SampleTexture(tex, P.x, P.y);
      err += (std::abs(v[0] - gt[0]) + std::abs(v[1] - gt[1]) + std::abs(v[2] - gt[2])) / 3.0;
    }
  }
  const double coverage = static_cast<double>(covered) / (ortho.image.rows * ortho.image.cols);
  EXPECT_GT(coverage, 0.98);
  EXPECT_LT(err / covered, 6.0); // mean abs intensity error (0-255), resampling only

  // The DEM of a flat plane is flat.
  double zmin, zmax;
  cv::minMaxLoc(ortho.dem, &zmin, &zmax);
  EXPECT_NEAR(zmin, kPlaneZ, 1e-3);
  EXPECT_NEAR(zmax, kPlaneZ, 1e-3);
}

} // namespace
} // namespace sfm
