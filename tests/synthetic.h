// Synthetic scenes with known ground truth for the geometry tests.

#ifndef SFM_TESTS_SYNTHETIC_H_
#define SFM_TESTS_SYNTHETIC_H_

#include <random>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/features.h"
#include "sfm/reconstruction.h"

namespace sfm
{
namespace test
{
inline Camera TestCamera()
{
  Camera c;
  c.id = 0;
  c.width = 1280;
  c.height = 960;
  c.fx = 1000;
  c.fy = 1000;
  c.cx = 640;
  c.cy = 480;
  c.dist = {-0.05, 0.01, 0.0005, -0.0003, 0.0};
  return c;
}

inline Eigen::Matrix3d RandomRotation(std::mt19937 &rng, double max_angle_deg)
{
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, max_angle_deg * 3.14159265358979 / 180.0);
  return Eigen::AngleAxisd(u(rng), Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized()).toRotationMatrix();
}

/// Camera at C looking at `target` (OpenCV frame: z forward, y down).
inline Pose LookAt(const Eigen::Vector3d &C, const Eigen::Vector3d &target, const Eigen::Vector3d &up = Eigen::Vector3d(0, 0, 1))
{
  const Eigen::Vector3d z = (target - C).normalized();
  const Eigen::Vector3d x = z.cross(up).normalized(); // right
  const Eigen::Vector3d y = z.cross(x);               // down
  Eigen::Matrix3d R;
  R.row(0) = x.transpose();
  R.row(1) = y.transpose();
  R.row(2) = z.transpose();
  return Pose::FromCenter(R, C);
}

struct Scene
{
  Camera camera;
  std::vector<Pose> poses;
  std::vector<Eigen::Vector3d> points;
  /// per image: point index of each keypoint
  std::vector<std::vector<int>> point_of_keypoint;
  Reconstruction rec; ///< images with features filled (no registration)
};

/// Cameras on an arc around a box of points, each image sees a subset.
inline Scene MakeScene(int num_images, int num_points, double pixel_noise, unsigned seed = 1)
{
  std::mt19937 rng(seed);
  Scene s;
  s.camera = TestCamera();
  std::uniform_real_distribution<double> ux(-4, 4), uy(-4, 4), uz(-1.5, 1.5);
  for (int i = 0; i < num_points; ++i)
    s.points.emplace_back(ux(rng), uy(rng), uz(rng));

  std::normal_distribution<double> jitter(0.0, 0.3);
  for (int i = 0; i < num_images; ++i)
  {
    const double a = -1.2 + 2.4 * i / std::max(1, num_images - 1);
    const Eigen::Vector3d C(12.0 * std::sin(a), -12.0 * std::cos(a), 3.0 + jitter(rng));
    const Eigen::Vector3d target(jitter(rng), jitter(rng), 0.0);
    s.poses.push_back(LookAt(C, target));
  }

  s.rec.cameras[0] = s.camera;
  std::normal_distribution<double> noise(0.0, pixel_noise);
  for (int i = 0; i < num_images; ++i)
  {
    Image image;
    image.id = i;
    image.name = "img_" + std::to_string(i);
    image.camera_id = 0;
    std::vector<int> ids;
    for (int p = 0; p < num_points; ++p)
    {
      const Eigen::Vector3d Xc = s.poses[i].Transform(s.points[p]);
      if (Xc.z() <= 0.1)
        continue;
      const Eigen::Vector2d px = s.camera.Project(Xc) + Eigen::Vector2d(noise(rng), noise(rng));
      if (px.x() < 0 || px.y() < 0 || px.x() >= s.camera.width || px.y() >= s.camera.height)
        continue;
      image.features.keypoints.emplace_back(cv::Point2f(static_cast<float>(px.x()), static_cast<float>(px.y())), 4.0f);
      ids.push_back(p);
    }
    UndistortKeypoints(s.camera, &image.features);
    image.point_ids.assign(image.features.keypoints.size(), kInvalidId);
    s.point_of_keypoint.push_back(ids);
    s.rec.images.push_back(std::move(image));
  }
  return s;
}

/// True correspondences between two images of the scene, plus `outlier_ratio` random wrong ones.
inline std::vector<FeatureMatch> SceneMatches(const Scene &s, int a, int b, double outlier_ratio, std::mt19937 &rng)
{
  std::vector<FeatureMatch> matches;
  std::vector<int> index_b(s.points.size(), -1);
  for (size_t k = 0; k < s.point_of_keypoint[b].size(); ++k)
    index_b[s.point_of_keypoint[b][k]] = static_cast<int>(k);
  for (size_t k = 0; k < s.point_of_keypoint[a].size(); ++k)
  {
    const int j = index_b[s.point_of_keypoint[a][k]];
    if (j >= 0)
      matches.push_back({static_cast<int>(k), j});
  }
  const int n_out = static_cast<int>(outlier_ratio * matches.size());
  std::uniform_int_distribution<int> ua(0, static_cast<int>(s.point_of_keypoint[a].size()) - 1);
  std::uniform_int_distribution<int> ub(0, static_cast<int>(s.point_of_keypoint[b].size()) - 1);
  for (int k = 0; k < n_out; ++k)
    matches.push_back({ua(rng), ub(rng)});
  return matches;
}

/// Similarity (s, R, t) with dst ~ s * R * src + t (Umeyama).
inline void AlignSimilarity(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &dst, double *scale,
                            Eigen::Matrix3d *R, Eigen::Vector3d *t)
{
  Eigen::Matrix3Xd A(3, src.size()), B(3, dst.size());
  for (size_t i = 0; i < src.size(); ++i)
  {
    A.col(i) = src[i];
    B.col(i) = dst[i];
  }
  const Eigen::Matrix4d T = Eigen::umeyama(A, B, true);
  *scale = T.block<3, 3>(0, 0).col(0).norm();
  *R = T.block<3, 3>(0, 0) / *scale;
  *t = T.block<3, 1>(0, 3);
}

} // namespace test
} // namespace sfm

#endif // SFM_TESTS_SYNTHETIC_H_
