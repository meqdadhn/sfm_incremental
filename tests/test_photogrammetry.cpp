// The original omega-phi-kappa convention (Rx*Ry*Rz, camera -> map, photogrammetric camera frame)
// against Pose, using the formulas copied from the original SfM.cpp.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "sfm/geometry.h"
#include "sfm/photogrammetry.h"

namespace sfm
{
namespace
{
// Verbatim from the original rotation_Mat (SfM.cpp).
Eigen::Matrix3d OriginalRotationMat(double om, double phi, double kap)
{
  const double c1 = cos(om), s1 = sin(om), c2 = cos(phi), s2 = sin(phi), c3 = cos(kap), s3 = sin(kap);
  Eigen::Matrix3d R;
  R << c2 * c3, -c2 * s3, s2,
       c1 * s3 + s1 * s2 * c3, c1 * c3 - s1 * s2 * s3, -s1 * c2,
       s1 * s3 - c1 * s2 * c3, s1 * c3 + c1 * s2 * s3, c1 * c2;
  return R;
}

// Original collinearity projection (Translation_Estimation_trimming): image coordinates in mm,
// origin at the image center, y up.
Eigen::Vector2d OriginalProject(const Eigen::Matrix3d &R, const Eigen::Vector3d &X0, double c, const Eigen::Vector3d &X)
{
  const Eigen::Vector3d d = X - X0;
  const double den = R(0, 2) * d.x() + R(1, 2) * d.y() + R(2, 2) * d.z();
  return {-c * (R(0, 0) * d.x() + R(1, 0) * d.y() + R(2, 0) * d.z()) / den,
          -c * (R(0, 1) * d.x() + R(1, 1) * d.y() + R(2, 1) * d.z()) / den};
}

TEST(Photogrammetry, MatchesOriginalRotationMat)
{
  std::mt19937 rng(51);
  std::uniform_real_distribution<double> u(-1.4, 1.4);
  for (int i = 0; i < 100; ++i)
  {
    const double om = u(rng), phi = u(rng), kap = 2 * u(rng);
    EXPECT_LT((RotationFromOPK(om, phi, kap) - OriginalRotationMat(om, phi, kap)).norm(), 1e-12);
    double o, p, k;
    OPKFromRotation(RotationFromOPK(om, phi, kap), &o, &p, &k);
    EXPECT_NEAR(o, om, 1e-9);
    EXPECT_NEAR(p, phi, 1e-9);
    EXPECT_NEAR(k, kap, 1e-9);
  }
}

TEST(Photogrammetry, OrderIsNotYawPitchRoll)
{
  const double om = 0.3, phi = -0.2, kap = 0.9;
  const Eigen::Matrix3d zyx = (Eigen::AngleAxisd(kap, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(om, Eigen::Vector3d::UnitX()))
                                  .toRotationMatrix();
  EXPECT_GT(RotationAngleDeg(RotationFromOPK(om, phi, kap), zyx), 5.0);
}

TEST(Photogrammetry, PoseProjectsLikeOriginalCollinearity)
{
  // Camera: c = 8.8 mm, 2.41 um pixels, 5472 x 3648 (so fx = c / pixel size).
  const double c = 8.8, pix = 0.00241;
  Camera cam;
  cam.width = 5472;
  cam.height = 3648;
  cam.fx = cam.fy = c / pix;
  cam.cx = cam.width / 2.0;
  cam.cy = cam.height / 2.0;

  std::mt19937 rng(53);
  std::uniform_real_distribution<double> ang(-0.15, 0.15), kap(-3.1, 3.1), xy(-20, 20);
  for (int i = 0; i < 50; ++i)
  {
    // Nadir-ish camera 100 m above the ground: omega = phi = 0 looks straight down (-z).
    const double om = ang(rng), phi = ang(rng), k = kap(rng);
    const Eigen::Vector3d X0(xy(rng), xy(rng), 100.0);
    const Eigen::Vector3d X(xy(rng), xy(rng), 0.0);

    const Eigen::Vector2d mm = OriginalProject(OriginalRotationMat(om, phi, k), X0, c, X);
    const Eigen::Vector2d px_original(cam.width / 2.0 + mm.x() / pix, cam.height / 2.0 - mm.y() / pix); // y up -> rows down

    const Pose pose = PoseFromOPK(om, phi, k, X0);
    ASSERT_GT(pose.Transform(X).z(), 0.0); // in front of the camera
    EXPECT_LT((cam.Project(pose.Transform(X)) - px_original).norm(), 1e-6);
    EXPECT_LT((pose.Center() - X0).norm(), 1e-9);

    double o2, p2, k2;
    OPKFromPose(pose, &o2, &p2, &k2);
    EXPECT_NEAR(o2, om, 1e-9);
    EXPECT_NEAR(p2, phi, 1e-9);
    EXPECT_NEAR(std::remainder(k2 - k, 2 * kPi), 0.0, 1e-9);
  }
}

} // namespace
} // namespace sfm
