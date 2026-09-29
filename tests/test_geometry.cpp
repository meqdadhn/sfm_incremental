#include <random>

#include <gtest/gtest.h>

#include "sfm/bundle_adjustment.h"
#include "sfm/geometry.h"
#include "sfm/resection.h"
#include "sfm/rotation_averaging.h"
#include "sfm/triangulation.h"
#include "sfm/two_view.h"
#include "synthetic.h"

namespace sfm
{
namespace
{
TEST(RotationAveraging, HornEqualsChordalMean)
{
  std::mt19937 rng(3);
  const Eigen::Matrix3d R0 = test::RandomRotation(rng, 180);
  std::vector<Eigen::Matrix3d> candidates;
  Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 6; ++i)
  {
    candidates.push_back(test::RandomRotation(rng, 3.0) * R0);
    sum += candidates.back();
  }
  const Eigen::Matrix3d avg = AverageRotationsQuaternion(candidates);
  EXPECT_LT(RotationAngleDeg(avg, ProjectToSO3(sum)), 1e-6);
  EXPECT_LT(RotationAngleDeg(avg, R0), 3.0);
}

TEST(RotationAveraging, RansacRejectsOutliers)
{
  std::mt19937 rng(5);
  const Eigen::Matrix3d R0 = test::RandomRotation(rng, 180);
  std::vector<Eigen::Matrix3d> candidates;
  for (int i = 0; i < 5; ++i)
    candidates.push_back(test::RandomRotation(rng, 0.3) * R0);
  for (int i = 0; i < 3; ++i)
    candidates.push_back(test::RandomRotation(rng, 180) * R0); // wrong ROPs
  RotationAveragingResult result;
  ASSERT_TRUE(SingleRotationAveragingRansac(candidates, RotationAveragingParams(), &result));
  EXPECT_EQ(result.num_inliers, 5);
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(result.inliers[i]);
  EXPECT_LT(RotationAngleDeg(result.R, R0), 0.3);
}

TEST(RotationAveraging, TwoDisagreeingCandidatesFail)
{
  std::mt19937 rng(7);
  const Eigen::Matrix3d R0 = test::RandomRotation(rng, 180);
  RotationAveragingResult result;
  EXPECT_FALSE(SingleRotationAveragingRansac({R0, test::RandomRotation(rng, 30) * R0 * Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX())},
                                             RotationAveragingParams(), &result));
}

TEST(Resection, KnownRotationWithOutliers)
{
  test::Scene s = test::MakeScene(3, 400, 0.5);
  const Image &image = s.rec.images[1];
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> normalized, pixels;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-3, 3);
  for (size_t k = 0; k < image.features.keypoints.size(); ++k)
  {
    Eigen::Vector3d P = s.points[s.point_of_keypoint[1][k]];
    if (k % 4 == 0)
      P += Eigen::Vector3d(u(rng), u(rng), u(rng)); // 25% wrong 2D-3D correspondences
    X.push_back(P);
    normalized.push_back(image.features.normalized[k]);
    pixels.emplace_back(image.features.keypoints[k].pt.x, image.features.keypoints[k].pt.y);
  }
  ResectionResult result;
  ASSERT_TRUE(ResectKnownRotation(s.camera, s.poses[1].R, X, normalized, pixels, ResectionParams(), &result));
  EXPECT_LT((result.pose.Center() - s.poses[1].Center()).norm(), 0.02);
  EXPECT_LT(result.score_px, 1.0);
  EXPECT_GT(result.num_inliers, static_cast<int>(0.7 * X.size()));
}

TEST(Triangulation, RobustDropsBadObservation)
{
  test::Scene s = test::MakeScene(4, 50, 0.0);
  const Eigen::Vector3d X0 = s.points[0];
  std::vector<const Camera *> cams(4, &s.camera);
  std::vector<Eigen::Vector2d> obs;
  for (int i = 0; i < 4; ++i)
  {
    Eigen::Vector2d uv;
    ProjectNormalized(s.poses[i].Transform(X0), &uv);
    obs.push_back(uv);
  }
  obs[2] += Eigen::Vector2d(0.05, -0.03); // ~50 px off
  Eigen::Vector3d X;
  std::vector<bool> keep;
  ASSERT_TRUE(TriangulateRobust(cams, s.poses, obs, TriangulationParams(), &X, &keep));
  EXPECT_FALSE(keep[2]);
  EXPECT_TRUE(keep[0] && keep[1] && keep[3]);
  EXPECT_LT((X - X0).norm(), 1e-6);
}

TEST(TwoView, RecoversGeneralMotion)
{
  test::Scene s = test::MakeScene(6, 600, 0.5);
  std::mt19937 rng(13);
  for (int b : {1, 3})
  {
    const std::vector<FeatureMatch> matches = test::SceneMatches(s, 0, b, 0.3, rng);
    TwoViewGeometry g;
    ASSERT_TRUE(EstimateTwoViewGeometry(s.camera, s.camera, s.rec.images[0].features.normalized, s.rec.images[b].features.normalized, matches,
                                        TwoViewParams(), &g));
    const Eigen::Matrix3d R_true = s.poses[b].R * s.poses[0].R.transpose();
    const Eigen::Vector3d t_true = (s.poses[b].t - R_true * s.poses[0].t).normalized();
    EXPECT_LT(RotationAngleDeg(g.R, R_true), 0.2);
    EXPECT_LT(RadToDeg(std::acos(std::min(1.0, g.t.dot(t_true)))), 1.0);
    EXPECT_GT(g.inliers.size(), 0.9 * matches.size() / 1.3);
    EXPECT_GT(g.median_triangulation_angle_deg, 1.0);
  }
}

TEST(TwoView, ForwardMotion)
{
  // Camera moving along its optical axis: the epipole is inside the image.
  std::mt19937 rng(17);
  Camera cam = test::TestCamera();
  std::uniform_real_distribution<double> uxy(-6, 6), uz(8, 30);
  const Pose p1;
  const Pose p2 = Pose::FromCenter(test::RandomRotation(rng, 3), Eigen::Vector3d(0.1, 0.05, 1.5));
  std::vector<Eigen::Vector2d> n1, n2;
  std::vector<FeatureMatch> matches;
  std::normal_distribution<double> noise(0, 0.5 / cam.fx);
  for (int i = 0; i < 500; ++i)
  {
    const Eigen::Vector3d X(uxy(rng), uxy(rng), uz(rng));
    Eigen::Vector2d a, b;
    if (!ProjectNormalized(p1.Transform(X), &a) || !ProjectNormalized(p2.Transform(X), &b))
      continue;
    n1.push_back(a + Eigen::Vector2d(noise(rng), noise(rng)));
    n2.push_back(b + Eigen::Vector2d(noise(rng), noise(rng)));
    matches.push_back({static_cast<int>(n1.size()) - 1, static_cast<int>(n2.size()) - 1});
  }
  TwoViewGeometry g;
  ASSERT_TRUE(EstimateTwoViewGeometry(cam, cam, n1, n2, matches, TwoViewParams(), &g));
  EXPECT_LT(RotationAngleDeg(g.R, p2.R), 0.2);
  EXPECT_GT(g.t.dot(p2.t.normalized()), std::cos(DegToRad(2.0)));
}

TEST(BundleAdjustment, ConvergesFromPerturbation)
{
  test::Scene s = test::MakeScene(8, 300, 0.0);
  Reconstruction &rec = s.rec;
  std::mt19937 rng(19);
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 8; ++i)
  {
    rec.images[i].registered = true;
    rec.images[i].pose = s.poses[i];
    if (i >= 2)
      rec.images[i].pose = Pose::FromCenter(test::RandomRotation(rng, 0.5) * s.poses[i].R, s.poses[i].Center() + 0.05 * Eigen::Vector3d(n(rng), n(rng), n(rng)));
  }
  for (size_t p = 0; p < s.points.size(); ++p)
  {
    std::vector<Observation> obs;
    for (int i = 0; i < 8; ++i)
      for (size_t k = 0; k < s.point_of_keypoint[i].size(); ++k)
        if (s.point_of_keypoint[i][k] == static_cast<int>(p))
          obs.push_back({i, static_cast<int>(k)});
    if (obs.size() >= 2)
      rec.AddPoint(s.points[p] + 0.05 * Eigen::Vector3d(n(rng), n(rng), n(rng)), obs);
  }
  BundleAdjustmentSetup setup;
  setup.variable_images = rec.RegisteredImages();
  setup.gauge_image1 = 0;
  setup.gauge_image2 = 1;
  const BundleAdjustmentSummary summary = RunBundleAdjustment(BundleAdjustmentParams(), setup, &rec);
  ASSERT_TRUE(summary.success);
  EXPECT_GT(summary.initial_rms_px, 5.0);
  EXPECT_LT(summary.final_rms_px, 1e-3);
  // Noise-free and gauge fixed on the true poses of images 0 and 1: exact recovery.
  for (int i = 0; i < 8; ++i)
  {
    EXPECT_LT(RotationAngleDeg(rec.images[i].pose.R, s.poses[i].R), 1e-3) << i;
    EXPECT_LT((rec.images[i].pose.Center() - s.poses[i].Center()).norm(), 1e-4) << i;
  }
}

} // namespace
} // namespace sfm
