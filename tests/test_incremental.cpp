// End-to-end on a synthetic scene: ROPs -> incremental SfM -> tracking -> final BA.

#include <random>

#include <gtest/gtest.h>

#include "sfm/bundle_adjustment.h"
#include "sfm/geometry.h"
#include "sfm/incremental_mapper.h"
#include "sfm/tracks.h"
#include "sfm/two_view.h"
#include "synthetic.h"

namespace sfm
{
namespace
{
ViewGraph BuildViewGraph(const test::Scene &s, double outlier_ratio, int max_gap)
{
  std::mt19937 rng(23);
  const int n = static_cast<int>(s.rec.images.size());
  ViewGraph vg(n);
  for (int a = 0; a < n; ++a)
  {
    for (int b = a + 1; b < std::min(n, a + 1 + max_gap); ++b)
    {
      const std::vector<FeatureMatch> matches = test::SceneMatches(s, a, b, outlier_ratio, rng);
      TwoViewGeometry g;
      g.image1 = a;
      g.image2 = b;
      if (EstimateTwoViewGeometry(s.camera, s.camera, s.rec.images[a].features.normalized, s.rec.images[b].features.normalized, matches,
                                  TwoViewParams(), &g))
        vg.AddPair(std::move(g));
    }
  }
  return vg;
}

void ExpectMatchesGroundTruth(const test::Scene &s, const Reconstruction &rec, double max_rot_deg, double max_center_rel)
{
  std::vector<Eigen::Vector3d> est, gt;
  for (const Image &image : rec.images)
  {
    ASSERT_TRUE(image.registered) << image.name;
    est.push_back(image.pose.Center());
    gt.push_back(s.poses[image.id].Center());
  }
  double scale;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  test::AlignSimilarity(est, gt, &scale, &R, &t);
  double extent = 0.0;
  for (const auto &c : gt)
    extent = std::max(extent, (c - gt[0]).norm());
  for (const Image &image : rec.images)
  {
    const Eigen::Vector3d C = scale * R * image.pose.Center() + t;
    EXPECT_LT((C - s.poses[image.id].Center()).norm() / extent, max_center_rel) << image.name;
    // World rotation of the estimate expressed in the ground-truth frame: R_cw * R^T.
    EXPECT_LT(RotationAngleDeg(image.pose.R * R.transpose(), s.poses[image.id].R), max_rot_deg) << image.name;
  }
}

TEST(IncrementalSfM, SyntheticSequence)
{
  test::Scene s = test::MakeScene(16, 1500, 0.5);
  const ViewGraph vg = BuildViewGraph(s, 0.2, 5);
  Reconstruction &rec = s.rec;

  IncrementalParams params;
  params.ba_interval = 3;
  params.ba_window = 6;
  IncrementalMapper mapper(params, vg, &rec);
  ASSERT_TRUE(mapper.Run());
  EXPECT_EQ(rec.NumRegistered(), 16);
  ExpectMatchesGroundTruth(s, rec, 0.3, 0.01);

  // Final stage: tracks over all ROP inliers, then global BA.
  std::vector<bool> use(rec.images.size(), true);
  const std::vector<Track> tracks = BuildTracks(vg, rec.images, use, 2);
  EXPECT_GT(TriangulateTracks(tracks, TriangulationParams(), &rec), 1000);
  BundleAdjustmentSetup setup;
  setup.variable_images = rec.RegisteredImages();
  setup.gauge_image1 = mapper.SeedImage1();
  setup.gauge_image2 = mapper.SeedImage2();
  const BundleAdjustmentSummary summary = RunBundleAdjustment(BundleAdjustmentParams(), setup, &rec);
  ASSERT_TRUE(summary.success);
  EXPECT_LT(summary.final_rms_px, 0.8);
  ExpectMatchesGroundTruth(s, rec, 0.1, 0.003);
}

TEST(IncrementalSfM, SingleConnectionFallback)
{
  // Only consecutive pairs: every new image has exactly one registered neighbour.
  test::Scene s = test::MakeScene(8, 1500, 0.5, 29);
  const ViewGraph vg = BuildViewGraph(s, 0.1, 1);
  IncrementalParams params;
  IncrementalMapper mapper(params, vg, &s.rec);
  ASSERT_TRUE(mapper.Run());
  EXPECT_EQ(s.rec.NumRegistered(), 8);
  ExpectMatchesGroundTruth(s, s.rec, 0.5, 0.02);
}

} // namespace
} // namespace sfm
