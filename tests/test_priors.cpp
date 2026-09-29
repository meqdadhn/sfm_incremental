// Trajectory priors from file and trajectory-based pair selection.

#include <filesystem>
#include <fstream>
#include <set>

#include <gtest/gtest.h>

#include "sfm/matching.h"
#include "sfm/priors.h"

namespace sfm
{
namespace
{
void WriteFile(const std::string &path, const std::string &text) { std::ofstream(path) << text; }

std::vector<Image> NamedImages(int n)
{
  std::vector<Image> images(n);
  for (int i = 0; i < n; ++i)
  {
    images[i].id = i;
    images[i].name = "img_" + std::to_string(i) + ".jpg";
  }
  return images;
}

TEST(Priors, TrajectoryFileByName)
{
  const std::string path = (std::filesystem::path(testing::TempDir()) / "traj_named.txt").string();
  WriteFile(path, "# name X Y Z roll pitch yaw\nimg_2.jpg 1.5 -2 30 0 -90 45\nimg_0.jpg 0 0 0 nan nan nan\nunknown.jpg 9 9 9\n");
  std::vector<Image> images = NamedImages(3);
  TrajectoryParams params;
  params.source = "file";
  params.file = path;
  EXPECT_EQ(LoadTrajectoryPriors(params, &images), 2);
  EXPECT_TRUE(images[0].has_prior);
  EXPECT_FALSE(images[1].has_prior);
  EXPECT_TRUE(images[2].has_prior);
  EXPECT_TRUE(images[2].prior_position.isApprox(Eigen::Vector3d(1.5, -2, 30)));
}

TEST(Priors, TrajectoryFileOriginalFormat)
{
  // Original format: omega phi kappa X Y Z per image, in image order.
  const std::string path = (std::filesystem::path(testing::TempDir()) / "traj_original.txt").string();
  WriteFile(path, "0.1 0.2 45.0 100 200 50\n0.0 0.0 46.0 110 205 51\n");
  std::vector<Image> images = NamedImages(3);
  TrajectoryParams params;
  params.source = "file";
  params.file = path;
  EXPECT_EQ(LoadTrajectoryPriors(params, &images), 2);
  EXPECT_TRUE(images[1].prior_position.isApprox(Eigen::Vector3d(110, 205, 51)));
  EXPECT_FALSE(images[2].has_prior);

  params.source = "none";
  EXPECT_EQ(LoadTrajectoryPriors(params, &images), 0);
}

TEST(Matching, TrajectoryPairSelection)
{
  // 10 images on a line, 1 m apart.
  std::vector<Image> images(10);
  for (int i = 0; i < 10; ++i)
  {
    images[i].id = i;
    images[i].has_prior = true;
    images[i].prior_position = Eigen::Vector3d(i, 0, 0);
  }
  MatchingParams params;
  params.mode = "trajectory";
  params.search = "knn";
  params.knn = 2;
  auto pairs = SelectPairs(images, params);
  std::set<std::pair<int, int>> s(pairs.begin(), pairs.end());
  EXPECT_TRUE(s.count({0, 1}) && s.count({0, 2}) && s.count({3, 4}) && s.count({4, 5})); // ends reach 2 ahead
  EXPECT_FALSE(s.count({0, 3}));
  for (const auto &p : pairs)
    EXPECT_LE(p.second - p.first, 2);

  params.search = "radius";
  params.radius = 3.5;
  pairs = SelectPairs(images, params);
  for (const auto &p : pairs)
    EXPECT_LE(p.second - p.first, 3);
  EXPECT_EQ(pairs.size(), 9u + 8u + 7u);

  // An image without a prior is matched with everything.
  images[5].has_prior = false;
  pairs = SelectPairs(images, params);
  int with5 = 0;
  for (const auto &p : pairs)
    with5 += (p.first == 5 || p.second == 5);
  EXPECT_EQ(with5, 9);
}

} // namespace
} // namespace sfm
