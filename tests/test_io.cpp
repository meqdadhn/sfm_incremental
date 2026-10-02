// Incremental SfM cache: round trip and rejection of stale files.

#include <filesystem>

#include <gtest/gtest.h>

#include "sfm/io.h"
#include "synthetic.h"

namespace sfm
{
namespace
{
std::string TempPath(const std::string &name) { return (std::filesystem::temp_directory_path() / name).string(); }

/// Scene with every other image registered at its true pose and a refined focal length.
test::Scene IncrementalResult()
{
  test::Scene s = test::MakeScene(6, 200, 0.0);
  for (size_t i = 0; i < s.rec.images.size(); i += 2)
  {
    s.rec.images[i].registered = true;
    s.rec.images[i].pose = s.poses[i];
  }
  s.rec.cameras.at(0).fx = 1012.5;
  s.rec.cameras.at(0).dist[0] = -0.07;
  return s;
}

TEST(IncrementalCache, RoundTrip)
{
  const test::Scene saved = IncrementalResult();
  const std::string path = TempPath("sfm_test_incremental.bin");
  SaveIncremental(path, 42, saved.rec, 0, 2);

  test::Scene loaded = test::MakeScene(6, 200, 0.0);
  ImageId seed1 = kInvalidId, seed2 = kInvalidId;
  ASSERT_TRUE(LoadIncremental(path, 42, &loaded.rec, &seed1, &seed2));
  EXPECT_EQ(seed1, 0);
  EXPECT_EQ(seed2, 2);
  for (size_t i = 0; i < saved.rec.images.size(); ++i)
  {
    EXPECT_EQ(loaded.rec.images[i].registered, saved.rec.images[i].registered);
    EXPECT_TRUE(loaded.rec.images[i].pose.R.isApprox(saved.rec.images[i].pose.R, 0.0));
    EXPECT_TRUE(loaded.rec.images[i].pose.t.isApprox(saved.rec.images[i].pose.t, 0.0));
  }
  EXPECT_EQ(loaded.rec.cameras.at(0).fx, 1012.5);
  EXPECT_EQ(loaded.rec.cameras.at(0).dist[0], -0.07);
  EXPECT_EQ(loaded.rec.cameras.at(0).width, saved.rec.cameras.at(0).width);
  std::filesystem::remove(path);
}

TEST(IncrementalCache, RejectsStaleFile)
{
  const std::string path = TempPath("sfm_test_incremental_stale.bin");
  SaveIncremental(path, 42, IncrementalResult().rec, 0, 2);
  ImageId seed1 = kInvalidId, seed2 = kInvalidId;

  test::Scene other_hash = test::MakeScene(6, 200, 0.0);
  EXPECT_FALSE(LoadIncremental(path, 43, &other_hash.rec, &seed1, &seed2));

  test::Scene renamed = test::MakeScene(6, 200, 0.0);
  renamed.rec.images[3].name = "other.jpg";
  EXPECT_FALSE(LoadIncremental(path, 42, &renamed.rec, &seed1, &seed2));
  EXPECT_FALSE(renamed.rec.images[0].registered); // untouched on failure
  EXPECT_EQ(renamed.rec.cameras.at(0).fx, test::TestCamera().fx);

  test::Scene fewer = test::MakeScene(5, 200, 0.0);
  EXPECT_FALSE(LoadIncremental(path, 42, &fewer.rec, &seed1, &seed2));
  EXPECT_EQ(seed1, kInvalidId);
  std::filesystem::remove(path);
}

} // namespace
} // namespace sfm
