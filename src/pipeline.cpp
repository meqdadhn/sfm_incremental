#include "sfm/pipeline.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>

#include "sfm/features.h"
#include "sfm/incremental_mapper.h"
#include "sfm/io.h"
#include "sfm/matching.h"
#include "sfm/tracks.h"
#include "sfm/two_view.h"

namespace fs = std::filesystem;

namespace sfm
{
namespace
{
class ScopedTimer
{
public:
  explicit ScopedTimer(std::string name) : name_(std::move(name)), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer()
  {
    LOG(INFO) << "== " << name_ << " took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count() << " s";
  }

private:
  std::string name_;
  std::chrono::steady_clock::time_point t0_;
};

uint64_t SiftHash(const SiftParams &p)
{
  std::ostringstream ss;
  ss << "sift|" << p.max_features << "|" << p.octave_layers << "|" << p.contrast_threshold << "|" << p.edge_threshold << "|" << p.sigma << "|"
     << p.max_image_dim << "|" << p.root_sift;
  return std::hash<std::string>()(ss.str());
}

uint64_t RopHash(const SfmConfig &c, const std::vector<Image> &images)
{
  std::ostringstream ss;
  ss.precision(17);
  ss << SiftHash(c.sift) << "|" << c.matching.mode << "|" << c.matching.sequential_overlap << "|" << c.matching.ratio << "|" << c.matching.cross_check
     << "|" << c.matching.flann_trees << "|" << c.matching.flann_checks << "|" << c.matching.min_matches << "|" << c.two_view.ransac_threshold_px
     << "|" << c.two_view.confidence << "|" << c.two_view.min_inliers << "|" << c.two_view.refine << "|" << c.two_view.refine_loss_px << "|"
     << c.two_view.homography_threshold_px;
  for (const auto &kv : c.cameras)
  {
    const Camera &cam = kv.second;
    ss << "|" << cam.id << "," << cam.fx << "," << cam.fy << "," << cam.cx << "," << cam.cy;
    for (double d : cam.dist)
      ss << "," << d;
  }
  for (const Image &im : images)
    ss << "|" << im.name << ":" << im.camera_id;
  return std::hash<std::string>()(ss.str());
}
} // namespace

Pipeline::Pipeline(SfmConfig config) : config_(std::move(config)) {}

bool Pipeline::Run()
{
  LoadImages();
  ExtractFeatures();
  MatchAndEstimateRops();
  if (!RunIncremental())
    return false;
  FinalTrackingAndBundle();
  Export();
  return true;
}

void Pipeline::LoadImages()
{
  ScopedTimer timer("1. read images");
  rec_.cameras = config_.cameras;
  const std::vector<std::string> names = config_.io.images.empty() ? ListImages(config_.io.image_dir) : config_.io.images;
  if (names.size() < 2)
    throw std::runtime_error("Need at least two images in " + config_.io.image_dir);

  rec_.images.clear();
  for (const std::string &name : names)
  {
    Image image;
    image.id = static_cast<ImageId>(rec_.images.size());
    image.name = name;
    image.path = (fs::path(config_.io.image_dir) / name).string();
    const auto it = config_.image_cameras.find(name);
    image.camera_id = it != config_.image_cameras.end() ? it->second : config_.default_camera;
    if (!rec_.cameras.count(image.camera_id))
      throw std::runtime_error("Image " + name + " uses unknown camera " + std::to_string(image.camera_id));
    rec_.images.push_back(std::move(image));
  }

  // Fill in missing camera sizes from the first image using each camera.
  for (auto &kv : rec_.cameras)
  {
    Camera &cam = kv.second;
    if (cam.width > 0 && cam.height > 0)
      continue;
    for (const Image &image : rec_.images)
    {
      if (image.camera_id != cam.id)
        continue;
      const cv::Mat im = cv::imread(image.path, cv::IMREAD_UNCHANGED);
      if (im.empty())
        throw std::runtime_error("Failed to read " + image.path);
      cam.width = im.cols;
      cam.height = im.rows;
      break;
    }
  }
  LOG(INFO) << rec_.images.size() << " images, " << rec_.cameras.size() << " camera(s)";
}

void Pipeline::ExtractFeatures()
{
  ScopedTimer timer("2. SIFT");
  const uint64_t hash = SiftHash(config_.sift);
  const fs::path cache_dir = fs::path(config_.io.cache_dir) / "features";
  const int n = static_cast<int>(rec_.images.size());
  int done = 0, cached = 0;
  std::vector<std::string> failed; // exceptions must not escape the parallel region

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; ++i)
  {
    Image &image = rec_.images[i];
    const std::string cache_file = (cache_dir / (image.name + ".sift")).string();
    bool from_cache = config_.io.use_cache && LoadFeatures(cache_file, hash, &image.features);
    if (!from_cache)
    {
      const cv::Mat gray = cv::imread(image.path, cv::IMREAD_GRAYSCALE);
      if (gray.empty())
      {
#pragma omp critical
        failed.push_back(image.path);
        continue;
      }
      ExtractSift(gray, config_.sift, &image.features.keypoints, &image.features.descriptors);
      if (config_.io.use_cache)
        SaveFeatures(cache_file, hash, image.features);
    }
    UndistortKeypoints(rec_.cameras.at(image.camera_id), &image.features);
    image.point_ids.assign(image.features.keypoints.size(), kInvalidId);
#pragma omp critical
    {
      ++done;
      cached += from_cache;
      VLOG(1) << "[" << done << "/" << n << "] " << image.name << ": " << image.features.keypoints.size() << " keypoints"
              << (from_cache ? " (cache)" : "");
    }
  }
  if (!failed.empty())
    throw std::runtime_error("Failed to read " + failed.front());
  size_t total = 0;
  for (const Image &image : rec_.images)
    total += image.features.keypoints.size();
  LOG(INFO) << "SIFT: " << total / std::max(n, 1) << " keypoints per image on average, " << cached << " loaded from cache";
}

void Pipeline::MatchAndEstimateRops()
{
  ScopedTimer timer("3. matching + ROP");
  const int n = static_cast<int>(rec_.images.size());
  const uint64_t hash = RopHash(config_, rec_.images);
  const std::string cache_file = (fs::path(config_.io.cache_dir) / "rops.bin").string();

  if (!(config_.io.use_cache && LoadViewGraph(cache_file, hash, n, &view_graph_)))
  {
    const auto pairs = SelectPairs(n, config_.matching);
    std::vector<TwoViewGeometry> results(pairs.size());
    std::vector<char> ok(pairs.size(), 0);
    int done = 0;
    LOG(INFO) << "Matching " << pairs.size() << " pairs";

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < static_cast<int>(pairs.size()); ++k)
    {
      const Image &a = rec_.images[pairs[k].first];
      const Image &b = rec_.images[pairs[k].second];
      const std::vector<FeatureMatch> matches = MatchDescriptors(a.features.descriptors, b.features.descriptors, config_.matching);
      TwoViewGeometry &g = results[k];
      g.image1 = a.id;
      g.image2 = b.id;
      if (static_cast<int>(matches.size()) >= config_.matching.min_matches)
        ok[k] = EstimateTwoViewGeometry(rec_.cameras.at(a.camera_id), rec_.cameras.at(b.camera_id), a.features.normalized, b.features.normalized,
                                        matches, config_.two_view, &g);
#pragma omp critical
      {
        ++done;
        if (done % 100 == 0 || done == static_cast<int>(pairs.size()))
          LOG(INFO) << "  " << done << "/" << pairs.size() << " pairs";
        VLOG(1) << a.name << " - " << b.name << ": " << matches.size() << " matches, "
                << (ok[k] ? std::to_string(g.inliers.size()) + " inliers" : std::string("rejected"));
      }
    }

    view_graph_.Reset(n);
    for (size_t k = 0; k < pairs.size(); ++k)
      if (ok[k])
        view_graph_.AddPair(std::move(results[k]));
    if (config_.io.use_cache)
      SaveViewGraph(cache_file, hash, view_graph_);
  }
  else
  {
    LOG(INFO) << "ROPs loaded from cache";
  }

  // Descriptors are not needed past this point.
  for (Image &image : rec_.images)
    image.features.descriptors.release();
  LOG(INFO) << view_graph_.Pairs().size() << " verified pairs";
}

bool Pipeline::RunIncremental()
{
  ScopedTimer timer("4. incremental SfM");
  IncrementalMapper mapper(config_.incremental, view_graph_, &rec_);
  if (!mapper.Run())
    return false;
  seed1_ = mapper.SeedImage1();
  seed2_ = mapper.SeedImage2();
  return true;
}

void Pipeline::FinalTrackingAndBundle()
{
  ScopedTimer timer("5. tracking + final BA");
  std::vector<bool> use(rec_.images.size());
  for (const Image &image : rec_.images)
    use[image.id] = image.registered;
  const std::vector<Track> tracks = BuildTracks(view_graph_, rec_.images, use, config_.final_ba.min_track_length);
  const int created = TriangulateTracks(tracks, config_.final_ba.triangulation, &rec_);
  LOG(INFO) << tracks.size() << " tracks, " << created << " triangulated, " << rec_.NumObservations() << " observations";

  BundleAdjustmentSetup setup;
  setup.variable_images = rec_.RegisteredImages();
  setup.gauge_image1 = seed1_;
  setup.gauge_image2 = seed2_;
  for (int round = 0; round < config_.final_ba.rounds; ++round)
  {
    const BundleAdjustmentSummary summary = RunBundleAdjustment(config_.final_ba.ba, setup, &rec_);
    if (config_.final_ba.ba.refine_intrinsics)
      for (Image &image : rec_.images)
        UndistortKeypoints(rec_.cameras.at(image.camera_id), &image.features);
    const int removed = rec_.FilterPoints(config_.final_ba.triangulation.max_reprojection_error_px,
                                          config_.final_ba.triangulation.min_triangulation_angle_deg);
    LOG(INFO) << "Final BA round " << round + 1 << ": " << summary.Brief() << ", filtered " << removed << " observations";
  }
  LOG(INFO) << "Final model: " << rec_.NumRegistered() << " images, " << rec_.NumPoints() << " points, mean reprojection error "
            << rec_.MeanReprojectionError() << " px";
}

void Pipeline::Export()
{
  ScopedTimer timer("export");
  ColorizePoints(&rec_);
  const fs::path out(config_.io.output_dir);
  fs::create_directories(out);
  WriteColmapText(rec_, (out / "colmap").string());
  WritePly(rec_, (out / "points.ply").string());
  WritePoses(rec_, (out / "poses.txt").string());
  LOG(INFO) << "Wrote " << (out / "colmap").string() << ", points.ply, poses.txt";
}

} // namespace sfm
