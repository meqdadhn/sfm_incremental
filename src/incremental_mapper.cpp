#include "sfm/incremental_mapper.h"

#include <algorithm>
#include <unordered_map>

#include <glog/logging.h>

#include "sfm/geometry.h"

namespace sfm
{
IncrementalMapper::IncrementalMapper(const IncrementalParams &params, const ViewGraph &view_graph, Reconstruction *reconstruction)
    : params_(params), view_graph_(view_graph), rec_(reconstruction)
{
}

bool IncrementalMapper::Run()
{
  if (!Initialize())
  {
    LOG(ERROR) << "No seed pair could be initialized";
    return false;
  }
  while (RegisterNextImage())
  {
  }
  if (since_window_ba_ > 0)
    WindowBundleAdjustment();

  const int n = static_cast<int>(rec_->images.size());
  LOG(INFO) << "Incremental SfM registered " << rec_->NumRegistered() << " / " << n << " images, " << rec_->NumPoints() << " points";
  for (const Image &image : rec_->images)
    if (!image.registered)
      LOG(INFO) << "  not registered: " << image.name;
  return true;
}

// ---------------------------------------------------------------------------
// Seed
// ---------------------------------------------------------------------------

bool IncrementalMapper::Initialize()
{
  // Images ordered by the number of verified connections, as in the original code.
  const int n = view_graph_.NumImages();
  std::vector<ImageId> by_degree(n);
  for (int i = 0; i < n; ++i)
    by_degree[i] = i;
  std::stable_sort(by_degree.begin(), by_degree.end(),
                   [&](ImageId a, ImageId b) { return view_graph_.Neighbors(a).size() > view_graph_.Neighbors(b).size(); });

  int trials = 0;
  for (ImageId a : by_degree)
  {
    std::vector<ImageId> neighbors = view_graph_.Neighbors(a);
    std::stable_sort(neighbors.begin(), neighbors.end(),
                     [&](ImageId x, ImageId y) { return view_graph_.NumInliers(a, x) > view_graph_.NumInliers(a, y); });
    for (ImageId b : neighbors)
    {
      const TwoViewGeometry *g = view_graph_.Pair(a, b);
      if (g->median_triangulation_angle_deg < params_.seed_min_triangulation_angle_deg ||
          g->homography_inlier_ratio > params_.seed_max_homography_ratio)
        continue;
      if (trials++ >= params_.seed_max_trials)
        return false;
      if (InitializeFromPair(a, b))
        return true;
    }
  }
  return false;
}

bool IncrementalMapper::InitializeFromPair(ImageId a, ImageId b)
{
  rec_->ClearPoints();
  for (Image &image : rec_->images)
    image.registered = false;

  Image &ia = rec_->images[a];
  Image &ib = rec_->images[b];
  ia.pose = Pose();
  view_graph_.RelativePose(a, b, &ib.pose.R, &ib.pose.t); // baseline = 1 sets the scale
  ia.registered = ib.registered = true;

  const std::vector<const Camera *> cams = {&rec_->CameraOf(a), &rec_->CameraOf(b)};
  const std::vector<Pose> poses = {ia.pose, ib.pose};
  for (const FeatureMatch &m : view_graph_.Matches(a, b))
  {
    Eigen::Vector3d X;
    std::vector<bool> keep;
    if (TriangulateRobust(cams, poses, {ia.features.normalized[m.idx1], ib.features.normalized[m.idx2]}, params_.triangulation, &X, &keep) &&
        keep[0] && keep[1])
      rec_->AddPoint(X, {{a, m.idx1}, {b, m.idx2}});
  }

  BundleAdjustmentSetup setup;
  setup.variable_images = {a, b};
  setup.gauge_image1 = a;
  setup.gauge_image2 = b;
  RunBundleAdjustment(params_.ba, setup, rec_);
  rec_->FilterPoints(params_.triangulation.max_reprojection_error_px, params_.triangulation.min_triangulation_angle_deg);

  const int num_points = static_cast<int>(rec_->NumPoints());
  LOG(INFO) << "Seed pair " << ia.name << " & " << ib.name << ": " << view_graph_.NumInliers(a, b) << " inliers, "
            << num_points << " points, median angle " << view_graph_.Pair(a, b)->median_triangulation_angle_deg << " deg";
  if (num_points < params_.seed_min_points)
  {
    ia.registered = ib.registered = false;
    rec_->ClearPoints();
    return false;
  }
  seed1_ = a;
  seed2_ = b;
  order_ = {a, b};
  return true;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

std::vector<IncrementalMapper::Candidate> IncrementalMapper::Candidates() const
{
  std::vector<Candidate> candidates;
  for (const Image &image : rec_->images)
  {
    if (image.registered)
      continue;
    Candidate c{image.id, {}};
    for (ImageId nb : view_graph_.Neighbors(image.id))
      if (rec_->images[nb].registered)
        c.registered_neighbors.push_back(nb);
    if (!c.registered_neighbors.empty())
      candidates.push_back(std::move(c));
  }
  // Best connected first; ties broken by the number of matches into the model.
  auto strength = [&](const Candidate &c) {
    int s = 0;
    for (ImageId nb : c.registered_neighbors)
      s += view_graph_.NumInliers(c.id, nb);
    return s;
  };
  std::stable_sort(candidates.begin(), candidates.end(), [&](const Candidate &x, const Candidate &y) {
    if (x.registered_neighbors.size() != y.registered_neighbors.size())
      return x.registered_neighbors.size() > y.registered_neighbors.size();
    return strength(x) > strength(y);
  });
  return candidates;
}

IncrementalMapper::Correspondences IncrementalMapper::Collect2D3D(ImageId id, const std::vector<ImageId> &refs) const
{
  // feature in `id` -> point, dropping features that reach different points and points reached twice.
  std::unordered_map<int, PointId> by_feature;
  std::unordered_map<PointId, int> point_uses;
  std::vector<int> ambiguous;
  for (ImageId ref : refs)
  {
    const Image &ref_image = rec_->images[ref];
    for (const FeatureMatch &m : view_graph_.Matches(id, ref))
    {
      const PointId pid = ref_image.point_ids[m.idx2];
      if (pid == kInvalidId)
        continue;
      auto it = by_feature.find(m.idx1);
      if (it == by_feature.end())
      {
        by_feature[m.idx1] = pid;
        ++point_uses[pid];
      }
      else if (it->second != pid)
        ambiguous.push_back(m.idx1);
    }
  }
  for (int f : ambiguous)
    by_feature.erase(f);

  Correspondences corr;
  for (const auto &kv : by_feature)
  {
    if (point_uses[kv.second] != 1)
      continue;
    corr.features.push_back(kv.first);
    corr.points.push_back(kv.second);
  }
  return corr;
}

bool IncrementalMapper::Resect(ImageId id, const Eigen::Matrix3d &R, const Correspondences &corr, Pose *pose, double *score,
                               std::vector<bool> *inliers)
{
  const Image &image = rec_->images[id];
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> normalized, pixels;
  for (size_t i = 0; i < corr.points.size(); ++i)
  {
    X.push_back(rec_->Point(corr.points[i]).X);
    normalized.push_back(image.features.normalized[corr.features[i]]);
    const cv::Point2f &px = image.features.keypoints[corr.features[i]].pt;
    pixels.emplace_back(px.x, px.y);
  }
  ResectionResult result;
  if (!ResectKnownRotation(rec_->CameraOf(id), R, X, normalized, pixels, params_.resection, &result))
    return false;
  *pose = result.pose;
  *score = result.score_px;
  *inliers = result.inliers;
  return true;
}

bool IncrementalMapper::TryRegisterMulti(const Candidate &cand, Pose *pose, double *score, Correspondences *corr, std::vector<bool> *inliers)
{
  // One rotation candidate per registered neighbour: R_k = R_ki * R_i.
  std::vector<Eigen::Matrix3d> rotations;
  for (ImageId ref : cand.registered_neighbors)
  {
    Eigen::Matrix3d R_ki;
    Eigen::Vector3d t_ki;
    view_graph_.RelativePose(ref, cand.id, &R_ki, &t_ki);
    rotations.push_back(R_ki * rec_->images[ref].pose.R);
  }
  RotationAveragingResult rot;
  if (!SingleRotationAveragingRansac(rotations, params_.rotation, &rot))
  {
    VLOG(1) << rec_->images[cand.id].name << ": rotation averaging failed";
    return false;
  }
  std::vector<ImageId> refs;
  for (size_t i = 0; i < rotations.size(); ++i)
    if (rot.inliers[i])
      refs.push_back(cand.registered_neighbors[i]);

  *corr = Collect2D3D(cand.id, refs);
  if (!Resect(cand.id, rot.R, *corr, pose, score, inliers))
    return false;
  VLOG(1) << rec_->images[cand.id].name << ": rotation " << rot.num_inliers << "/" << rotations.size() << " (score " << rot.score
          << "), position score " << *score << " px";
  return *score < params_.accept_score_px;
}

bool IncrementalMapper::TryRegisterSingle(ImageId id, ImageId ref, Pose *pose, double *score, Correspondences *corr,
                                          std::vector<bool> *inliers)
{
  Eigen::Matrix3d R_ki;
  Eigen::Vector3d t_ki;
  view_graph_.RelativePose(ref, id, &R_ki, &t_ki);
  *corr = Collect2D3D(id, {ref});
  return Resect(id, R_ki * rec_->images[ref].pose.R, *corr, pose, score, inliers);
}

bool IncrementalMapper::RegisterNextImage()
{
  // A newly registered image gives the failed images it overlaps with another chance.
  const ImageId just_added = order_.back();
  for (auto it = failed_.begin(); it != failed_.end();)
    it = view_graph_.Pair(*it, just_added) ? failed_.erase(it) : std::next(it);

  const std::vector<Candidate> candidates = Candidates();
  if (candidates.empty())
    return false;

  Pose pose;
  double score = 0.0;
  Correspondences corr;
  std::vector<bool> inliers;

  // Pass 1: images with >= 2 connections, rotation averaging + position.
  for (const Candidate &cand : candidates)
  {
    if (cand.registered_neighbors.size() < 2 || failed_.count(cand.id))
      continue;
    if (TryRegisterMulti(cand, &pose, &score, &corr, &inliers))
    {
      LOG(INFO) << "[" << rec_->NumRegistered() + 1 << "/" << rec_->images.size() << "] " << rec_->images[cand.id].name << ": "
                << cand.registered_neighbors.size() << " connections, score " << score << " px";
      AddImage(cand.id, pose, corr, inliers);
      return true;
    }
    failed_.insert(cand.id);
  }

  // Pass 2: single-connection fallback over every candidate, keep the best score.
  ImageId best_id = kInvalidId;
  Pose best_pose;
  double best_score = params_.accept_score_single_px;
  Correspondences best_corr;
  std::vector<bool> best_inliers;
  for (const Candidate &cand : candidates)
  {
    std::vector<ImageId> refs = cand.registered_neighbors;
    std::stable_sort(refs.begin(), refs.end(),
                     [&](ImageId x, ImageId y) { return view_graph_.NumInliers(cand.id, x) > view_graph_.NumInliers(cand.id, y); });
    for (ImageId ref : refs)
    {
      if (TryRegisterSingle(cand.id, ref, &pose, &score, &corr, &inliers) && score < best_score)
      {
        best_id = cand.id;
        best_pose = pose;
        best_score = score;
        best_corr = corr;
        best_inliers = inliers;
      }
    }
    if (best_id != kInvalidId && best_score < params_.accept_score_px)
      break; // good enough, no need to look further
  }
  if (best_id == kInvalidId)
    return false;

  LOG(INFO) << "[" << rec_->NumRegistered() + 1 << "/" << rec_->images.size() << "] " << rec_->images[best_id].name
            << ": single-connection fallback, score " << best_score << " px";
  if (best_score >= params_.accept_score_px)
    since_window_ba_ = params_.ba_interval; // poor registration: adjust right away
  AddImage(best_id, best_pose, best_corr, best_inliers);
  return true;
}

void IncrementalMapper::AddImage(ImageId id, const Pose &pose, const Correspondences &corr, const std::vector<bool> &inliers)
{
  Image &image = rec_->images[id];
  image.pose = pose;
  image.registered = true;
  order_.push_back(id);
  for (size_t i = 0; i < corr.points.size(); ++i)
    if (inliers[i] && rec_->HasPoint(corr.points[i]))
      rec_->AddObservation(corr.points[i], {id, corr.features[i]});
  TriangulateImage(id);

  ++since_window_ba_;
  ++since_global_ba_;
  if (params_.global_ba_interval > 0 && since_global_ba_ >= params_.global_ba_interval)
    GlobalBundleAdjustment();
  else if (since_window_ba_ >= params_.ba_interval)
    WindowBundleAdjustment();
}

int IncrementalMapper::TriangulateImage(ImageId id)
{
  const Camera &cam = rec_->CameraOf(id);
  int created = 0;
  for (ImageId ref : view_graph_.Neighbors(id))
  {
    if (!rec_->images[ref].registered)
      continue;
    const Camera &ref_cam = rec_->CameraOf(ref);
    for (const FeatureMatch &m : view_graph_.Matches(id, ref))
    {
      Image &image = rec_->images[id];
      Image &ref_image = rec_->images[ref];
      const PointId p_new = image.point_ids[m.idx1];
      const PointId p_ref = ref_image.point_ids[m.idx2];
      const double max_err = params_.triangulation.max_reprojection_error_px;
      if (p_new != kInvalidId && p_ref != kInvalidId)
        continue;
      if (p_ref != kInvalidId)
      {
        if (rec_->ObservationError(rec_->Point(p_ref), {id, m.idx1}) < max_err)
          rec_->AddObservation(p_ref, {id, m.idx1});
        continue;
      }
      if (p_new != kInvalidId)
      {
        if (rec_->ObservationError(rec_->Point(p_new), {ref, m.idx2}) < max_err)
          rec_->AddObservation(p_new, {ref, m.idx2});
        continue;
      }
      Eigen::Vector3d X;
      std::vector<bool> keep;
      if (TriangulateRobust({&cam, &ref_cam}, {image.pose, ref_image.pose},
                            {image.features.normalized[m.idx1], ref_image.features.normalized[m.idx2]}, params_.triangulation, &X, &keep) &&
          keep[0] && keep[1])
      {
        if (rec_->AddPoint(X, {{id, m.idx1}, {ref, m.idx2}}) != kInvalidId)
          ++created;
      }
    }
  }
  return created;
}

// ---------------------------------------------------------------------------
// Bundle adjustment
// ---------------------------------------------------------------------------

void IncrementalMapper::WindowBundleAdjustment()
{
  since_window_ba_ = 0;
  BundleAdjustmentSetup setup;
  const int n = static_cast<int>(order_.size());
  const int start = params_.ba_window > 0 ? std::max(0, n - params_.ba_window) : 0;
  setup.variable_images.assign(order_.begin() + start, order_.end());
  setup.gauge_image1 = seed1_;
  setup.gauge_image2 = seed2_;
  const BundleAdjustmentSummary summary = RunBundleAdjustment(params_.ba, setup, rec_);

  std::vector<PointId> touched;
  for (ImageId id : setup.variable_images)
    for (PointId pid : rec_->images[id].point_ids)
      if (pid != kInvalidId)
        touched.push_back(pid);
  std::sort(touched.begin(), touched.end());
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
  const int removed = rec_->FilterPoints(params_.triangulation.max_reprojection_error_px, params_.triangulation.min_triangulation_angle_deg, &touched);
  LOG(INFO) << "Window BA: " << summary.Brief() << ", filtered " << removed << " observations";
}

void IncrementalMapper::GlobalBundleAdjustment()
{
  since_global_ba_ = 0;
  since_window_ba_ = 0;
  BundleAdjustmentSetup setup;
  setup.variable_images = order_;
  setup.gauge_image1 = seed1_;
  setup.gauge_image2 = seed2_;
  const BundleAdjustmentSummary summary = RunBundleAdjustment(params_.ba, setup, rec_);
  const int removed = rec_->FilterPoints(params_.triangulation.max_reprojection_error_px, params_.triangulation.min_triangulation_angle_deg);
  LOG(INFO) << "Global BA: " << summary.Brief() << ", filtered " << removed << " observations";
}

} // namespace sfm
