#include "sfm/bundle_adjustment.h"

#include <chrono>
#include <cmath>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <glog/logging.h>

#include "cost_functions.h"
#include "sfm/features.h"

namespace sfm
{
namespace
{
ceres::LossFunction *MakeLoss(const BundleAdjustmentParams &params)
{
  if (params.loss == "huber")
    return new ceres::HuberLoss(params.loss_scale_px);
  if (params.loss == "cauchy")
    return new ceres::CauchyLoss(params.loss_scale_px);
  return nullptr;
}

double RmsError(const Reconstruction &rec, const std::vector<PointId> &point_ids)
{
  double sum = 0.0;
  size_t n = 0;
  for (PointId id : point_ids)
  {
    const MapPoint &p = rec.Point(id);
    for (const Observation &obs : p.observations)
    {
      if (!rec.images[obs.image_id].registered)
        continue;
      const double e = rec.ObservationError(p, obs);
      sum += e * e;
      ++n;
    }
  }
  return n > 0 ? std::sqrt(sum / n) : 0.0;
}
} // namespace

std::string BundleAdjustmentSummary::Brief() const
{
  std::ostringstream ss;
  ss << num_images << " images, " << num_points << " points, " << num_residuals << " residuals, " << iterations << " iterations, rms "
     << initial_rms_px << " -> " << final_rms_px << " px, " << time_s << " s";
  return ss.str();
}

BundleAdjustmentSummary RunBundleAdjustment(const BundleAdjustmentParams &params, const BundleAdjustmentSetup &setup,
                                            Reconstruction *rec)
{
  const auto t0 = std::chrono::steady_clock::now();
  BundleAdjustmentSummary out;

  const std::unordered_set<ImageId> variable(setup.variable_images.begin(), setup.variable_images.end());
  std::unordered_set<PointId> point_set;
  for (ImageId id : setup.variable_images)
    for (PointId pid : rec->images[id].point_ids)
      if (pid != kInvalidId)
        point_set.insert(pid);
  const std::vector<PointId> point_ids(point_set.begin(), point_set.end());
  if (point_ids.empty())
    return out;
  out.initial_rms_px = RmsError(*rec, point_ids);

  std::unordered_map<ImageId, PoseBlock> poses;
  std::unordered_map<CameraId, std::array<double, 9>> intrinsics;
  std::unordered_map<PointId, std::array<double, 3>> points;
  poses.reserve(setup.variable_images.size() * 2);
  points.reserve(point_ids.size());

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  ceres::LossFunction *loss = MakeLoss(params);
  std::unordered_set<ImageId> constant_images;

  for (PointId pid : point_ids)
  {
    const MapPoint &mp = rec->Point(pid);
    auto &X = points[pid];
    X = {mp.X.x(), mp.X.y(), mp.X.z()};
    for (const Observation &obs : mp.observations)
    {
      const Image &image = rec->images[obs.image_id];
      if (!image.registered)
        continue;
      auto pose_it = poses.find(obs.image_id);
      if (pose_it == poses.end())
        pose_it = poses.emplace(obs.image_id, PoseBlock::FromPose(image.pose)).first;
      auto k_it = intrinsics.find(image.camera_id);
      if (k_it == intrinsics.end())
        k_it = intrinsics.emplace(image.camera_id, IntrinsicsBlock(rec->cameras.at(image.camera_id))).first;
      const cv::Point2f &px = image.features.keypoints[obs.feature_idx].pt;
      problem.AddResidualBlock(ReprojectionError::Create(px.x, px.y), loss, pose_it->second.aa.data(), pose_it->second.center.data(),
                               k_it->second.data(), X.data());
      ++out.num_residuals;
      if (!variable.count(obs.image_id))
        constant_images.insert(obs.image_id);
    }
  }

  for (ImageId id : constant_images)
  {
    problem.SetParameterBlockConstant(poses.at(id).aa.data());
    problem.SetParameterBlockConstant(poses.at(id).center.data());
  }

  // Gauge: 7 DoF need at least two fixed cameras, otherwise fix one pose + one center coordinate.
  auto freeze_one_coordinate = [&](ImageId fixed, ImageId other) {
    if (!poses.count(fixed) || !poses.count(other) || constant_images.count(other) || fixed == other)
      return;
    const auto &c1 = poses.at(fixed).center;
    const auto &c2 = poses.at(other).center;
    int axis = 0;
    double best = -1.0;
    for (int k = 0; k < 3; ++k)
    {
      if (std::abs(c2[k] - c1[k]) > best)
      {
        best = std::abs(c2[k] - c1[k]);
        axis = k;
      }
    }
    problem.SetParameterization(poses.at(other).center.data(), new ceres::SubsetParameterization(3, {axis}));
  };
  if (constant_images.empty() && poses.count(setup.gauge_image1))
  {
    problem.SetParameterBlockConstant(poses.at(setup.gauge_image1).aa.data());
    problem.SetParameterBlockConstant(poses.at(setup.gauge_image1).center.data());
    freeze_one_coordinate(setup.gauge_image1, setup.gauge_image2);
  }
  else if (constant_images.size() == 1)
  {
    // Scale is still free: freeze the dominant coordinate of the variable image farthest from the fixed one.
    const ImageId fixed = *constant_images.begin();
    ImageId farthest = kInvalidId;
    double best = -1.0;
    const Eigen::Vector3d Cf = rec->images[fixed].pose.Center();
    for (ImageId id : setup.variable_images)
    {
      const double d = (rec->images[id].pose.Center() - Cf).norm();
      if (poses.count(id) && d > best)
      {
        best = d;
        farthest = id;
      }
    }
    freeze_one_coordinate(fixed, farthest);
  }

  for (auto &kv : intrinsics)
  {
    std::vector<int> fixed;
    if (!params.refine_focal_length)
      fixed.insert(fixed.end(), {0, 1});
    if (!params.refine_principal_point)
      fixed.insert(fixed.end(), {2, 3});
    if (!params.refine_intrinsics)
      problem.SetParameterBlockConstant(kv.second.data());
    else if (!fixed.empty())
      problem.SetParameterization(kv.second.data(), new ceres::SubsetParameterization(9, fixed));
  }

  ceres::Solver::Options options;
  const int num_variable = static_cast<int>(poses.size() - constant_images.size());
  options.linear_solver_type = num_variable <= 50 ? ceres::DENSE_SCHUR : ceres::SPARSE_SCHUR;
  options.max_num_iterations = params.max_iterations;
  options.function_tolerance = params.function_tolerance;
  options.num_threads = params.num_threads > 0 ? params.num_threads : static_cast<int>(std::thread::hardware_concurrency());
  options.minimizer_progress_to_stdout = params.verbose;
  options.logging_type = params.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (params.verbose)
    LOG(INFO) << summary.BriefReport();

  out.success = summary.IsSolutionUsable();
  out.iterations = static_cast<int>(summary.iterations.size());
  out.num_images = num_variable;
  out.num_points = static_cast<int>(point_ids.size());
  if (out.success)
  {
    for (ImageId id : setup.variable_images)
      if (poses.count(id))
        rec->images[id].pose = poses.at(id).ToPose();
    for (const auto &kv : points)
      rec->Point(kv.first).X = Eigen::Vector3d(kv.second[0], kv.second[1], kv.second[2]);
    if (params.refine_intrinsics)
    {
      for (const auto &kv : intrinsics)
        SetIntrinsics(kv.second, &rec->cameras.at(kv.first));
      // Normalized coordinates depend on the intrinsics: refresh them before anything measures errors.
      for (Image &image : rec->images)
        if (intrinsics.count(image.camera_id))
          UndistortKeypoints(rec->cameras.at(image.camera_id), &image.features);
    }
  }
  out.final_rms_px = RmsError(*rec, point_ids);
  out.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

} // namespace sfm
