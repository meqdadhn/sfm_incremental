#include "sfm/resection.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

#include "cost_functions.h"
#include "sfm/triangulation.h"

namespace sfm
{
namespace
{
std::vector<double> Residuals(const Camera &camera, const Pose &pose, const std::vector<Eigen::Vector3d> &points,
                              const std::vector<Eigen::Vector2d> &normalized)
{
  std::vector<double> res(points.size());
  for (size_t i = 0; i < points.size(); ++i)
    res[i] = ReprojectionErrorPx(camera, pose, points[i], normalized[i]);
  return res;
}

double RmsOver(const std::vector<double> &res, const std::vector<bool> &mask)
{
  double sum = 0.0;
  int n = 0;
  for (size_t i = 0; i < res.size(); ++i)
  {
    if (!mask[i])
      continue;
    sum += res[i] * res[i];
    ++n;
  }
  return n > 0 ? std::sqrt(sum / n) : std::numeric_limits<double>::max();
}

int Count(const std::vector<bool> &mask) { return static_cast<int>(std::count(mask.begin(), mask.end(), true)); }
} // namespace

bool SolveTranslationKnownRotation(const Eigen::Matrix3d &R, const std::vector<Eigen::Vector3d> &points,
                                   const std::vector<Eigen::Vector2d> &normalized, const std::vector<bool> &mask, Eigen::Vector3d *t)
{
  const int n = Count(mask);
  if (n < 2)
    return false;
  Eigen::MatrixXd A(2 * n, 3);
  Eigen::VectorXd b(2 * n);
  int row = 0;
  for (size_t i = 0; i < points.size(); ++i)
  {
    if (!mask[i])
      continue;
    const Eigen::Vector3d RX = R * points[i];
    const double u = normalized[i].x();
    const double v = normalized[i].y();
    A.row(row) << 1.0, 0.0, -u;
    b(row++) = u * RX.z() - RX.x();
    A.row(row) << 0.0, 1.0, -v;
    b(row++) = v * RX.z() - RX.y();
  }
  *t = A.colPivHouseholderQr().solve(b);
  return t->allFinite();
}

bool ResectKnownRotation(const Camera &camera, const Eigen::Matrix3d &R, const std::vector<Eigen::Vector3d> &points,
                         const std::vector<Eigen::Vector2d> &normalized, const std::vector<Eigen::Vector2d> &pixels,
                         const ResectionParams &params, ResectionResult *result)
{
  const int n = static_cast<int>(points.size());
  if (n < params.min_correspondences)
    return false;

  Pose pose;
  pose.R = R;
  std::vector<bool> mask(n, true);
  std::vector<double> res;

  // Least trimmed squares: keep the best trim_ratio fraction until the RMS error is acceptable.
  for (int iter = 0; iter < 200; ++iter)
  {
    if (!SolveTranslationKnownRotation(R, points, normalized, mask, &pose.t))
      return false;
    res = Residuals(camera, pose, points, normalized);
    const int kept = Count(mask);
    if (RmsOver(res, mask) <= params.trim_trigger_px)
      break;
    const int keep = static_cast<int>(std::ceil(params.trim_ratio * kept));
    if (keep < params.min_correspondences || keep == kept)
      break;
    std::vector<int> order;
    for (int i = 0; i < n; ++i)
      if (mask[i])
        order.push_back(i);
    std::nth_element(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) { return res[a] < res[b]; });
    std::fill(mask.begin(), mask.end(), false);
    for (int k = 0; k < keep; ++k)
      mask[order[k]] = true;
  }

  // Final inlier set over all correspondences, then re-solve once.
  for (int pass = 0; pass < 2; ++pass)
  {
    for (int i = 0; i < n; ++i)
      mask[i] = res[i] < params.max_error_px;
    if (Count(mask) < params.min_correspondences)
      return false;
    if (!SolveTranslationKnownRotation(R, points, normalized, mask, &pose.t))
      return false;
    res = Residuals(camera, pose, points, normalized);
  }

  if (params.refine_pose)
  {
    std::vector<Eigen::Vector3d> in_pts;
    std::vector<Eigen::Vector2d> in_px;
    for (int i = 0; i < n; ++i)
    {
      if (res[i] < params.max_error_px)
      {
        in_pts.push_back(points[i]);
        in_px.push_back(pixels[i]);
      }
    }
    RefinePose(camera, in_pts, in_px, params.refine_loss_px, &pose);
    res = Residuals(camera, pose, points, normalized);
  }

  for (int i = 0; i < n; ++i)
    mask[i] = res[i] < params.max_error_px;
  result->pose = pose;
  result->inliers = mask;
  result->num_inliers = Count(mask);
  result->score_px = RmsOver(res, mask);
  return result->num_inliers >= params.min_correspondences;
}

void RefinePose(const Camera &camera, const std::vector<Eigen::Vector3d> &points, const std::vector<Eigen::Vector2d> &pixels,
                double loss_px, Pose *pose)
{
  if (points.size() < 3)
    return;
  PoseBlock block = PoseBlock::FromPose(*pose);
  std::array<double, 9> K = IntrinsicsBlock(camera);
  std::vector<std::array<double, 3>> X(points.size());

  ceres::Problem problem;
  for (size_t i = 0; i < points.size(); ++i)
  {
    X[i] = {points[i].x(), points[i].y(), points[i].z()};
    problem.AddResidualBlock(ReprojectionError::Create(pixels[i].x(), pixels[i].y()), new ceres::HuberLoss(loss_px), block.aa.data(),
                             block.center.data(), K.data(), X[i].data());
    problem.SetParameterBlockConstant(X[i].data());
  }
  problem.SetParameterBlockConstant(K.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 30;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (summary.IsSolutionUsable())
    *pose = block.ToPose();
}

} // namespace sfm
