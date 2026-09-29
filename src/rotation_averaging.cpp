#include "sfm/rotation_averaging.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "sfm/geometry.h"

namespace sfm
{
namespace
{
/// Accumulates Horn's 4x4 matrix for the correspondences a -> b.
void AccumulateHorn(const Eigen::Vector3d &a, const Eigen::Vector3d &b, Eigen::Matrix4d *N)
{
  const double Sxx = a.x() * b.x(), Sxy = a.x() * b.y(), Sxz = a.x() * b.z();
  const double Syx = a.y() * b.x(), Syy = a.y() * b.y(), Syz = a.y() * b.z();
  const double Szx = a.z() * b.x(), Szy = a.z() * b.y(), Szz = a.z() * b.z();
  Eigen::Matrix4d M;
  M << Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
       Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
       Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
       Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz;
  *N += M;
}

double ChordalScore(const Eigen::Matrix3d &R, const std::vector<Eigen::Matrix3d> &candidates, const std::vector<bool> &inliers)
{
  // Columns of R are R * n_j for the unit axes, so the residual is the Frobenius difference.
  double sum = 0.0;
  int count = 0;
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    if (!inliers[i])
      continue;
    sum += (R - candidates[i]).squaredNorm();
    ++count;
  }
  return count > 0 ? std::sqrt(sum / (9.0 * count)) : 0.0;
}

int CountInliers(const Eigen::Matrix3d &R, const std::vector<Eigen::Matrix3d> &candidates, double threshold_deg, std::vector<bool> *inliers)
{
  inliers->assign(candidates.size(), false);
  int count = 0;
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    if (RotationAngleDeg(R, candidates[i]) <= threshold_deg)
    {
      (*inliers)[i] = true;
      ++count;
    }
  }
  return count;
}
} // namespace

Eigen::Matrix3d AverageRotationsQuaternion(const std::vector<Eigen::Matrix3d> &candidates, const std::vector<bool> &mask)
{
  Eigen::Matrix4d N = Eigen::Matrix4d::Zero();
  const Eigen::Matrix3d axes = Eigen::Matrix3d::Identity();
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    if (!mask.empty() && !mask[i])
      continue;
    for (int j = 0; j < 3; ++j)
      AccumulateHorn(axes.col(j), candidates[i] * axes.col(j), &N); // n_j -> R_c * n_j
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(N);
  const Eigen::Vector4d q = solver.eigenvectors().col(3); // eigenvalues ascending
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).normalized().toRotationMatrix();
}

bool SingleRotationAveragingRansac(const std::vector<Eigen::Matrix3d> &candidates, const RotationAveragingParams &params,
                                   RotationAveragingResult *result)
{
  const int n = static_cast<int>(candidates.size());
  if (n < 2)
    return false;

  // All pairs when there are few candidates (typical), random pairs otherwise.
  std::vector<std::pair<int, int>> samples;
  const int num_pairs = n * (n - 1) / 2;
  if (num_pairs <= params.max_iterations)
  {
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        samples.emplace_back(i, j);
  }
  else
  {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int k = 0; k < params.max_iterations; ++k)
    {
      int i = dist(rng), j = dist(rng);
      while (j == i)
        j = dist(rng);
      samples.emplace_back(i, j);
    }
  }

  std::vector<bool> best_inliers, inliers;
  int best_count = 0;
  int max_iterations = static_cast<int>(samples.size());
  for (int k = 0; k < max_iterations; ++k)
  {
    std::vector<bool> mask(n, false);
    mask[samples[k].first] = mask[samples[k].second] = true;
    const Eigen::Matrix3d R = AverageRotationsQuaternion(candidates, mask);
    const int count = CountInliers(R, candidates, params.inlier_threshold_deg, &inliers);
    if (count > best_count)
    {
      best_count = count;
      best_inliers = inliers;
      if (num_pairs > params.max_iterations)
      {
        // Adaptive stopping for random sampling, as in the original implementation.
        const double eps = static_cast<double>(count) / n;
        const double denom = std::log(1.0 - eps * eps);
        if (denom < 0)
          max_iterations = std::min(max_iterations, static_cast<int>(std::ceil(std::log(1.0 - params.confidence) / denom)));
      }
    }
  }
  if (best_count < params.min_inliers)
    return false;

  // Refit on all inliers and re-evaluate the consensus once.
  Eigen::Matrix3d R = AverageRotationsQuaternion(candidates, best_inliers);
  if (CountInliers(R, candidates, params.inlier_threshold_deg, &inliers) >= best_count)
  {
    best_inliers = inliers;
    R = AverageRotationsQuaternion(candidates, best_inliers);
  }

  result->R = R;
  result->inliers = best_inliers;
  result->num_inliers = static_cast<int>(std::count(best_inliers.begin(), best_inliers.end(), true));
  result->score = ChordalScore(R, candidates, best_inliers);
  return result->num_inliers >= params.min_inliers;
}

} // namespace sfm
