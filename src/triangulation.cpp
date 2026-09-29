#include "sfm/triangulation.h"

#include <limits>

#include <Eigen/SVD>

#include "sfm/geometry.h"

namespace sfm
{
bool TriangulateDLT(const std::vector<Pose> &poses, const std::vector<Eigen::Vector2d> &normalized, Eigen::Vector3d *X)
{
  const int n = static_cast<int>(poses.size());
  if (n < 2)
    return false;
  Eigen::MatrixXd A(2 * n, 4);
  for (int i = 0; i < n; ++i)
  {
    Eigen::Matrix<double, 3, 4> P;
    P << poses[i].R, poses[i].t;
    A.row(2 * i) = normalized[i].x() * P.row(2) - P.row(0);
    A.row(2 * i + 1) = normalized[i].y() * P.row(2) - P.row(1);
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d Xh = svd.matrixV().col(3);
  if (std::abs(Xh(3)) < 1e-12)
    return false;
  *X = Xh.head<3>() / Xh(3);
  return X->allFinite();
}

double MaxTriangulationAngleDeg(const std::vector<Eigen::Vector3d> &centers, const Eigen::Vector3d &X)
{
  double best = 0.0;
  for (size_t i = 0; i < centers.size(); ++i)
    for (size_t j = i + 1; j < centers.size(); ++j)
      best = std::max(best, TriangulationAngleDeg(centers[i], centers[j], X));
  return best;
}

double ReprojectionErrorPx(const Camera &camera, const Pose &pose, const Eigen::Vector3d &X, const Eigen::Vector2d &normalized)
{
  Eigen::Vector2d uv;
  if (!ProjectNormalized(pose.Transform(X), &uv))
    return std::numeric_limits<double>::max();
  const Eigen::Vector2d d = uv - normalized;
  return std::hypot(d.x() * camera.fx, d.y() * camera.fy);
}

bool TriangulateRobust(const std::vector<const Camera *> &cameras, const std::vector<Pose> &poses,
                       const std::vector<Eigen::Vector2d> &normalized, const TriangulationParams &params, Eigen::Vector3d *X,
                       std::vector<bool> *keep)
{
  const int n = static_cast<int>(poses.size());
  keep->assign(n, true);
  int num_kept = n;
  while (num_kept >= 2)
  {
    std::vector<Pose> p;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> centers;
    for (int i = 0; i < n; ++i)
    {
      if (!(*keep)[i])
        continue;
      p.push_back(poses[i]);
      x.push_back(normalized[i]);
      centers.push_back(poses[i].Center());
    }
    if (!TriangulateDLT(p, x, X))
      return false;

    int worst = -1;
    double worst_err = params.max_reprojection_error_px;
    for (int i = 0; i < n; ++i)
    {
      if (!(*keep)[i])
        continue;
      const double err = ReprojectionErrorPx(*cameras[i], poses[i], *X, normalized[i]);
      if (err > worst_err)
      {
        worst_err = err;
        worst = i;
      }
    }
    if (worst < 0)
      return MaxTriangulationAngleDeg(centers, *X) >= params.min_triangulation_angle_deg;
    (*keep)[worst] = false;
    --num_kept;
  }
  return false;
}

} // namespace sfm
