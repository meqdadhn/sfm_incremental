#include "sfm/geometry.h"

#include <algorithm>

#include <Eigen/SVD>

#include "sfm/types.h"

namespace sfm
{
double RotationAngleDeg(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2)
{
  const double c = std::clamp(0.5 * ((R1.transpose() * R2).trace() - 1.0), -1.0, 1.0);
  return RadToDeg(std::acos(c));
}

double TriangulationAngleDeg(const Eigen::Vector3d &C1, const Eigen::Vector3d &C2, const Eigen::Vector3d &X)
{
  const Eigen::Vector3d r1 = (X - C1).normalized();
  const Eigen::Vector3d r2 = (X - C2).normalized();
  return RadToDeg(std::acos(std::clamp(r1.dot(r2), -1.0, 1.0)));
}

Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d &M)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() > 0 ? 1.0 : -1.0;
  return svd.matrixU() * D * svd.matrixV().transpose();
}

Eigen::Vector2d Camera::Project(const Eigen::Vector3d &X_cam) const
{
  const double x = X_cam.x() / X_cam.z();
  const double y = X_cam.y() / X_cam.z();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + r2 * (dist[0] + r2 * (dist[1] + r2 * dist[4]));
  const double xd = x * radial + 2.0 * dist[2] * x * y + dist[3] * (r2 + 2.0 * x * x);
  const double yd = y * radial + dist[2] * (r2 + 2.0 * y * y) + 2.0 * dist[3] * x * y;
  return {fx * xd + cx, fy * yd + cy};
}

int Image::NumPoints() const
{
  return static_cast<int>(std::count_if(point_ids.begin(), point_ids.end(), [](PointId p) { return p != kInvalidId; }));
}

} // namespace sfm
