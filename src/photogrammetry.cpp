#include "sfm/photogrammetry.h"

#include <cmath>

namespace sfm
{
namespace
{
const Eigen::Matrix3d kFlip = Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal(); // OpenCV camera <-> photogrammetric camera
} // namespace

Eigen::Matrix3d RotationFromOPK(double omega, double phi, double kappa)
{
  return (Eigen::AngleAxisd(omega, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(kappa, Eigen::Vector3d::UnitZ()))
      .toRotationMatrix();
}

void OPKFromRotation(const Eigen::Matrix3d &R, double *omega, double *phi, double *kappa)
{
  *phi = std::asin(std::max(-1.0, std::min(1.0, R(0, 2))));
  *kappa = std::atan2(-R(0, 1), R(0, 0));
  *omega = std::atan2(-R(1, 2), R(2, 2));
}

Pose PoseFromOPK(double omega, double phi, double kappa, const Eigen::Vector3d &X0)
{
  // R_opk = R_cw^T * F  =>  R_cw = F * R_opk^T  (F is its own inverse)
  return Pose::FromCenter(kFlip * RotationFromOPK(omega, phi, kappa).transpose(), X0);
}

void OPKFromPose(const Pose &pose, double *omega, double *phi, double *kappa)
{
  OPKFromRotation(pose.R.transpose() * kFlip, omega, phi, kappa);
}

} // namespace sfm
