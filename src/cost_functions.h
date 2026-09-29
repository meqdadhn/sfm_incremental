/*******************************************************************************
* @file    cost_functions.h
* @brief   Ceres residuals and parameter-block conversions (internal header).
*
* Pose block: angle-axis of R_cw (3) + camera center C (3), so x_cam = R (X - C).
* Parameterizing by the center lets the gauge fix the scale on one coordinate.
* Intrinsics block: fx, fy, cx, cy, k1, k2, p1, p2, k3.
*******************************************************************************/

#ifndef SFM_COST_FUNCTIONS_H_
#define SFM_COST_FUNCTIONS_H_

#include <array>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "sfm/types.h"

namespace sfm
{
struct PoseBlock
{
  std::array<double, 3> aa;
  std::array<double, 3> center;

  static PoseBlock FromPose(const Pose &pose)
  {
    PoseBlock b;
    ceres::RotationMatrixToAngleAxis(pose.R.data(), b.aa.data()); // Eigen is column-major, as Ceres expects
    const Eigen::Vector3d C = pose.Center();
    b.center = {C.x(), C.y(), C.z()};
    return b;
  }

  Pose ToPose() const
  {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(aa.data(), R.data());
    return Pose::FromCenter(R, Eigen::Vector3d(center[0], center[1], center[2]));
  }
};

inline std::array<double, 9> IntrinsicsBlock(const Camera &c)
{
  return {c.fx, c.fy, c.cx, c.cy, c.dist[0], c.dist[1], c.dist[2], c.dist[3], c.dist[4]};
}

inline void SetIntrinsics(const std::array<double, 9> &k, Camera *c)
{
  c->fx = k[0];
  c->fy = k[1];
  c->cx = k[2];
  c->cy = k[3];
  for (int i = 0; i < 5; ++i)
    c->dist[i] = k[4 + i];
}

/// Pixel reprojection error with the OpenCV distortion model.
struct ReprojectionError
{
  ReprojectionError(double u, double v) : u_(u), v_(v) {}

  template <typename T>
  bool operator()(const T *aa, const T *C, const T *K, const T *X, T *residual) const
  {
    const T d[3] = {X[0] - C[0], X[1] - C[1], X[2] - C[2]};
    T p[3];
    ceres::AngleAxisRotatePoint(aa, d, p);
    const T x = p[0] / p[2];
    const T y = p[1] / p[2];
    const T r2 = x * x + y * y;
    const T radial = T(1) + r2 * (K[4] + r2 * (K[5] + r2 * K[8]));
    const T xd = x * radial + T(2) * K[6] * x * y + K[7] * (r2 + T(2) * x * x);
    const T yd = y * radial + K[6] * (r2 + T(2) * y * y) + T(2) * K[7] * x * y;
    residual[0] = K[0] * xd + K[2] - T(u_);
    residual[1] = K[1] * yd + K[3] - T(v_);
    return true;
  }

  static ceres::CostFunction *Create(double u, double v)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 9, 3>(new ReprojectionError(u, v));
  }

  double u_, v_;
};

} // namespace sfm

#endif // SFM_COST_FUNCTIONS_H_
