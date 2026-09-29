/*******************************************************************************
* @file    geometry.h
* @brief   Small geometric helpers used across modules.
*******************************************************************************/

#ifndef SFM_GEOMETRY_H_
#define SFM_GEOMETRY_H_

#include <cmath>

#include <Eigen/Core>

namespace sfm
{
constexpr double kPi = 3.14159265358979323846;

inline double DegToRad(double deg) { return deg * kPi / 180.0; }
inline double RadToDeg(double rad) { return rad * 180.0 / kPi; }

inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

/// Geodesic distance between two rotations, in degrees.
double RotationAngleDeg(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2);

/// Angle between the viewing rays from two camera centers to X, in degrees.
double TriangulationAngleDeg(const Eigen::Vector3d &C1, const Eigen::Vector3d &C2, const Eigen::Vector3d &X);

/// Projects X_cam onto the normalized image plane. Returns false if behind the camera.
inline bool ProjectNormalized(const Eigen::Vector3d &X_cam, Eigen::Vector2d *uv)
{
  if (X_cam.z() <= 1e-12)
    return false;
  *uv = X_cam.head<2>() / X_cam.z();
  return true;
}

/// Closest rotation (Frobenius norm) to an arbitrary 3x3 matrix.
Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d &M);

} // namespace sfm

#endif // SFM_GEOMETRY_H_
