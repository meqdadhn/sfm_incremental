/*******************************************************************************
* @file    photogrammetry.h
* @brief   Conversions between the original photogrammetric EOP convention and Pose.
*
* Original convention (SfM.cpp rotation_Mat / Find_Rotation):
*   - R = Rx(omega) * Ry(phi) * Rz(kappa)   (NOT the common yaw-pitch-roll Rz*Ry*Rx)
*   - R rotates camera -> mapping frame:     X - X0 = R * x_p
*   - photogrammetric camera frame x_p:      x right, y up, looking along -z;
*                                            an image point is (x, y, -c)
* This code base (Pose):
*   - x_cam = R_cw * (X - C), OpenCV camera frame (x right, y down, looking along +z)
* Relation, with F = diag(1, -1, -1) (x_p = F * x_cam):
*   R_opk = R_cw^T * F,   X0 = C
*******************************************************************************/

#ifndef SFM_PHOTOGRAMMETRY_H_
#define SFM_PHOTOGRAMMETRY_H_

#include "sfm/types.h"

namespace sfm
{
/// Rx(omega) * Ry(phi) * Rz(kappa), angles in radians.
Eigen::Matrix3d RotationFromOPK(double omega, double phi, double kappa);

/// Inverse of RotationFromOPK (same formulas as the original Find_Rotation), radians.
void OPKFromRotation(const Eigen::Matrix3d &R, double *omega, double *phi, double *kappa);

/// Pose from original EOPs: omega/phi/kappa in radians, perspective center X0.
Pose PoseFromOPK(double omega, double phi, double kappa, const Eigen::Vector3d &X0);

/// Original EOPs (radians) of a pose; the perspective center is pose.Center().
void OPKFromPose(const Pose &pose, double *omega, double *phi, double *kappa);

} // namespace sfm

#endif // SFM_PHOTOGRAMMETRY_H_
