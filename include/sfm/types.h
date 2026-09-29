/*******************************************************************************
* @file    types.h
* @brief   Core data types shared by every stage of the pipeline.
*
* Conventions
*   - Pose is world-to-camera: x_cam = R * X_world + t, camera center C = -R^T t.
*   - Camera frame is the OpenCV one: x right, y down, z forward.
*   - Relative pose of a pair (i, j) maps camera i to camera j:
*       x_j = R_ji * x_i + t_ji,  |t_ji| = 1.
*******************************************************************************/

#ifndef SFM_TYPES_H_
#define SFM_TYPES_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace sfm
{
using ImageId = int;
using CameraId = int;
using PointId = int64_t;

constexpr int kInvalidId = -1;

/// Pinhole camera with the OpenCV 5-parameter distortion model (k1, k2, p1, p2, k3).
struct Camera
{
  CameraId id = 0;
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 5> dist = {0.0, 0.0, 0.0, 0.0, 0.0};

  cv::Matx33d K() const { return cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1); }
  cv::Mat DistCoeffs() const { return cv::Mat(1, 5, CV_64F, const_cast<double *>(dist.data())).clone(); }
  double MeanFocal() const { return 0.5 * (fx + fy); }

  /// Projects a point in camera coordinates to distorted pixel coordinates.
  Eigen::Vector2d Project(const Eigen::Vector3d &X_cam) const;
};

/// World-to-camera rigid transform.
struct Pose
{
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Eigen::Vector3d Center() const { return -R.transpose() * t; }
  Eigen::Vector3d Transform(const Eigen::Vector3d &X) const { return R * X + t; }
  static Pose FromCenter(const Eigen::Matrix3d &R, const Eigen::Vector3d &C) { return {R, -R * C}; }
};

struct FeatureMatch
{
  int idx1 = kInvalidId;
  int idx2 = kInvalidId;
};

/// Keypoints of one image. Descriptors are released once matching is done.
struct Features
{
  std::vector<cv::KeyPoint> keypoints;   ///< pixel coordinates in the full-resolution image
  cv::Mat descriptors;                   ///< N x 128, CV_32F
  std::vector<Eigen::Vector2d> normalized; ///< undistorted, normalized image coordinates
};

/// Verified relative orientation (ROP) of an image pair, image1 < image2.
struct TwoViewGeometry
{
  ImageId image1 = kInvalidId;
  ImageId image2 = kInvalidId;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); ///< x2 = R * x1 + t
  Eigen::Vector3d t = Eigen::Vector3d::Zero();     ///< unit length
  std::vector<FeatureMatch> inliers;
  int num_raw_matches = 0;
  double median_triangulation_angle_deg = 0.0;
  double homography_inlier_ratio = 0.0; ///< H inliers / E inliers; close to 1 means planar or pure rotation
};

struct Observation
{
  ImageId image_id = kInvalidId;
  int feature_idx = kInvalidId;
};

struct MapPoint
{
  Eigen::Vector3d X = Eigen::Vector3d::Zero();
  std::vector<Observation> observations;
  cv::Vec3b color = cv::Vec3b(0, 0, 0); ///< BGR
};

struct Image
{
  ImageId id = kInvalidId;
  std::string name; ///< file name, used as a key in caches and exports
  std::string path;
  CameraId camera_id = 0;
  Features features;

  bool has_prior = false;           ///< position prior from GPS / trajectory file (preprocessing)
  Eigen::Vector3d prior_position = Eigen::Vector3d::Zero();
  bool has_prior_rotation = false;  ///< attitude prior (original omega-phi-kappa trajectory format)
  Eigen::Matrix3d prior_rotation = Eigen::Matrix3d::Identity(); ///< as Pose::R (world -> OpenCV camera)

  bool registered = false;
  Pose pose;
  std::vector<PointId> point_ids; ///< per keypoint, kInvalidId if not part of a map point

  int NumPoints() const;
};

} // namespace sfm

#endif // SFM_TYPES_H_
