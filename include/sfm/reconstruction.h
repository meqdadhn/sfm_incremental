/*******************************************************************************
* @file    reconstruction.h
* @brief   Cameras, images with poses, and the 3D points with their observations.
*******************************************************************************/

#ifndef SFM_RECONSTRUCTION_H_
#define SFM_RECONSTRUCTION_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "sfm/types.h"

namespace sfm
{
class Reconstruction
{
public:
  std::map<CameraId, Camera> cameras;
  std::vector<Image> images;

  const std::unordered_map<PointId, MapPoint> &Points() const { return points_; }
  MapPoint &Point(PointId id) { return points_.at(id); }
  const MapPoint &Point(PointId id) const { return points_.at(id); }
  bool HasPoint(PointId id) const { return points_.count(id) > 0; }
  size_t NumPoints() const { return points_.size(); }

  const Camera &CameraOf(ImageId id) const { return cameras.at(images[id].camera_id); }
  std::vector<ImageId> RegisteredImages() const;
  int NumRegistered() const;

  /// Creates a point from >= 2 observations whose features are not yet assigned.
  PointId AddPoint(const Eigen::Vector3d &X, const std::vector<Observation> &observations);
  /// Adds an observation if its feature is free and the point has none in that image.
  bool AddObservation(PointId id, const Observation &obs);
  /// Removes one observation; deletes the point if fewer than 2 remain.
  void DeleteObservation(PointId id, ImageId image_id);
  void DeletePoint(PointId id);
  void ClearPoints();

  /// Reprojection error (pixels, undistorted) of one observation of a point.
  double ObservationError(const MapPoint &point, const Observation &obs) const;

  /// Removes observations above max_error_px, then points with < 2 observations or a small
  /// triangulation angle. Restricted to `subset` if given. Returns the number of observations removed.
  int FilterPoints(double max_error_px, double min_triangulation_angle_deg, const std::vector<PointId> *subset = nullptr);

  double MeanReprojectionError() const;
  size_t NumObservations() const;

private:
  std::unordered_map<PointId, MapPoint> points_;
  PointId next_point_id_ = 0;
};

} // namespace sfm

#endif // SFM_RECONSTRUCTION_H_
