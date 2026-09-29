#include "sfm/reconstruction.h"

#include <algorithm>

#include "sfm/triangulation.h"

namespace sfm
{
std::vector<ImageId> Reconstruction::RegisteredImages() const
{
  std::vector<ImageId> ids;
  for (const Image &image : images)
    if (image.registered)
      ids.push_back(image.id);
  return ids;
}

int Reconstruction::NumRegistered() const
{
  return static_cast<int>(std::count_if(images.begin(), images.end(), [](const Image &im) { return im.registered; }));
}

PointId Reconstruction::AddPoint(const Eigen::Vector3d &X, const std::vector<Observation> &observations)
{
  const PointId id = next_point_id_++;
  MapPoint &point = points_[id];
  point.X = X;
  for (const Observation &obs : observations)
    AddObservation(id, obs);
  if (point.observations.size() < 2)
  {
    DeletePoint(id);
    return kInvalidId;
  }
  return id;
}

bool Reconstruction::AddObservation(PointId id, const Observation &obs)
{
  MapPoint &point = points_.at(id);
  Image &image = images[obs.image_id];
  if (image.point_ids[obs.feature_idx] != kInvalidId)
    return false;
  for (const Observation &o : point.observations)
    if (o.image_id == obs.image_id)
      return false;
  point.observations.push_back(obs);
  image.point_ids[obs.feature_idx] = id;
  return true;
}

void Reconstruction::DeleteObservation(PointId id, ImageId image_id)
{
  MapPoint &point = points_.at(id);
  auto it = std::find_if(point.observations.begin(), point.observations.end(), [&](const Observation &o) { return o.image_id == image_id; });
  if (it == point.observations.end())
    return;
  images[it->image_id].point_ids[it->feature_idx] = kInvalidId;
  point.observations.erase(it);
  if (point.observations.size() < 2)
    DeletePoint(id);
}

void Reconstruction::DeletePoint(PointId id)
{
  auto it = points_.find(id);
  if (it == points_.end())
    return;
  for (const Observation &obs : it->second.observations)
    images[obs.image_id].point_ids[obs.feature_idx] = kInvalidId;
  points_.erase(it);
}

void Reconstruction::ClearPoints()
{
  points_.clear();
  for (Image &image : images)
    std::fill(image.point_ids.begin(), image.point_ids.end(), kInvalidId);
}

double Reconstruction::ObservationError(const MapPoint &point, const Observation &obs) const
{
  const Image &image = images[obs.image_id];
  return ReprojectionErrorPx(CameraOf(obs.image_id), image.pose, point.X, image.features.normalized[obs.feature_idx]);
}

int Reconstruction::FilterPoints(double max_error_px, double min_triangulation_angle_deg, const std::vector<PointId> *subset)
{
  std::vector<PointId> ids;
  if (subset)
    ids = *subset;
  else
    for (const auto &kv : points_)
      ids.push_back(kv.first);

  int removed = 0;
  for (PointId id : ids)
  {
    if (!HasPoint(id))
      continue;
    std::vector<ImageId> bad;
    for (const Observation &obs : points_.at(id).observations)
      if (ObservationError(points_.at(id), obs) > max_error_px)
        bad.push_back(obs.image_id);
    for (ImageId im : bad)
    {
      if (!HasPoint(id))
        break;
      DeleteObservation(id, im);
      ++removed;
    }
    if (!HasPoint(id))
      continue;
    std::vector<Eigen::Vector3d> centers;
    for (const Observation &obs : points_.at(id).observations)
      centers.push_back(images[obs.image_id].pose.Center());
    if (MaxTriangulationAngleDeg(centers, points_.at(id).X) < min_triangulation_angle_deg)
    {
      removed += static_cast<int>(points_.at(id).observations.size());
      DeletePoint(id);
    }
  }
  return removed;
}

double Reconstruction::MeanReprojectionError() const
{
  double sum = 0.0;
  size_t n = 0;
  for (const auto &kv : points_)
  {
    for (const Observation &obs : kv.second.observations)
    {
      sum += ObservationError(kv.second, obs);
      ++n;
    }
  }
  return n > 0 ? sum / n : 0.0;
}

size_t Reconstruction::NumObservations() const
{
  size_t n = 0;
  for (const auto &kv : points_)
    n += kv.second.observations.size();
  return n;
}

} // namespace sfm
