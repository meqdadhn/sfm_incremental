#include "sfm/tracks.h"

#include <numeric>
#include <unordered_map>

namespace sfm
{
namespace
{
class UnionFind
{
public:
  explicit UnionFind(size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), 0); }
  size_t Find(size_t x)
  {
    while (parent_[x] != x)
    {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }
  void Union(size_t a, size_t b)
  {
    a = Find(a);
    b = Find(b);
    if (a != b)
      parent_[std::max(a, b)] = std::min(a, b);
  }

private:
  std::vector<size_t> parent_;
};
} // namespace

std::vector<Track> BuildTracks(const ViewGraph &view_graph, const std::vector<Image> &images, const std::vector<bool> &use_image,
                               int min_track_length)
{
  std::vector<size_t> offset(images.size() + 1, 0);
  for (size_t i = 0; i < images.size(); ++i)
    offset[i + 1] = offset[i] + images[i].features.keypoints.size();

  UnionFind uf(offset.back());
  for (const TwoViewGeometry &g : view_graph.Pairs())
  {
    if (!use_image[g.image1] || !use_image[g.image2])
      continue;
    for (const FeatureMatch &m : g.inliers)
      uf.Union(offset[g.image1] + m.idx1, offset[g.image2] + m.idx2);
  }

  // Group nodes that take part in at least one match.
  std::unordered_map<size_t, std::vector<Observation>> components;
  std::vector<bool> touched(offset.back(), false);
  for (const TwoViewGeometry &g : view_graph.Pairs())
  {
    if (!use_image[g.image1] || !use_image[g.image2])
      continue;
    for (const FeatureMatch &m : g.inliers)
    {
      const size_t n1 = offset[g.image1] + m.idx1;
      const size_t n2 = offset[g.image2] + m.idx2;
      if (!touched[n1])
      {
        touched[n1] = true;
        components[uf.Find(n1)].push_back({g.image1, m.idx1});
      }
      if (!touched[n2])
      {
        touched[n2] = true;
        components[uf.Find(n2)].push_back({g.image2, m.idx2});
      }
    }
  }

  std::vector<Track> tracks;
  tracks.reserve(components.size());
  for (auto &kv : components)
  {
    std::unordered_map<ImageId, int> count;
    for (const Observation &obs : kv.second)
      ++count[obs.image_id];
    Track track;
    for (const Observation &obs : kv.second)
      if (count[obs.image_id] == 1)
        track.observations.push_back(obs);
    if (static_cast<int>(track.observations.size()) >= min_track_length)
      tracks.push_back(std::move(track));
  }
  return tracks;
}

int TriangulateTracks(const std::vector<Track> &tracks, const TriangulationParams &params, Reconstruction *rec)
{
  rec->ClearPoints();
  std::vector<Eigen::Vector3d> X(tracks.size());
  std::vector<std::vector<bool>> keep(tracks.size());
  std::vector<char> ok(tracks.size(), 0);

#pragma omp parallel for schedule(dynamic, 256)
  for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
  {
    std::vector<const Camera *> cams;
    std::vector<Pose> poses;
    std::vector<Eigen::Vector2d> obs;
    for (const Observation &o : tracks[i].observations)
    {
      const Image &image = rec->images[o.image_id];
      cams.push_back(&rec->cameras.at(image.camera_id));
      poses.push_back(image.pose);
      obs.push_back(image.features.normalized[o.feature_idx]);
    }
    ok[i] = TriangulateRobust(cams, poses, obs, params, &X[i], &keep[i]);
  }

  int created = 0;
  for (size_t i = 0; i < tracks.size(); ++i)
  {
    if (!ok[i])
      continue;
    std::vector<Observation> observations;
    for (size_t k = 0; k < tracks[i].observations.size(); ++k)
      if (keep[i][k])
        observations.push_back(tracks[i].observations[k]);
    if (rec->AddPoint(X[i], observations) != kInvalidId)
      ++created;
  }
  return created;
}

} // namespace sfm
