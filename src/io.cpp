#include "sfm/io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>

#include "sfm/geometry.h"
#include "sfm/photogrammetry.h"

namespace fs = std::filesystem;

namespace sfm
{
namespace
{
constexpr uint32_t kFeaturesMagic = 0x53464654; // "SFFT"
constexpr uint32_t kViewGraphMagic = 0x53465647; // "SFVG"

template <typename T>
void Put(std::ofstream &out, const T &v)
{
  out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <typename T>
bool Get(std::ifstream &in, T *v)
{
  return static_cast<bool>(in.read(reinterpret_cast<char *>(v), sizeof(T)));
}

void EnsureParentDir(const std::string &path)
{
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty())
    fs::create_directories(parent);
}
} // namespace

std::vector<std::string> ListImages(const std::string &dir)
{
  static const std::vector<std::string> kExt = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"};
  std::vector<std::string> names;
  for (const auto &entry : fs::directory_iterator(dir))
  {
    if (!entry.is_regular_file())
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (std::find(kExt.begin(), kExt.end(), ext) != kExt.end())
      names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

bool LoadFeatures(const std::string &path, uint64_t hash, Features *features)
{
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0;
  uint64_t file_hash = 0;
  int32_t n = 0, dim = 0;
  if (!in || !Get(in, &magic) || magic != kFeaturesMagic || !Get(in, &file_hash) || file_hash != hash || !Get(in, &n) || !Get(in, &dim))
    return false;
  features->keypoints.resize(n);
  for (auto &kp : features->keypoints)
  {
    float v[5];
    int32_t octave;
    if (!Get(in, &v) || !Get(in, &octave))
      return false;
    kp = cv::KeyPoint(v[0], v[1], v[2], v[3], v[4], octave);
  }
  features->descriptors.create(n, dim, CV_32F);
  if (n > 0 && !in.read(reinterpret_cast<char *>(features->descriptors.data), static_cast<std::streamsize>(n) * dim * sizeof(float)))
    return false;
  return true;
}

void SaveFeatures(const std::string &path, uint64_t hash, const Features &features)
{
  EnsureParentDir(path);
  std::ofstream out(path, std::ios::binary);
  CV_Assert(features.descriptors.type() == CV_32F && features.descriptors.isContinuous());
  Put(out, kFeaturesMagic);
  Put(out, hash);
  Put(out, static_cast<int32_t>(features.keypoints.size()));
  Put(out, static_cast<int32_t>(features.descriptors.cols));
  for (const auto &kp : features.keypoints)
  {
    const float v[5] = {kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response};
    Put(out, v);
    Put(out, static_cast<int32_t>(kp.octave));
  }
  out.write(reinterpret_cast<const char *>(features.descriptors.data),
            static_cast<std::streamsize>(features.descriptors.total() * sizeof(float)));
}

bool LoadViewGraph(const std::string &path, uint64_t hash, int num_images, ViewGraph *view_graph)
{
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0;
  uint64_t file_hash = 0;
  int32_t n_images = 0, n_pairs = 0;
  if (!in || !Get(in, &magic) || magic != kViewGraphMagic || !Get(in, &file_hash) || file_hash != hash || !Get(in, &n_images) ||
      n_images != num_images || !Get(in, &n_pairs))
    return false;
  view_graph->Reset(num_images);
  for (int p = 0; p < n_pairs; ++p)
  {
    TwoViewGeometry g;
    int32_t i1, i2, raw, n_in;
    double R[9], t[3];
    if (!Get(in, &i1) || !Get(in, &i2) || !Get(in, &raw) || !Get(in, &R) || !Get(in, &t) || !Get(in, &g.median_triangulation_angle_deg) ||
        !Get(in, &g.homography_inlier_ratio) || !Get(in, &n_in))
      return false;
    g.image1 = i1;
    g.image2 = i2;
    g.num_raw_matches = raw;
    g.R = Eigen::Map<Eigen::Matrix3d>(R);
    g.t = Eigen::Map<Eigen::Vector3d>(t);
    g.inliers.resize(n_in);
    for (auto &m : g.inliers)
    {
      int32_t a, b;
      if (!Get(in, &a) || !Get(in, &b))
        return false;
      m = {a, b};
    }
    view_graph->AddPair(std::move(g));
  }
  return true;
}

void SaveViewGraph(const std::string &path, uint64_t hash, const ViewGraph &view_graph)
{
  EnsureParentDir(path);
  std::ofstream out(path, std::ios::binary);
  Put(out, kViewGraphMagic);
  Put(out, hash);
  Put(out, static_cast<int32_t>(view_graph.NumImages()));
  Put(out, static_cast<int32_t>(view_graph.Pairs().size()));
  for (const TwoViewGeometry &g : view_graph.Pairs())
  {
    Put(out, static_cast<int32_t>(g.image1));
    Put(out, static_cast<int32_t>(g.image2));
    Put(out, static_cast<int32_t>(g.num_raw_matches));
    out.write(reinterpret_cast<const char *>(g.R.data()), 9 * sizeof(double));
    out.write(reinterpret_cast<const char *>(g.t.data()), 3 * sizeof(double));
    Put(out, g.median_triangulation_angle_deg);
    Put(out, g.homography_inlier_ratio);
    Put(out, static_cast<int32_t>(g.inliers.size()));
    for (const FeatureMatch &m : g.inliers)
    {
      Put(out, static_cast<int32_t>(m.idx1));
      Put(out, static_cast<int32_t>(m.idx2));
    }
  }
}

void ColorizePoints(Reconstruction *rec)
{
  std::unordered_map<PointId, cv::Vec3d> sums;
  std::unordered_map<PointId, int> counts;
  for (const Image &image : rec->images)
  {
    if (!image.registered)
      continue;
    const cv::Mat bgr = cv::imread(image.path, cv::IMREAD_COLOR);
    if (bgr.empty())
      continue;
    for (size_t f = 0; f < image.point_ids.size(); ++f)
    {
      const PointId pid = image.point_ids[f];
      if (pid == kInvalidId)
        continue;
      const cv::Point2f &pt = image.features.keypoints[f].pt;
      const int x = std::clamp(static_cast<int>(std::lround(pt.x)), 0, bgr.cols - 1);
      const int y = std::clamp(static_cast<int>(std::lround(pt.y)), 0, bgr.rows - 1);
      const cv::Vec3b c = bgr.at<cv::Vec3b>(y, x);
      sums[pid] += cv::Vec3d(c[0], c[1], c[2]);
      ++counts[pid];
    }
  }
  for (const auto &kv : sums)
  {
    const cv::Vec3d avg = kv.second / counts[kv.first];
    rec->Point(kv.first).color = cv::Vec3b(cv::saturate_cast<uchar>(avg[0]), cv::saturate_cast<uchar>(avg[1]), cv::saturate_cast<uchar>(avg[2]));
  }
}

void WriteColmapText(const Reconstruction &rec, const std::string &dir)
{
  fs::create_directories(dir);
  std::ofstream cams(fs::path(dir) / "cameras.txt");
  cams << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" << std::setprecision(12);
  for (const auto &kv : rec.cameras)
  {
    const Camera &c = kv.second;
    // FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
    cams << c.id + 1 << " FULL_OPENCV " << c.width << " " << c.height << " " << c.fx << " " << c.fy << " " << c.cx << " " << c.cy << " "
         << c.dist[0] << " " << c.dist[1] << " " << c.dist[2] << " " << c.dist[3] << " " << c.dist[4] << " 0 0 0\n";
  }

  std::ofstream imgs(fs::path(dir) / "images.txt");
  imgs << "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n" << std::setprecision(12);
  for (const Image &image : rec.images)
  {
    if (!image.registered)
      continue;
    const Eigen::Quaterniond q(image.pose.R);
    const Eigen::Vector3d &t = image.pose.t;
    imgs << image.id + 1 << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << t.x() << " " << t.y() << " " << t.z() << " "
         << image.camera_id + 1 << " " << image.name << "\n";
    bool first = true;
    for (size_t f = 0; f < image.point_ids.size(); ++f)
    {
      if (image.point_ids[f] == kInvalidId)
        continue;
      // COLMAP puts the pixel center at 0.5.
      imgs << (first ? "" : " ") << image.features.keypoints[f].pt.x + 0.5 << " " << image.features.keypoints[f].pt.y + 0.5 << " "
           << image.point_ids[f] + 1;
      first = false;
    }
    imgs << "\n";
  }

  // POINT2D_IDX is the index among the image's points written in images.txt.
  std::vector<std::vector<int>> point2d_index(rec.images.size());
  for (const Image &image : rec.images)
  {
    point2d_index[image.id].assign(image.point_ids.size(), -1);
    int k = 0;
    for (size_t f = 0; f < image.point_ids.size(); ++f)
      if (image.point_ids[f] != kInvalidId)
        point2d_index[image.id][f] = k++;
  }

  std::ofstream pts(fs::path(dir) / "points3D.txt");
  pts << "# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" << std::setprecision(12);
  for (const auto &kv : rec.Points())
  {
    const MapPoint &p = kv.second;
    double err = 0.0;
    for (const Observation &obs : p.observations)
      err += rec.ObservationError(p, obs);
    err /= p.observations.size();
    pts << kv.first + 1 << " " << p.X.x() << " " << p.X.y() << " " << p.X.z() << " " << int(p.color[2]) << " " << int(p.color[1]) << " "
        << int(p.color[0]) << " " << err;
    for (const Observation &obs : p.observations)
      pts << " " << obs.image_id + 1 << " " << point2d_index[obs.image_id][obs.feature_idx];
    pts << "\n";
  }
}

void WritePly(const Reconstruction &rec, const std::string &path)
{
  EnsureParentDir(path);
  const std::vector<ImageId> registered = rec.RegisteredImages();
  std::ofstream out(path);
  out << "ply\nformat ascii 1.0\nelement vertex " << rec.NumPoints() + registered.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  for (const auto &kv : rec.Points())
  {
    const MapPoint &p = kv.second;
    out << p.X.x() << " " << p.X.y() << " " << p.X.z() << " " << int(p.color[2]) << " " << int(p.color[1]) << " " << int(p.color[0]) << "\n";
  }
  for (ImageId id : registered)
  {
    const Eigen::Vector3d C = rec.images[id].pose.Center();
    out << C.x() << " " << C.y() << " " << C.z() << " 255 0 0\n";
  }
}

void WritePoses(const Reconstruction &rec, const std::string &path)
{
  EnsureParentDir(path);
  std::ofstream out(path);
  out << "# name qw qx qy qz tx ty tz cx cy cz   (x_cam = R * X + t, C = camera center)\n" << std::setprecision(12);
  for (const Image &image : rec.images)
  {
    if (!image.registered)
      continue;
    const Eigen::Quaterniond q(image.pose.R);
    const Eigen::Vector3d C = image.pose.Center();
    out << image.name << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << image.pose.t.transpose() << " "
        << C.transpose() << "\n";
  }
}

void WriteEopsOPK(const Reconstruction &rec, const std::string &path)
{
  EnsureParentDir(path);
  std::ofstream out(path);
  out << "# name omega phi kappa [deg] X0 Y0 Z0   (R = Rx(omega)*Ry(phi)*Rz(kappa), camera -> map; photogrammetric camera frame)\n"
      << std::setprecision(12);
  for (const Image &image : rec.images)
  {
    if (!image.registered)
      continue;
    double o, p, k;
    OPKFromPose(image.pose, &o, &p, &k);
    const Eigen::Vector3d C = image.pose.Center();
    out << image.name << "\t" << RadToDeg(o) << "\t" << RadToDeg(p) << "\t" << RadToDeg(k) << "\t" << C.x() << "\t" << C.y() << "\t" << C.z() << "\n";
  }
}

} // namespace sfm
