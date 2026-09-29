#include "sfm/config.h"

#include <filesystem>
#include <stdexcept>

#include <opencv2/core/persistence.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace sfm
{
namespace
{
template <typename T>
void Read(const YAML::Node &node, const char *key, T *value)
{
  if (node && node[key])
    *value = node[key].as<T>();
}

std::string Resolve(const fs::path &base, const std::string &p)
{
  if (p.empty())
    return p;
  const fs::path path(p);
  return path.is_absolute() ? path.string() : (base / path).lexically_normal().string();
}

void ReadTriangulation(const YAML::Node &n, TriangulationParams *p)
{
  Read(n, "max_reprojection_error_px", &p->max_reprojection_error_px);
  Read(n, "min_triangulation_angle_deg", &p->min_triangulation_angle_deg);
}

void ReadBundle(const YAML::Node &n, BundleAdjustmentParams *p)
{
  Read(n, "max_iterations", &p->max_iterations);
  Read(n, "loss", &p->loss);
  Read(n, "loss_scale_px", &p->loss_scale_px);
  Read(n, "refine_intrinsics", &p->refine_intrinsics);
  Read(n, "refine_focal_length", &p->refine_focal_length);
  Read(n, "refine_principal_point", &p->refine_principal_point);
  Read(n, "num_threads", &p->num_threads);
  Read(n, "function_tolerance", &p->function_tolerance);
  Read(n, "verbose", &p->verbose);
}

Camera ReadCamera(const YAML::Node &n, const fs::path &base, bool *from_exif)
{
  Camera cam;
  Read(n, "id", &cam.id);
  *from_exif = false;
  Read(n, "from_exif", from_exif);
  if (n["calibration_file"])
  {
    std::string k_key = "camera_matrix", d_key = "distortion_coefficients";
    Read(n, "camera_matrix_key", &k_key);
    Read(n, "distortion_key", &d_key);
    cam = LoadOpenCVCalibration(Resolve(base, n["calibration_file"].as<std::string>()), cam.id, k_key, d_key);
  }
  Read(n, "width", &cam.width);
  Read(n, "height", &cam.height);
  Read(n, "fx", &cam.fx);
  Read(n, "fy", &cam.fy);
  Read(n, "cx", &cam.cx);
  Read(n, "cy", &cam.cy);
  if (n["dist"])
  {
    const auto d = n["dist"].as<std::vector<double>>();
    if (d.size() > 5)
      throw std::runtime_error("dist supports up to 5 coefficients (k1, k2, p1, p2, k3)");
    cam.dist.fill(0.0);
    std::copy(d.begin(), d.end(), cam.dist.begin());
  }
  if (!*from_exif && (cam.fx <= 0 || cam.fy <= 0))
    throw std::runtime_error("Camera " + std::to_string(cam.id) + " needs fx, fy > 0 (or from_exif: true)");
  return cam;
}
} // namespace

Camera LoadOpenCVCalibration(const std::string &path, CameraId id, const std::string &camera_matrix_key, const std::string &distortion_key)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened())
    throw std::runtime_error("Cannot open calibration file " + path);
  Camera cam;
  cam.id = id;
  cv::Mat K, D;
  fs[camera_matrix_key] >> K;
  fs[distortion_key] >> D;
  if (!fs["image_width"].empty())
    fs["image_width"] >> cam.width;
  if (!fs["image_height"].empty())
    fs["image_height"] >> cam.height;
  if (K.empty())
    throw std::runtime_error("No '" + camera_matrix_key + "' in " + path);
  {
    K.convertTo(K, CV_64F);
    cam.fx = K.at<double>(0, 0);
    cam.fy = K.at<double>(1, 1);
    cam.cx = K.at<double>(0, 2);
    cam.cy = K.at<double>(1, 2);
  }
  if (!D.empty())
  {
    D.convertTo(D, CV_64F);
    for (int i = 0; i < std::min<int>(5, static_cast<int>(D.total())); ++i)
      cam.dist[i] = D.at<double>(i);
  }
  return cam;
}

SfmConfig LoadConfig(const std::string &path)
{
  const YAML::Node root = YAML::LoadFile(path);
  const fs::path base = fs::absolute(path).parent_path();
  SfmConfig cfg;

  const YAML::Node io = root["io"];
  if (!io || !io["image_dir"])
    throw std::runtime_error("Config needs io.image_dir");
  cfg.io.image_dir = Resolve(base, io["image_dir"].as<std::string>());
  Read(io, "images", &cfg.io.images);
  Read(io, "output_dir", &cfg.io.output_dir);
  cfg.io.output_dir = Resolve(base, cfg.io.output_dir);
  Read(io, "cache_dir", &cfg.io.cache_dir);
  cfg.io.cache_dir = cfg.io.cache_dir.empty() ? (fs::path(cfg.io.output_dir) / "cache").string() : Resolve(base, cfg.io.cache_dir);
  Read(io, "use_cache", &cfg.io.use_cache);

  if (!root["cameras"] || !root["cameras"].IsSequence() || root["cameras"].size() == 0)
    throw std::runtime_error("Config needs at least one entry in cameras");
  for (const auto &n : root["cameras"])
  {
    bool from_exif = false;
    Camera cam = ReadCamera(n, base, &from_exif);
    cfg.cameras[cam.id] = cam;
    if (from_exif)
    {
      double sensor_width = 0.0;
      Read(n, "sensor_width_mm", &sensor_width);
      cfg.exif_cameras[cam.id] = sensor_width;
    }
  }

  const YAML::Node trj = root["trajectory"];
  Read(trj, "source", &cfg.trajectory.source);
  Read(trj, "file", &cfg.trajectory.file);
  cfg.trajectory.file = Resolve(base, cfg.trajectory.file);
  cfg.default_camera = cfg.cameras.begin()->first;
  Read(root, "default_camera", &cfg.default_camera);
  Read(root, "image_cameras", &cfg.image_cameras);

  const YAML::Node f = root["features"];
  Read(f, "max_features", &cfg.sift.max_features);
  Read(f, "octave_layers", &cfg.sift.octave_layers);
  Read(f, "contrast_threshold", &cfg.sift.contrast_threshold);
  Read(f, "edge_threshold", &cfg.sift.edge_threshold);
  Read(f, "sigma", &cfg.sift.sigma);
  Read(f, "max_image_dim", &cfg.sift.max_image_dim);
  Read(f, "root_sift", &cfg.sift.root_sift);

  const YAML::Node m = root["matching"];
  Read(m, "mode", &cfg.matching.mode);
  Read(m, "sequential_overlap", &cfg.matching.sequential_overlap);
  Read(m, "search", &cfg.matching.search);
  Read(m, "knn", &cfg.matching.knn);
  Read(m, "radius", &cfg.matching.radius);
  Read(m, "max_neighbors", &cfg.matching.max_neighbors);
  Read(m, "ratio", &cfg.matching.ratio);
  Read(m, "cross_check", &cfg.matching.cross_check);
  Read(m, "flann_trees", &cfg.matching.flann_trees);
  Read(m, "flann_checks", &cfg.matching.flann_checks);
  Read(m, "min_matches", &cfg.matching.min_matches);

  const YAML::Node tv = root["two_view"];
  Read(tv, "ransac_threshold_px", &cfg.two_view.ransac_threshold_px);
  Read(tv, "confidence", &cfg.two_view.confidence);
  Read(tv, "min_inliers", &cfg.two_view.min_inliers);
  Read(tv, "refine", &cfg.two_view.refine);
  Read(tv, "refine_loss_px", &cfg.two_view.refine_loss_px);
  Read(tv, "homography_threshold_px", &cfg.two_view.homography_threshold_px);

  const YAML::Node inc = root["incremental"];
  IncrementalParams &ip = cfg.incremental;
  if (inc)
  {
    const YAML::Node seed = inc["seed"];
    Read(seed, "min_triangulation_angle_deg", &ip.seed_min_triangulation_angle_deg);
    Read(seed, "max_homography_ratio", &ip.seed_max_homography_ratio);
    Read(seed, "min_points", &ip.seed_min_points);
    Read(seed, "max_trials", &ip.seed_max_trials);

    const YAML::Node ra = inc["rotation_averaging"];
    Read(ra, "inlier_threshold_deg", &ip.rotation.inlier_threshold_deg);
    Read(ra, "max_iterations", &ip.rotation.max_iterations);
    Read(ra, "confidence", &ip.rotation.confidence);
    Read(ra, "min_inliers", &ip.rotation.min_inliers);

    const YAML::Node rs = inc["resection"];
    Read(rs, "min_correspondences", &ip.resection.min_correspondences);
    Read(rs, "trim_trigger_px", &ip.resection.trim_trigger_px);
    Read(rs, "trim_ratio", &ip.resection.trim_ratio);
    Read(rs, "max_error_px", &ip.resection.max_error_px);
    Read(rs, "refine_pose", &ip.resection.refine_pose);
    Read(rs, "refine_loss_px", &ip.resection.refine_loss_px);

    Read(inc, "accept_score_px", &ip.accept_score_px);
    Read(inc, "accept_score_single_px", &ip.accept_score_single_px);
    ReadTriangulation(inc["triangulation"], &ip.triangulation);

    const YAML::Node wba = inc["window_ba"];
    Read(wba, "window", &ip.ba_window);
    Read(wba, "interval", &ip.ba_interval);
    Read(wba, "global_interval", &ip.global_ba_interval);
    ReadBundle(wba, &ip.ba);
  }

  const YAML::Node fin = root["final"];
  Read(fin, "min_track_length", &cfg.final_ba.min_track_length);
  Read(fin, "rounds", &cfg.final_ba.rounds);
  if (fin)
  {
    ReadTriangulation(fin["triangulation"], &cfg.final_ba.triangulation);
    ReadBundle(fin["ba"], &cfg.final_ba.ba);
  }

  const YAML::Node o = root["ortho"];
  Read(o, "enabled", &cfg.ortho.enabled);
  Read(o, "gsd", &cfg.ortho.gsd);
  Read(o, "max_dimension", &cfg.ortho.max_dimension);
  Read(o, "bounds_percentile", &cfg.ortho.bounds_percentile);
  Read(o, "dem_cell_factor", &cfg.ortho.dem_cell_factor);
  Read(o, "dem_neighbors", &cfg.ortho.dem_neighbors);
  Read(o, "dem_max_gap_cells", &cfg.ortho.dem_max_gap_cells);
  Read(o, "dem_median_filter", &cfg.ortho.dem_median_filter);
  Read(o, "candidate_images", &cfg.ortho.candidate_images);
  Read(o, "image_border_px", &cfg.ortho.image_border_px);
  Read(o, "image_batch", &cfg.ortho.image_batch);
  Read(o, "draw_trajectory", &cfg.ortho.draw_trajectory);
  return cfg;
}

} // namespace sfm
