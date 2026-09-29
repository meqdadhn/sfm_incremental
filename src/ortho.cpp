#include "sfm/ortho.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

#include <glog/logging.h>
#include <opencv2/flann.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "sfm/geometry.h"

namespace fs = std::filesystem;

namespace sfm
{
namespace
{
double Percentile(std::vector<double> v, double p)
{
  if (v.empty())
    return 0.0;
  const size_t k = std::min(v.size() - 1, static_cast<size_t>(std::lround(p / 100.0 * (v.size() - 1))));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

/// Native ground resolution: median over cameras of (median point depth / focal length).
double NativeGsd(const Reconstruction &rec)
{
  std::vector<double> footprints;
  for (const Image &image : rec.images)
  {
    if (!image.registered)
      continue;
    std::vector<double> depths;
    for (PointId pid : image.point_ids)
      if (pid != kInvalidId)
        depths.push_back(image.pose.Transform(rec.Point(pid).X).z());
    if (depths.empty())
      continue;
    footprints.push_back(Percentile(depths, 50.0) / rec.CameraOf(image.id).MeanFocal());
  }
  return Percentile(footprints, 50.0);
}

/// Bilinear DEM lookup; falls back to the valid corners when some are no-data.
bool SampleDem(const cv::Mat &dem, double u, double v, double *z)
{
  u = std::clamp(u, 0.0, dem.cols - 1.0);
  v = std::clamp(v, 0.0, dem.rows - 1.0);
  const int c0 = std::min(static_cast<int>(u), dem.cols - 2 < 0 ? 0 : dem.cols - 2);
  const int r0 = std::min(static_cast<int>(v), dem.rows - 2 < 0 ? 0 : dem.rows - 2);
  const int c1 = std::min(c0 + 1, dem.cols - 1);
  const int r1 = std::min(r0 + 1, dem.rows - 1);
  const double fu = u - c0, fv = v - r0;
  const double w[4] = {(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv};
  const float h[4] = {dem.at<float>(r0, c0), dem.at<float>(r0, c1), dem.at<float>(r1, c0), dem.at<float>(r1, c1)};
  double sum = 0.0, wsum = 0.0;
  for (int k = 0; k < 4; ++k)
  {
    if (std::isnan(h[k]))
      continue;
    sum += w[k] * h[k];
    wsum += w[k];
  }
  if (wsum <= 1e-9)
    return false;
  *z = sum / wsum;
  return true;
}

/// 3x3 median that ignores no-data (NaN) cells.
cv::Mat MedianFilterNan(const cv::Mat &dem)
{
  cv::Mat out = dem.clone();
#pragma omp parallel for schedule(static)
  for (int r = 0; r < dem.rows; ++r)
  {
    float vals[9];
    for (int c = 0; c < dem.cols; ++c)
    {
      if (std::isnan(dem.at<float>(r, c)))
        continue;
      int n = 0;
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc)
        {
          const int rr = r + dr, cc = c + dc;
          if (rr >= 0 && rr < dem.rows && cc >= 0 && cc < dem.cols && !std::isnan(dem.at<float>(rr, cc)))
            vals[n++] = dem.at<float>(rr, cc);
        }
      std::nth_element(vals, vals + n / 2, vals + n);
      out.at<float>(r, c) = vals[n / 2];
    }
  }
  return out;
}

cv::Vec3b SampleBilinear(const cv::Mat &img, float x, float y)
{
  const int x0 = std::clamp(static_cast<int>(x), 0, img.cols - 2);
  const int y0 = std::clamp(static_cast<int>(y), 0, img.rows - 2);
  const float fx = x - x0, fy = y - y0;
  const cv::Vec3b &a = img.at<cv::Vec3b>(y0, x0), &b = img.at<cv::Vec3b>(y0, x0 + 1);
  const cv::Vec3b &c = img.at<cv::Vec3b>(y0 + 1, x0), &d = img.at<cv::Vec3b>(y0 + 1, x0 + 1);
  cv::Vec3b out;
  for (int k = 0; k < 3; ++k)
    out[k] = cv::saturate_cast<uchar>((a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy);
  return out;
}
} // namespace

bool GenerateOrthophoto(const Reconstruction &rec, const OrthoParams &params, Orthophoto *ortho)
{
  const std::vector<ImageId> registered = rec.RegisteredImages();
  if (registered.empty() || rec.NumPoints() < 10)
    return false;

  // Projection direction check: map +Z should be close to the mean viewing direction.
  Eigen::Vector3d mean_view = Eigen::Vector3d::Zero();
  for (ImageId id : registered)
    mean_view += rec.images[id].pose.R.row(2).transpose(); // camera z axis in the map frame
  const double tilt = RadToDeg(std::acos(std::clamp(mean_view.normalized().z(), -1.0, 1.0)));
  LOG(INFO) << "Ortho: mean viewing direction is " << tilt << " deg from map Z";
  if (tilt > 30.0)
    LOG(WARNING) << "Ortho: cameras are far from looking along map Z (the seed camera's axis); the ortho will look oblique";

  // Extent from robust point percentiles.
  std::vector<double> xs, ys;
  std::vector<cv::Point2f> xy;
  std::vector<float> zs;
  for (const auto &kv : rec.Points())
  {
    xs.push_back(kv.second.X.x());
    ys.push_back(kv.second.X.y());
  }
  const double p = params.bounds_percentile;
  const double xmin = Percentile(xs, p), xmax = Percentile(xs, 100.0 - p);
  const double ymin = Percentile(ys, p), ymax = Percentile(ys, 100.0 - p);

  double gsd = params.gsd > 0 ? params.gsd : NativeGsd(rec);
  if (gsd <= 0)
    return false;
  gsd = std::max(gsd, std::max(xmax - xmin, ymax - ymin) / params.max_dimension);
  const int width = std::max(1, static_cast<int>(std::ceil((xmax - xmin) / gsd)));
  const int height = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / gsd)));
  ortho->x0 = xmin;
  ortho->y0 = ymin;
  ortho->gsd = gsd;
  LOG(INFO) << "Ortho: " << width << " x " << height << " px, gsd " << gsd << " map units";

  // 1. DEM from the sparse points (IDW over the k nearest, in coordinates relative to the corner).
  for (const auto &kv : rec.Points())
  {
    xy.emplace_back(static_cast<float>(kv.second.X.x() - xmin), static_cast<float>(kv.second.X.y() - ymin));
    zs.push_back(static_cast<float>(kv.second.X.z()));
  }
  const double cell = params.dem_cell_factor * gsd;
  ortho->dem_cell = cell;
  const int dem_w = std::max(2, static_cast<int>(std::ceil(width / params.dem_cell_factor)));
  const int dem_h = std::max(2, static_cast<int>(std::ceil(height / params.dem_cell_factor)));
  const cv::Mat features(static_cast<int>(xy.size()), 2, CV_32F, xy.data());
  cv::flann::Index kdtree(features, cv::flann::KDTreeIndexParams(1));
  cv::Mat queries(dem_w * dem_h, 2, CV_32F);
  for (int r = 0; r < dem_h; ++r)
  {
    for (int c = 0; c < dem_w; ++c)
    {
      queries.at<float>(r * dem_w + c, 0) = static_cast<float>((c + 0.5) * cell);
      queries.at<float>(r * dem_w + c, 1) = static_cast<float>((r + 0.5) * cell);
    }
  }
  const int k = std::max(1, std::min(params.dem_neighbors, static_cast<int>(xy.size())));
  cv::Mat indices, dists;
  kdtree.knnSearch(queries, indices, dists, k, cv::flann::SearchParams(64));
  ortho->dem.create(dem_h, dem_w, CV_32F);
  const float max_gap2 = static_cast<float>(std::pow(params.dem_max_gap_cells * cell, 2));
  for (int q = 0; q < dem_w * dem_h; ++q)
  {
    float &out = ortho->dem.at<float>(q / dem_w, q % dem_w);
    if (dists.at<float>(q, 0) > max_gap2)
    {
      out = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    double sum = 0.0, wsum = 0.0;
    for (int j = 0; j < k; ++j)
    {
      const double w = 1.0 / (std::sqrt(dists.at<float>(q, j)) + 1e-3 * cell);
      sum += w * zs[indices.at<int>(q, j)];
      wsum += w;
    }
    out = static_cast<float>(sum / wsum);
  }
  if (params.dem_median_filter)
    ortho->dem = MedianFilterNan(ortho->dem);

  // 2. Candidate cameras per DEM cell: the closest centers in XY.
  std::vector<Eigen::Vector2d> centers;
  for (ImageId id : registered)
  {
    const Eigen::Vector3d C = rec.images[id].pose.Center();
    centers.emplace_back(C.x() - xmin, C.y() - ymin);
  }
  const int n_cand = std::min<int>(params.candidate_images, static_cast<int>(registered.size()));
  std::vector<int> candidates(static_cast<size_t>(dem_w) * dem_h * n_cand);
#pragma omp parallel for schedule(static)
  for (int q = 0; q < dem_w * dem_h; ++q)
  {
    const Eigen::Vector2d P(((q % dem_w) + 0.5) * cell, ((q / dem_w) + 0.5) * cell);
    std::vector<int> order(centers.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + n_cand, order.end(),
                      [&](int a, int b) { return (centers[a] - P).squaredNorm() < (centers[b] - P).squaredNorm(); });
    std::copy(order.begin(), order.begin() + n_cand, candidates.begin() + static_cast<size_t>(q) * n_cand);
  }

  // 3. Per ortho pixel: height from the DEM, closest candidate camera that sees the ground point.
  cv::Mat assigned(height, width, CV_32S, cv::Scalar(-1)); // index into `registered`
  cv::Mat coords(height, width, CV_32FC2);
#pragma omp parallel for schedule(dynamic, 16)
  for (int r = 0; r < height; ++r)
  {
    std::vector<std::pair<double, int>> order(n_cand);
    for (int c = 0; c < width; ++c)
    {
      const double u = (c + 0.5) / params.dem_cell_factor - 0.5;
      const double v = (r + 0.5) / params.dem_cell_factor - 0.5;
      double z;
      if (!SampleDem(ortho->dem, u, v, &z))
        continue;
      const Eigen::Vector2d P((c + 0.5) * gsd, (r + 0.5) * gsd);
      const Eigen::Vector3d X(P.x() + xmin, P.y() + ymin, z);
      const int cell_c = std::clamp(static_cast<int>((c + 0.5) / params.dem_cell_factor), 0, dem_w - 1);
      const int cell_r = std::clamp(static_cast<int>((r + 0.5) / params.dem_cell_factor), 0, dem_h - 1);
      const int *cand = &candidates[(static_cast<size_t>(cell_r) * dem_w + cell_c) * n_cand];
      for (int j = 0; j < n_cand; ++j)
        order[j] = {(centers[cand[j]] - P).squaredNorm(), cand[j]};
      std::sort(order.begin(), order.end());
      for (const auto &o : order)
      {
        const Image &image = rec.images[registered[o.second]];
        const Camera &cam = rec.CameraOf(image.id);
        const Eigen::Vector3d Xc = image.pose.Transform(X);
        if (Xc.z() <= 0)
          continue;
        const Eigen::Vector2d px = cam.Project(Xc);
        const int b = params.image_border_px;
        if (px.x() < b || px.y() < b || px.x() >= cam.width - 1 - b || px.y() >= cam.height - 1 - b)
          continue;
        assigned.at<int>(r, c) = o.second;
        coords.at<cv::Vec2f>(r, c) = cv::Vec2f(static_cast<float>(px.x()), static_cast<float>(px.y()));
        break;
      }
    }
  }

  // 4. Fill from images, a batch at a time.
  ortho->image.create(height, width, CV_8UC4);
  ortho->image.setTo(cv::Scalar(0, 0, 0, 0));
  const int n = static_cast<int>(registered.size());
  const int batch = std::max(1, params.image_batch);
  for (int start = 0; start < n; start += batch)
  {
    const int end = std::min(n, start + batch);
    std::vector<cv::Mat> loaded(end - start);
#pragma omp parallel for schedule(dynamic)
    for (int i = start; i < end; ++i)
      loaded[i - start] = cv::imread(rec.images[registered[i]].path, cv::IMREAD_COLOR);
#pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < height; ++r)
    {
      for (int c = 0; c < width; ++c)
      {
        const int a = assigned.at<int>(r, c);
        if (a < start || a >= end || loaded[a - start].empty())
          continue;
        const cv::Vec2f px = coords.at<cv::Vec2f>(r, c);
        const cv::Vec3b bgr = SampleBilinear(loaded[a - start], px[0], px[1]);
        ortho->image.at<cv::Vec4b>(r, c) = cv::Vec4b(bgr[0], bgr[1], bgr[2], 255);
      }
    }
    LOG(INFO) << "Ortho: filled from images " << end << " / " << n;
  }
  return true;
}

void WriteOrthophoto(const Orthophoto &ortho, const Reconstruction &rec, const OrthoParams &params, const std::string &dir)
{
  fs::create_directories(dir);
  const fs::path out(dir);
  cv::imwrite((out / "ortho.png").string(), ortho.image);
  cv::imwrite((out / "dem.tiff").string(), ortho.dem);

  // Colour-mapped DEM preview (map Z grows along the viewing direction, so invert for "height").
  cv::Mat valid = ortho.dem == ortho.dem; // false for NaN
  double zmin = 0, zmax = 1;
  cv::minMaxLoc(ortho.dem, &zmin, &zmax, nullptr, nullptr, valid);
  cv::Mat norm;
  ortho.dem.convertTo(norm, CV_8U, -255.0 / std::max(zmax - zmin, 1e-9), 255.0 * zmax / std::max(zmax - zmin, 1e-9));
  cv::Mat preview;
  cv::applyColorMap(norm, preview, cv::COLORMAP_JET);
  preview.setTo(cv::Scalar(0, 0, 0), ~valid);
  cv::imwrite((out / "dem_preview.png").string(), preview);

  std::ofstream yaml(out / "ortho.yaml");
  yaml.precision(12);
  yaml << "# Orthophoto in the map frame: column -> map X, row -> map Y, projected along map Z.\n"
       << "# map X of pixel center (col, row) = x0 + (col + 0.5) * gsd, map Y = y0 + (row + 0.5) * gsd\n"
       << "width: " << ortho.image.cols << "\nheight: " << ortho.image.rows << "\nx0: " << ortho.x0 << "\ny0: " << ortho.y0
       << "\ngsd: " << ortho.gsd << "\ndem_cell: " << ortho.dem_cell << "\n";

  if (params.draw_trajectory)
  {
    cv::Mat traj;
    cv::cvtColor(ortho.image, traj, cv::COLOR_BGRA2BGR);
    std::vector<cv::Point> path;
    for (const Image &image : rec.images) // file order ~ flight order
    {
      if (!image.registered)
        continue;
      const Eigen::Vector3d C = image.pose.Center();
      const cv::Point2d px = ortho.MapToPixel(C.x(), C.y());
      path.emplace_back(static_cast<int>(std::lround(px.x)), static_cast<int>(std::lround(px.y)));
    }
    const int thickness = std::max(1, std::max(traj.cols, traj.rows) / 800);
    cv::polylines(traj, path, false, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
    for (const cv::Point &pt : path)
      cv::circle(traj, pt, 3 * thickness, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
    cv::imwrite((out / "ortho_trajectory.jpg").string(), traj);
  }
}

} // namespace sfm
