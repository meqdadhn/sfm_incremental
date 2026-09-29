#include "sfm/priors.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "sfm/geometry.h"

namespace sfm
{
namespace
{
std::string Lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool IsNumber(const std::string &s)
{
  char *end = nullptr;
  std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0';
}
} // namespace

double LookupSensorWidthMm(const std::string &make, const std::string &model)
{
  // Sensor widths of common drone cameras (manufacturer specs).
  static const std::map<std::string, double> kSensors = {
      {"dji fc300s", 6.17},   // Phantom 3 Professional / Advanced
      {"dji fc300x", 6.17},   // Phantom 3 Professional 4K
      {"dji fc300c", 6.17},   // Phantom 3 Standard
      {"dji fc330", 6.17},    // Phantom 4
      {"dji fc220", 6.17},    // Mavic Pro
      {"dji fc6310", 13.2},   // Phantom 4 Pro / Advanced
      {"dji fc6310s", 13.2},  // Phantom 4 Pro V2
      {"dji fc6310r", 13.2},  // Phantom 4 RTK
      {"hasselblad l1d-20c", 13.2}, // Mavic 2 Pro
      {"dji fc6510", 13.2},   // Zenmuse X4S
      {"dji fc6520", 17.3},   // Zenmuse X5S
  };
  const auto it = kSensors.find(Lower(make + " " + model));
  return it == kSensors.end() ? 0.0 : it->second;
}

bool CameraFromExif(const ExifData &exif, int width, int height, double sensor_width_mm, Camera *camera, std::string *method)
{
  if (width <= 0 || height <= 0)
    return false;
  double f_px = 0.0;
  if (exif.focal_mm > 0)
  {
    double sw = sensor_width_mm > 0 ? sensor_width_mm : LookupSensorWidthMm(exif.make, exif.model);
    if (sw > 0)
    {
      f_px = exif.focal_mm / sw * width;
      *method = (sensor_width_mm > 0 ? "configured" : "known") + std::string(" sensor width ") + std::to_string(sw) + " mm";
    }
    else if (exif.focal_plane_x_res > 0 && (exif.focal_plane_unit == 2 || exif.focal_plane_unit == 3 || exif.focal_plane_unit == 4))
    {
      const double unit_mm = exif.focal_plane_unit == 2 ? 25.4 : exif.focal_plane_unit == 3 ? 10.0 : 1.0;
      f_px = exif.focal_mm * exif.focal_plane_x_res / unit_mm;
      *method = "EXIF focal plane resolution";
    }
  }
  if (f_px <= 0 && exif.focal_35mm > 0)
  {
    // 35 mm equivalent: 36 mm film width mapped to the long image side (approximate).
    f_px = exif.focal_35mm / 36.0 * std::max(width, height);
    *method = "35 mm equivalent focal length (approximate)";
  }
  if (f_px <= 0)
    return false;
  camera->width = width;
  camera->height = height;
  camera->fx = camera->fy = f_px;
  camera->cx = width / 2.0;
  camera->cy = height / 2.0;
  camera->dist.fill(0.0);
  return true;
}

Eigen::Vector3d GeodeticToEnu(double lat, double lon, double alt, double lat0, double lon0, double alt0)
{
  constexpr double a = 6378137.0;           // WGS84
  constexpr double e2 = 6.69437999014e-3;
  auto ecef = [&](double la, double lo, double h) {
    const double sla = std::sin(DegToRad(la)), cla = std::cos(DegToRad(la));
    const double slo = std::sin(DegToRad(lo)), clo = std::cos(DegToRad(lo));
    const double N = a / std::sqrt(1.0 - e2 * sla * sla);
    return Eigen::Vector3d((N + h) * cla * clo, (N + h) * cla * slo, (N * (1.0 - e2) + h) * sla);
  };
  const Eigen::Vector3d d = ecef(lat, lon, alt) - ecef(lat0, lon0, alt0);
  const double sla = std::sin(DegToRad(lat0)), cla = std::cos(DegToRad(lat0));
  const double slo = std::sin(DegToRad(lon0)), clo = std::cos(DegToRad(lon0));
  Eigen::Matrix3d R;
  R << -slo, clo, 0, -sla * clo, -sla * slo, cla, cla * clo, cla * slo, sla;
  return R * d;
}

int LoadTrajectoryPriors(const TrajectoryParams &params, std::vector<Image> *images)
{
  for (Image &image : *images)
    image.has_prior = false;
  if (params.source == "none")
    return 0;

  if (params.source == "exif")
  {
    bool have_origin = false;
    double lat0 = 0, lon0 = 0, alt0 = 0;
    for (Image &image : *images)
    {
      ExifData exif;
      if (!ReadExif(image.path, &exif) || !exif.has_gps)
        continue;
      if (!have_origin)
      {
        lat0 = exif.latitude;
        lon0 = exif.longitude;
        alt0 = exif.altitude;
        have_origin = true;
      }
      image.prior_position = GeodeticToEnu(exif.latitude, exif.longitude, exif.altitude, lat0, lon0, alt0);
      image.has_prior = true;
    }
  }
  else if (params.source == "file")
  {
    std::ifstream in(params.file);
    if (!in)
      throw std::runtime_error("Cannot open trajectory file " + params.file);
    std::map<std::string, ImageId> by_name;
    for (const Image &image : *images)
      by_name[image.name] = image.id;
    std::string line;
    int row = 0;
    while (std::getline(in, line))
    {
      std::istringstream ss(line);
      std::vector<std::string> tok;
      for (std::string t; ss >> t;)
        tok.push_back(t);
      if (tok.empty() || tok[0][0] == '#')
        continue;
      if (IsNumber(tok[0]) && tok.size() >= 6) // original format: omega phi kappa X Y Z, image order
      {
        if (row < static_cast<int>(images->size()))
        {
          (*images)[row].prior_position = Eigen::Vector3d(std::stod(tok[3]), std::stod(tok[4]), std::stod(tok[5]));
          (*images)[row].has_prior = true;
        }
        ++row;
      }
      else if (tok.size() >= 4 && by_name.count(tok[0])) // name X Y Z ...
      {
        Image &image = (*images)[by_name[tok[0]]];
        image.prior_position = Eigen::Vector3d(std::stod(tok[1]), std::stod(tok[2]), std::stod(tok[3]));
        image.has_prior = true;
      }
    }
  }
  else
  {
    throw std::invalid_argument("Unknown trajectory source: " + params.source);
  }
  return static_cast<int>(std::count_if(images->begin(), images->end(), [](const Image &im) { return im.has_prior; }));
}

} // namespace sfm
