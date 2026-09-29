/*******************************************************************************
* @file    priors.h
* @brief   Preprocessing: intrinsics and trajectory priors from EXIF or a file.
*
* Intrinsics: fx = focal_mm / sensor_width_mm * width. The sensor width comes
* from the config, a built-in table of common drone cameras, or the EXIF focal
* plane resolution; the 35 mm equivalent focal length is the last resort.
*
* Trajectory: EXIF GPS converted to a local ENU frame (meters, origin at the
* first image), or a trajectory file. Priors drive trajectory-based pair
* selection: each image is only matched with its nearest images.
*******************************************************************************/

#ifndef SFM_PRIORS_H_
#define SFM_PRIORS_H_

#include <string>
#include <vector>

#include "sfm/exif.h"
#include "sfm/types.h"

namespace sfm
{
struct TrajectoryParams
{
  std::string source = "none"; ///< "none", "exif" (GPS) or "file"
  /// "file" source. Either the original format, one "omega phi kappa X Y Z" line per image in
  /// image order (angles in degrees), or "name X Y Z [...]" lines matched by image name.
  std::string file;
};

/// Sensor width in mm for known camera models, 0 if unknown.
double LookupSensorWidthMm(const std::string &make, const std::string &model);

/// Fills fx, fy, cx, cy (principal point at the center, no distortion) from EXIF.
/// `sensor_width_mm` > 0 overrides the lookup. `method` describes where the focal came from.
bool CameraFromExif(const ExifData &exif, int width, int height, double sensor_width_mm, Camera *camera, std::string *method);

/// Geodetic (WGS84, degrees / meters) to local east-north-up relative to an origin.
Eigen::Vector3d GeodeticToEnu(double lat, double lon, double alt, double lat0, double lon0, double alt0);

/// Sets Image::has_prior / prior_position. Returns the number of images with a prior.
int LoadTrajectoryPriors(const TrajectoryParams &params, std::vector<Image> *images);

} // namespace sfm

#endif // SFM_PRIORS_H_
