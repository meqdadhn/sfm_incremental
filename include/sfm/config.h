/*******************************************************************************
* @file    config.h
* @brief   Pipeline configuration, loaded from a YAML file (see config/example.yaml).
*******************************************************************************/

#ifndef SFM_CONFIG_H_
#define SFM_CONFIG_H_

#include <map>
#include <string>
#include <vector>

#include "sfm/bundle_adjustment.h"
#include "sfm/features.h"
#include "sfm/incremental_mapper.h"
#include "sfm/matching.h"
#include "sfm/ortho.h"
#include "sfm/priors.h"
#include "sfm/triangulation.h"
#include "sfm/two_view.h"

namespace sfm
{
struct IoConfig
{
  std::string image_dir;
  std::vector<std::string> images; ///< optional explicit list, relative to image_dir; empty = every image in image_dir
  std::string output_dir = "output";
  std::string cache_dir;           ///< features / ROP cache, defaults to <output_dir>/cache
  bool use_cache = true;
};

struct FinalParams
{
  int min_track_length = 2;
  TriangulationParams triangulation;
  BundleAdjustmentParams ba;
  int rounds = 2; ///< BA + outlier filtering rounds
};

struct SfmConfig
{
  IoConfig io;
  std::map<CameraId, Camera> cameras;
  CameraId default_camera = 0;
  std::map<std::string, CameraId> image_cameras; ///< per-image camera overrides, by file name
  TrajectoryParams trajectory;

  SiftParams sift;
  MatchingParams matching;
  TwoViewParams two_view;
  IncrementalParams incremental;
  FinalParams final_ba;
  OrthoParams ortho;
};

/// Parses the YAML file. Relative paths are resolved against the file's directory.
SfmConfig LoadConfig(const std::string &path);

/// Reads K and distortion from an OpenCV FileStorage calibration file.
Camera LoadOpenCVCalibration(const std::string &path, CameraId id, const std::string &camera_matrix_key = "camera_matrix",
                             const std::string &distortion_key = "distortion_coefficients");

} // namespace sfm

#endif // SFM_CONFIG_H_
