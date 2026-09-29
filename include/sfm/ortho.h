/*******************************************************************************
* @file    ortho.h
* @brief   Orthophoto generation in the map frame (after Generate_Ortho).
*
* The ortho plane is the XY plane of the map and the projection direction is
* map Z, which is the viewing direction of the seed camera, i.e. roughly
* straight down for nadir drone imagery. No georeferencing is needed; ortho
* columns follow map X and rows follow map Y.
*
*  1. DEM: the sparse points are gridded (cell = dem_cell_factor * gsd) with
*     inverse-distance weighting of the k nearest points (k = 1 reproduces the
*     original nearest-point lookup), then a 3x3 median removes spikes; cells
*     far from any point are no-data.
*  2. For every ortho pixel, Z is interpolated from the DEM and the ground
*     point is projected into the camera whose center is closest in XY. If it
*     falls outside that image, the next closest camera is tried.
*  3. Images are loaded in batches and the assigned pixels are sampled
*     bilinearly.
*******************************************************************************/

#ifndef SFM_ORTHO_H_
#define SFM_ORTHO_H_

#include <string>

#include <opencv2/core.hpp>

#include "sfm/reconstruction.h"

namespace sfm
{
struct OrthoParams
{
  bool enabled = false;
  double gsd = 0.0;                ///< map units per ortho pixel, 0 = native (median camera footprint)
  int max_dimension = 8000;        ///< gsd is increased if the ortho would be larger than this
  double bounds_percentile = 1.0;  ///< ortho extent = [p, 100 - p] percentiles of the points in X and Y
  double dem_cell_factor = 8.0;    ///< DEM cell size in ortho pixels
  int dem_neighbors = 8;           ///< IDW neighbours, 1 = nearest point
  double dem_max_gap_cells = 10.0; ///< DEM cells farther than this from any point are no-data
  bool dem_median_filter = true;   ///< 3x3 median on the DEM, removes spikes from outlier points
  int candidate_images = 8;        ///< closest cameras considered per DEM cell
  int image_border_px = 5;
  int image_batch = 32;            ///< images held in memory at once
  bool draw_trajectory = true;
};

struct Orthophoto
{
  cv::Mat image; ///< CV_8UC4 (BGRA), alpha = 0 where no image covers the pixel
  cv::Mat dem;   ///< CV_32F map Z per DEM cell, NaN where no-data
  double x0 = 0.0, y0 = 0.0; ///< map XY of the ortho's top-left corner
  double gsd = 0.0;
  double dem_cell = 0.0;

  /// Map XY of the center of ortho pixel (col, row).
  cv::Point2d PixelToMap(double col, double row) const { return {x0 + (col + 0.5) * gsd, y0 + (row + 0.5) * gsd}; }
  cv::Point2d MapToPixel(double x, double y) const { return {(x - x0) / gsd - 0.5, (y - y0) / gsd - 0.5}; }
};

bool GenerateOrthophoto(const Reconstruction &reconstruction, const OrthoParams &params, Orthophoto *ortho);

/// ortho.png, dem.tiff (float, NaN = no-data), dem_preview.png, ortho.yaml (frame info)
/// and ortho_trajectory.jpg (camera centers drawn on top) if enabled.
void WriteOrthophoto(const Orthophoto &ortho, const Reconstruction &reconstruction, const OrthoParams &params, const std::string &dir);

} // namespace sfm

#endif // SFM_ORTHO_H_
