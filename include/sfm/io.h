/*******************************************************************************
* @file    io.h
* @brief   Image listing, feature / ROP caches, and reconstruction export.
*
* Caches are binary and stamped with a hash of the parameters that produced
* them, so changing SIFT / matching / two-view settings invalidates them.
*******************************************************************************/

#ifndef SFM_IO_H_
#define SFM_IO_H_

#include <string>
#include <vector>

#include "sfm/reconstruction.h"
#include "sfm/view_graph.h"

namespace sfm
{
/// Image files (jpg, png, tif, bmp) in `dir`, sorted by name.
std::vector<std::string> ListImages(const std::string &dir);

bool LoadFeatures(const std::string &path, uint64_t hash, Features *features);
void SaveFeatures(const std::string &path, uint64_t hash, const Features &features);

bool LoadViewGraph(const std::string &path, uint64_t hash, int num_images, ViewGraph *view_graph);
void SaveViewGraph(const std::string &path, uint64_t hash, const ViewGraph &view_graph);

/// Averages the image colors of each point's observations.
void ColorizePoints(Reconstruction *reconstruction);

/// COLMAP text model (cameras.txt, images.txt, points3D.txt), viewable in the COLMAP GUI.
void WriteColmapText(const Reconstruction &reconstruction, const std::string &dir);
/// Colored point cloud plus camera centers (red).
void WritePly(const Reconstruction &reconstruction, const std::string &path);
/// One line per registered image: name, qw qx qy qz (R_cw), tx ty tz, camera center.
void WritePoses(const Reconstruction &reconstruction, const std::string &path);

} // namespace sfm

#endif // SFM_IO_H_
