/*******************************************************************************
* @file    tracks.h
* @brief   Multi-view feature tracks from pairwise ROP inlier matches.
*******************************************************************************/

#ifndef SFM_TRACKS_H_
#define SFM_TRACKS_H_

#include <vector>

#include "sfm/reconstruction.h"
#include "sfm/triangulation.h"
#include "sfm/view_graph.h"

namespace sfm
{
struct Track
{
  std::vector<Observation> observations;
};

/// Connected components of the match graph over images flagged in `use_image`.
/// Images seen more than once in a component are dropped from that track (ambiguous).
std::vector<Track> BuildTracks(const ViewGraph &view_graph, const std::vector<Image> &images, const std::vector<bool> &use_image,
                               int min_track_length);

/// Replaces all points of the reconstruction by the triangulated tracks. Returns the number of points created.
int TriangulateTracks(const std::vector<Track> &tracks, const TriangulationParams &params, Reconstruction *reconstruction);

} // namespace sfm

#endif // SFM_TRACKS_H_
