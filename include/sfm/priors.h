/*******************************************************************************
* @file    priors.h
* @brief   Position priors from a trajectory file.
*
* Priors are produced by the preprocessing scripts (tools/preprocess), e.g. EXIF
* GPS converted to local ENU meters, or come from any other trajectory source.
* They drive trajectory-based pair selection (matching.mode: trajectory).
*******************************************************************************/

#ifndef SFM_PRIORS_H_
#define SFM_PRIORS_H_

#include <string>
#include <vector>

#include "sfm/types.h"

namespace sfm
{
struct TrajectoryParams
{
  std::string source = "none"; ///< "none" or "file"
  /// Either the original format, one "omega phi kappa X Y Z" line per image in image order
  /// (angles in degrees), or "name X Y Z [...]" lines matched by image name ('#' starts a comment).
  std::string file;
};

/// Sets Image::has_prior / prior_position. Returns the number of images with a prior.
int LoadTrajectoryPriors(const TrajectoryParams &params, std::vector<Image> *images);

} // namespace sfm

#endif // SFM_PRIORS_H_
