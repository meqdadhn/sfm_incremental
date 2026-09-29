/*******************************************************************************
* @file    pipeline.h
* @brief   End-to-end SfM:
*            1. read images + cameras
*            2. SIFT extraction
*            3. matching + ROP (essential matrix) per pair
*            4. incremental extrinsic estimation with window BA
*            5. feature tracking + final BA
*******************************************************************************/

#ifndef SFM_PIPELINE_H_
#define SFM_PIPELINE_H_

#include "sfm/config.h"
#include "sfm/reconstruction.h"
#include "sfm/view_graph.h"

namespace sfm
{
class Pipeline
{
public:
  explicit Pipeline(SfmConfig config);

  bool Run();

  void LoadImages();
  void ExtractFeatures();
  void MatchAndEstimateRops();
  bool RunIncremental();
  void FinalTrackingAndBundle();
  void Export();

  const Reconstruction &GetReconstruction() const { return rec_; }
  Reconstruction &GetReconstruction() { return rec_; }
  const ViewGraph &GetViewGraph() const { return view_graph_; }

private:
  SfmConfig config_;
  Reconstruction rec_;
  ViewGraph view_graph_;
  ImageId seed1_ = kInvalidId;
  ImageId seed2_ = kInvalidId;
};

} // namespace sfm

#endif // SFM_PIPELINE_H_
