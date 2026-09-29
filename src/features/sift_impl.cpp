/*******************************************************************************
* @file    sift_impl.cpp
* @brief   Bundled SIFT (Lowe, IJCV 2004), for OpenCV builds without SIFT (< 4.4).
*
* Follows the structure of OpenCV's implementation: 2x upsampled base image,
* octave_layers + 3 Gaussian images per octave, 3x3x3 DoG extrema, quadratic
* sub-pixel refinement, contrast and edge rejection, 36-bin orientation
* histogram and a 4x4x8 trilinearly interpolated descriptor.
* Image intensities are in [0, 1], so contrast_threshold has the OpenCV meaning.
*******************************************************************************/

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "sfm/features.h"

namespace sfm
{
namespace
{
constexpr int kImgBorder = 5;
constexpr int kMaxInterpSteps = 5;
constexpr int kOriHistBins = 36;
constexpr float kOriSigFactor = 1.5f;
constexpr float kOriRadiusFactor = 3.0f * kOriSigFactor;
constexpr float kOriPeakRatio = 0.8f;
constexpr int kDescWidth = 4;
constexpr int kDescHistBins = 8;
constexpr float kDescSclFactor = 3.0f;
constexpr float kDescMagThr = 0.2f;
constexpr float kInitSigma = 0.5f;
constexpr float kDegPerRad = 57.29577951308232f;

struct Extremum
{
  int octave;
  int layer;
  int r;          ///< integer row in the octave after refinement
  int c;          ///< integer column in the octave after refinement
  float x_oct;    ///< sub-pixel column in the octave
  float y_oct;    ///< sub-pixel row in the octave
  float scl_octv; ///< scale in octave pixels
  float response;
};

using Pyramid = std::vector<std::vector<cv::Mat>>;

void BuildGaussianPyramid(const cv::Mat &gray, const SiftParams &p, Pyramid *gauss)
{
  cv::Mat img;
  gray.convertTo(img, CV_32F, 1.0 / 255.0);
  cv::Mat base;
  cv::resize(img, base, cv::Size(img.cols * 2, img.rows * 2), 0, 0, cv::INTER_LINEAR);
  const double sig_diff = std::sqrt(std::max(p.sigma * p.sigma - 4.0 * kInitSigma * kInitSigma, 0.01));
  cv::GaussianBlur(base, base, cv::Size(), sig_diff, sig_diff);

  const int L = p.octave_layers;
  const int n_octaves = std::max(1, static_cast<int>(std::lround(std::log2(std::min(base.cols, base.rows)) - 2.0)) + 1);

  std::vector<double> sig(L + 3);
  sig[0] = p.sigma;
  const double k = std::pow(2.0, 1.0 / L);
  for (int i = 1; i < L + 3; ++i)
  {
    const double sig_prev = std::pow(k, i - 1) * p.sigma;
    const double sig_total = sig_prev * k;
    sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
  }

  gauss->assign(n_octaves, std::vector<cv::Mat>(L + 3));
  for (int o = 0; o < n_octaves; ++o)
  {
    for (int i = 0; i < L + 3; ++i)
    {
      cv::Mat &dst = (*gauss)[o][i];
      if (o == 0 && i == 0)
        dst = base;
      else if (i == 0)
      {
        const cv::Mat &src = (*gauss)[o - 1][L];
        cv::resize(src, dst, cv::Size(src.cols / 2, src.rows / 2), 0, 0, cv::INTER_NEAREST);
      }
      else
        cv::GaussianBlur((*gauss)[o][i - 1], dst, cv::Size(), sig[i], sig[i]);
    }
    if ((*gauss)[o][0].cols < 2 * kImgBorder + 3 || (*gauss)[o][0].rows < 2 * kImgBorder + 3)
    {
      gauss->resize(o + 1);
      break;
    }
  }
}

// Quadratic fit around a DoG extremum. Mirrors OpenCV's adjustLocalExtrema.
bool RefineExtremum(const std::vector<cv::Mat> &dog, const SiftParams &p, int octave, int layer, int r, int c, Extremum *out)
{
  const int L = p.octave_layers;
  float xi = 0, xr = 0, xc = 0, contr = 0;
  float dxx = 0, dyy = 0, dxy = 0;
  int step = 0;
  for (; step < kMaxInterpSteps; ++step)
  {
    const cv::Mat &prev = dog[layer - 1];
    const cv::Mat &cur = dog[layer];
    const cv::Mat &next = dog[layer + 1];
    auto C = [&](int rr, int cc) { return cur.at<float>(rr, cc); };
    const cv::Vec3f dD((C(r, c + 1) - C(r, c - 1)) * 0.5f, (C(r + 1, c) - C(r - 1, c)) * 0.5f,
                       (next.at<float>(r, c) - prev.at<float>(r, c)) * 0.5f);
    const float v2 = C(r, c) * 2.0f;
    dxx = C(r, c + 1) + C(r, c - 1) - v2;
    dyy = C(r + 1, c) + C(r - 1, c) - v2;
    const float dss = next.at<float>(r, c) + prev.at<float>(r, c) - v2;
    dxy = (C(r + 1, c + 1) - C(r + 1, c - 1) - C(r - 1, c + 1) + C(r - 1, c - 1)) * 0.25f;
    const float dxs = (next.at<float>(r, c + 1) - next.at<float>(r, c - 1) - prev.at<float>(r, c + 1) + prev.at<float>(r, c - 1)) * 0.25f;
    const float dys = (next.at<float>(r + 1, c) - next.at<float>(r - 1, c) - prev.at<float>(r + 1, c) + prev.at<float>(r - 1, c)) * 0.25f;
    const cv::Matx33f H(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
    const cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);
    xc = -X[0];
    xr = -X[1];
    xi = -X[2];
    if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
    {
      contr = C(r, c) + 0.5f * dD.dot(cv::Vec3f(xc, xr, xi));
      break;
    }
    if (std::abs(xi) > 1e6f || std::abs(xr) > 1e6f || std::abs(xc) > 1e6f)
      return false;
    c += static_cast<int>(std::lround(xc));
    r += static_cast<int>(std::lround(xr));
    layer += static_cast<int>(std::lround(xi));
    if (layer < 1 || layer > L || c < kImgBorder || c >= cur.cols - kImgBorder || r < kImgBorder || r >= cur.rows - kImgBorder)
      return false;
  }
  if (step >= kMaxInterpSteps)
    return false;
  if (std::abs(contr) * L < p.contrast_threshold)
    return false;
  const float tr = dxx + dyy;
  const float det = dxx * dyy - dxy * dxy;
  const float edge = static_cast<float>(p.edge_threshold);
  if (det <= 0 || tr * tr * edge >= (edge + 1) * (edge + 1) * det)
    return false;

  out->octave = octave;
  out->layer = layer;
  out->r = r;
  out->c = c;
  out->x_oct = c + xc;
  out->y_oct = r + xr;
  out->scl_octv = static_cast<float>(p.sigma * std::pow(2.0, (layer + xi) / L));
  out->response = std::abs(contr);
  return true;
}

void FindExtrema(const Pyramid &gauss, const SiftParams &p, std::vector<Extremum> *extrema)
{
  const int L = p.octave_layers;
  const float threshold = static_cast<float>(0.5 * p.contrast_threshold / L);
  for (int o = 0; o < static_cast<int>(gauss.size()); ++o)
  {
    std::vector<cv::Mat> dog(L + 2);
    for (int i = 0; i < L + 2; ++i)
      cv::subtract(gauss[o][i + 1], gauss[o][i], dog[i]);

    for (int i = 1; i <= L; ++i)
    {
      const cv::Mat &prev = dog[i - 1];
      const cv::Mat &cur = dog[i];
      const cv::Mat &next = dog[i + 1];
      for (int r = kImgBorder; r < cur.rows - kImgBorder; ++r)
      {
        for (int c = kImgBorder; c < cur.cols - kImgBorder; ++c)
        {
          const float val = cur.at<float>(r, c);
          if (std::abs(val) <= threshold)
            continue;
          bool is_ext = true;
          for (int dr = -1; dr <= 1 && is_ext; ++dr)
          {
            for (int dc = -1; dc <= 1 && is_ext; ++dc)
            {
              const float a = prev.at<float>(r + dr, c + dc);
              const float b = cur.at<float>(r + dr, c + dc);
              const float d = next.at<float>(r + dr, c + dc);
              if (val > 0)
                is_ext = val >= a && val >= d && (val >= b);
              else
                is_ext = val <= a && val <= d && (val <= b);
            }
          }
          if (!is_ext)
            continue;
          Extremum e;
          if (RefineExtremum(dog, p, o, i, r, c, &e))
            extrema->push_back(e);
        }
      }
    }
  }
}

/// Dominant gradient orientations (degrees, image frame with y down).
void ComputeOrientations(const cv::Mat &img, const Extremum &e, std::vector<float> *angles)
{
  const float sigma_w = kOriSigFactor * e.scl_octv;
  const int radius = static_cast<int>(std::lround(kOriRadiusFactor * e.scl_octv));
  const float exp_scale = -1.0f / (2.0f * sigma_w * sigma_w);
  float hist[kOriHistBins] = {0};
  for (int i = -radius; i <= radius; ++i)
  {
    const int y = e.r + i;
    if (y <= 0 || y >= img.rows - 1)
      continue;
    for (int j = -radius; j <= radius; ++j)
    {
      const int x = e.c + j;
      if (x <= 0 || x >= img.cols - 1)
        continue;
      const float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
      const float dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
      const float w = std::exp((i * i + j * j) * exp_scale);
      float ori = std::atan2(dy, dx) * kDegPerRad;
      if (ori < 0)
        ori += 360.0f;
      int bin = static_cast<int>(std::lround(ori * kOriHistBins / 360.0f));
      if (bin >= kOriHistBins)
        bin -= kOriHistBins;
      hist[bin] += w * std::sqrt(dx * dx + dy * dy);
    }
  }
  float smooth[kOriHistBins];
  for (int i = 0; i < kOriHistBins; ++i)
  {
    auto h = [&](int k) { return hist[(k + kOriHistBins) % kOriHistBins]; };
    smooth[i] = (h(i - 2) + h(i + 2)) * (1.0f / 16) + (h(i - 1) + h(i + 1)) * (4.0f / 16) + h(i) * (6.0f / 16);
  }
  const float max_val = *std::max_element(smooth, smooth + kOriHistBins);
  const float thr = max_val * kOriPeakRatio;
  for (int i = 0; i < kOriHistBins; ++i)
  {
    const int l = (i - 1 + kOriHistBins) % kOriHistBins;
    const int r = (i + 1) % kOriHistBins;
    if (smooth[i] > smooth[l] && smooth[i] > smooth[r] && smooth[i] >= thr)
    {
      float bin = i + 0.5f * (smooth[l] - smooth[r]) / (smooth[l] - 2 * smooth[i] + smooth[r]);
      if (bin < 0)
        bin += kOriHistBins;
      else if (bin >= kOriHistBins)
        bin -= kOriHistBins;
      angles->push_back(bin * 360.0f / kOriHistBins);
    }
  }
}

void ComputeDescriptor(const cv::Mat &img, const Extremum &e, float angle_deg, float *dst)
{
  constexpr int d = kDescWidth;
  constexpr int n = kDescHistBins;
  const float angle = angle_deg / kDegPerRad;
  const float hist_width = kDescSclFactor * e.scl_octv;
  int radius = static_cast<int>(std::lround(hist_width * 1.4142135623730951f * (d + 1) * 0.5f));
  radius = std::min(radius, static_cast<int>(std::sqrt(static_cast<double>(img.rows) * img.rows + static_cast<double>(img.cols) * img.cols)));
  const float cos_t = std::cos(angle) / hist_width;
  const float sin_t = std::sin(angle) / hist_width;
  const float bins_per_rad = n / (2.0f * 3.14159265358979f);
  const float exp_scale = -1.0f / (d * d * 0.5f);

  std::vector<float> hist((d + 2) * (d + 2) * (n + 2), 0.0f);
  for (int i = -radius; i <= radius; ++i)
  {
    for (int j = -radius; j <= radius; ++j)
    {
      // Rotate the sample offset (j, i) by -angle into the keypoint frame.
      const float c_rot = j * cos_t + i * sin_t;
      const float r_rot = -j * sin_t + i * cos_t;
      float rbin = r_rot + d / 2.0f - 0.5f;
      float cbin = c_rot + d / 2.0f - 0.5f;
      const int r = e.r + i;
      const int c = e.c + j;
      if (rbin <= -1 || rbin >= d || cbin <= -1 || cbin >= d || r <= 0 || r >= img.rows - 1 || c <= 0 || c >= img.cols - 1)
        continue;
      const float dx = img.at<float>(r, c + 1) - img.at<float>(r, c - 1);
      const float dy = img.at<float>(r + 1, c) - img.at<float>(r - 1, c);
      const float mag = std::sqrt(dx * dx + dy * dy) * std::exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
      float obin = (std::atan2(dy, dx) - angle) * bins_per_rad;
      while (obin < 0)
        obin += n;
      while (obin >= n)
        obin -= n;

      const int r0 = static_cast<int>(std::floor(rbin));
      const int c0 = static_cast<int>(std::floor(cbin));
      int o0 = static_cast<int>(std::floor(obin));
      rbin -= r0;
      cbin -= c0;
      obin -= o0;
      if (o0 >= n)
        o0 -= n;

      const float v_r1 = mag * rbin, v_r0 = mag - v_r1;
      const float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
      const float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
      const float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
      const float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
      const float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
      const float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

      const int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
      hist[idx] += v_rco000;
      hist[idx + 1] += v_rco001;
      hist[idx + (n + 2)] += v_rco010;
      hist[idx + (n + 3)] += v_rco011;
      hist[idx + (d + 2) * (n + 2)] += v_rco100;
      hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
      hist[idx + (d + 3) * (n + 2)] += v_rco110;
      hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
    }
  }

  // Fold the circular orientation bins and copy out.
  for (int i = 0; i < d; ++i)
  {
    for (int j = 0; j < d; ++j)
    {
      const int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
      hist[idx] += hist[idx + n];
      hist[idx + 1] += hist[idx + n + 1];
      for (int k = 0; k < n; ++k)
        dst[(i * d + j) * n + k] = hist[idx + k];
    }
  }

  const int len = d * d * n;
  float nrm2 = 0;
  for (int k = 0; k < len; ++k)
    nrm2 += dst[k] * dst[k];
  const float thr = std::sqrt(nrm2) * kDescMagThr;
  nrm2 = 0;
  for (int k = 0; k < len; ++k)
  {
    dst[k] = std::min(dst[k], thr);
    nrm2 += dst[k] * dst[k];
  }
  const float inv = 1.0f / std::max(std::sqrt(nrm2), 1e-12f);
  for (int k = 0; k < len; ++k)
    dst[k] *= inv;
}
} // namespace

void ExtractSiftBundled(const cv::Mat &gray, const SiftParams &params, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors)
{
  CV_Assert(gray.type() == CV_8UC1);
  Pyramid gauss;
  BuildGaussianPyramid(gray, params, &gauss);
  std::vector<Extremum> extrema;
  FindExtrema(gauss, params, &extrema);

  keypoints->clear();
  std::vector<std::vector<float>> desc_rows;
  desc_rows.reserve(extrema.size());
  constexpr int kDescLen = kDescWidth * kDescWidth * kDescHistBins;
  for (const Extremum &e : extrema)
  {
    const cv::Mat &img = gauss[e.octave][e.layer];
    std::vector<float> angles;
    ComputeOrientations(img, e, &angles);
    const float octave_scale = static_cast<float>(1 << e.octave);
    for (float angle : angles)
    {
      cv::KeyPoint kp;
      // Octave -> upsampled base (x * 2^o) -> original image ((x_up - 0.5) / 2).
      kp.pt.x = (e.x_oct * octave_scale - 0.5f) * 0.5f;
      kp.pt.y = (e.y_oct * octave_scale - 0.5f) * 0.5f;
      kp.size = e.scl_octv * octave_scale; // 2 * sigma in the original image
      kp.angle = angle;
      kp.response = e.response;
      kp.octave = e.octave - 1;
      keypoints->push_back(kp);
      desc_rows.emplace_back(kDescLen);
      ComputeDescriptor(img, e, angle, desc_rows.back().data());
    }
  }

  descriptors->create(static_cast<int>(desc_rows.size()), kDescLen, CV_32F);
  for (int i = 0; i < descriptors->rows; ++i)
    std::copy(desc_rows[i].begin(), desc_rows[i].end(), descriptors->ptr<float>(i));
}

} // namespace sfm
