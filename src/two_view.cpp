#include "sfm/two_view.h"

#include <algorithm>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "sfm/geometry.h"
#include "sfm/triangulation.h"

namespace sfm
{
namespace
{
/// sqrt of the Sampson distance, scaled to pixels. Parameters: quaternion (w, x, y, z), unit translation.
struct SampsonResidual
{
  SampsonResidual(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, double scale) : x1_(x1), x2_(x2), scale_(scale) {}

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const
  {
    T R[9];
    ceres::QuaternionToRotation(q, R); // row-major
    // E = [t]x R
    const T tx[9] = {T(0), -t[2], t[1], t[2], T(0), -t[0], -t[1], t[0], T(0)};
    T E[9];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        E[3 * r + c] = tx[3 * r] * R[c] + tx[3 * r + 1] * R[3 + c] + tx[3 * r + 2] * R[6 + c];

    const T a[3] = {T(x1_.x()), T(x1_.y()), T(1)};
    const T b[3] = {T(x2_.x()), T(x2_.y()), T(1)};
    T Ea[3], Etb[3];
    for (int r = 0; r < 3; ++r)
    {
      Ea[r] = E[3 * r] * a[0] + E[3 * r + 1] * a[1] + E[3 * r + 2] * a[2];
      Etb[r] = E[r] * b[0] + E[3 + r] * b[1] + E[6 + r] * b[2];
    }
    const T num = b[0] * Ea[0] + b[1] * Ea[1] + b[2] * Ea[2];
    const T den = Ea[0] * Ea[0] + Ea[1] * Ea[1] + Etb[0] * Etb[0] + Etb[1] * Etb[1];
    residual[0] = T(scale_) * num / ceres::sqrt(den + T(1e-20));
    return true;
  }

  Eigen::Vector2d x1_, x2_;
  double scale_;
};

void RefineRelativePose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, double focal, double loss_px,
                        Eigen::Matrix3d *R, Eigen::Vector3d *t)
{
  Eigen::Quaterniond eq(*R);
  double q[4] = {eq.w(), eq.x(), eq.y(), eq.z()};
  double tv[3] = {t->x(), t->y(), t->z()};

  ceres::Problem problem;
  for (size_t i = 0; i < x1.size(); ++i)
  {
    auto *cost = new ceres::AutoDiffCostFunction<SampsonResidual, 1, 4, 3>(new SampsonResidual(x1[i], x2[i], focal));
    problem.AddResidualBlock(cost, new ceres::CauchyLoss(loss_px), q, tv);
  }
  problem.SetParameterization(q, new ceres::QuaternionParameterization());
  problem.SetParameterization(tv, new ceres::HomogeneousVectorParameterization(3));

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 50;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  *R = Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized().toRotationMatrix();
  *t = Eigen::Vector3d(tv[0], tv[1], tv[2]).normalized();
}

/// Counts correspondences in front of both cameras and collects their triangulation angles.
int CheiralityCheck(const Pose &pose2, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                    std::vector<bool> *valid, std::vector<double> *angles)
{
  const Pose pose1;
  const Eigen::Vector3d C1 = pose1.Center();
  const Eigen::Vector3d C2 = pose2.Center();
  valid->assign(x1.size(), false);
  angles->assign(x1.size(), 0.0);
  int count = 0;
  for (size_t i = 0; i < x1.size(); ++i)
  {
    Eigen::Vector3d X;
    if (!TriangulateDLT({pose1, pose2}, {x1[i], x2[i]}, &X))
      continue;
    if (X.z() <= 0 || pose2.Transform(X).z() <= 0)
      continue;
    (*valid)[i] = true;
    (*angles)[i] = TriangulationAngleDeg(C1, C2, X);
    ++count;
  }
  return count;
}
} // namespace

double SampsonError(const Eigen::Matrix3d &E, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2)
{
  const Eigen::Vector3d a = x1.homogeneous();
  const Eigen::Vector3d b = x2.homogeneous();
  const Eigen::Vector3d Ea = E * a;
  const Eigen::Vector3d Etb = E.transpose() * b;
  const double num = b.dot(Ea);
  const double den = Ea.head<2>().squaredNorm() + Etb.head<2>().squaredNorm();
  return std::abs(num) / std::sqrt(den + 1e-20);
}

bool EstimateTwoViewGeometry(const Camera &camera1, const Camera &camera2, const std::vector<Eigen::Vector2d> &normalized1,
                             const std::vector<Eigen::Vector2d> &normalized2, const std::vector<FeatureMatch> &matches,
                             const TwoViewParams &params, TwoViewGeometry *geometry)
{
  geometry->num_raw_matches = static_cast<int>(matches.size());
  if (static_cast<int>(matches.size()) < params.min_inliers)
    return false;

  const int n = static_cast<int>(matches.size());
  std::vector<Eigen::Vector2d> x1(n), x2(n);
  cv::Mat pts1(n, 2, CV_64F), pts2(n, 2, CV_64F);
  for (int i = 0; i < n; ++i)
  {
    x1[i] = normalized1[matches[i].idx1];
    x2[i] = normalized2[matches[i].idx2];
    pts1.at<double>(i, 0) = x1[i].x();
    pts1.at<double>(i, 1) = x1[i].y();
    pts2.at<double>(i, 0) = x2[i].x();
    pts2.at<double>(i, 1) = x2[i].y();
  }
  const double focal = 0.5 * (camera1.MeanFocal() + camera2.MeanFocal());
  const double threshold = params.ransac_threshold_px / focal;

  // 1. Five-point essential matrix in RANSAC on normalized coordinates (K = I).
  cv::Mat inlier_mask;
  const cv::Mat E_cv = cv::findEssentialMat(pts1, pts2, cv::Mat::eye(3, 3, CV_64F), cv::RANSAC, params.confidence, threshold, inlier_mask);
  if (E_cv.rows != 3 || E_cv.cols != 3 || cv::countNonZero(inlier_mask) < params.min_inliers)
    return false;

  // 2. Decomposition into the four (R, t) candidates, picking the one with points in front of both cameras.
  cv::Mat R_cv, t_cv;
  if (cv::recoverPose(E_cv, pts1, pts2, cv::Mat::eye(3, 3, CV_64F), R_cv, t_cv, inlier_mask) < params.min_inliers)
    return false;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  cv::cv2eigen(R_cv, R);
  cv::cv2eigen(t_cv, t);
  t.normalize();

  // 3. Nonlinear refinement on the RANSAC inliers.
  if (params.refine)
  {
    std::vector<Eigen::Vector2d> in1, in2;
    for (int i = 0; i < n; ++i)
    {
      if (inlier_mask.at<uchar>(i))
      {
        in1.push_back(x1[i]);
        in2.push_back(x2[i]);
      }
    }
    RefineRelativePose(in1, in2, focal, params.refine_loss_px, &R, &t);
  }

  // 4. Final inlier set over all putative matches: epipolar error + cheirality.
  const Eigen::Matrix3d E = Skew(t) * R;
  std::vector<int> epi_inliers;
  for (int i = 0; i < n; ++i)
    if (SampsonError(E, x1[i], x2[i]) < threshold)
      epi_inliers.push_back(i);
  std::vector<Eigen::Vector2d> e1, e2;
  for (int i : epi_inliers)
  {
    e1.push_back(x1[i]);
    e2.push_back(x2[i]);
  }
  // The Sampson error does not see the sign of t; keep the sign with more points in front.
  std::vector<bool> valid, valid_flip;
  std::vector<double> angles, angles_flip;
  const int count = CheiralityCheck({R, t}, e1, e2, &valid, &angles);
  const int count_flip = CheiralityCheck({R, -t}, e1, e2, &valid_flip, &angles_flip);
  if (count_flip > count)
  {
    t = -t;
    valid.swap(valid_flip);
    angles.swap(angles_flip);
  }

  geometry->R = R;
  geometry->t = t;
  geometry->inliers.clear();
  std::vector<double> good_angles;
  for (size_t k = 0; k < epi_inliers.size(); ++k)
  {
    if (!valid[k])
      continue;
    geometry->inliers.push_back(matches[epi_inliers[k]]);
    good_angles.push_back(angles[k]);
  }
  if (static_cast<int>(geometry->inliers.size()) < params.min_inliers)
    return false;
  std::nth_element(good_angles.begin(), good_angles.begin() + good_angles.size() / 2, good_angles.end());
  geometry->median_triangulation_angle_deg = good_angles[good_angles.size() / 2];

  // Homography support, used to avoid planar / pure-rotation pairs when seeding.
  cv::Mat h_mask;
  const cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, params.homography_threshold_px / focal, h_mask);
  const int h_inliers = H.empty() ? 0 : cv::countNonZero(h_mask);
  geometry->homography_inlier_ratio = static_cast<double>(h_inliers) / geometry->inliers.size();
  return true;
}

} // namespace sfm
