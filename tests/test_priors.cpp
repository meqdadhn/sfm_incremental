// EXIF parsing, intrinsics from EXIF, GPS -> ENU and trajectory-based pair selection.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "sfm/exif.h"
#include "sfm/matching.h"
#include "sfm/priors.h"

namespace sfm
{
namespace
{
/// Builds a TIFF-structured EXIF payload with IFD0 (make, model, pointers), Exif IFD and GPS IFD.
class ExifBuilder
{
public:
  explicit ExifBuilder(bool little_endian) : le_(little_endian) {}

  std::vector<unsigned char> Build()
  {
    // Layout: header(8) | IFD0 | ExifIFD | GPSIFD | data area
    const std::vector<Entry> ifd0 = {Ascii(0x010F, "DJI"), Ascii(0x0110, "FC300S"), Long(0x8769, 0), Long(0x8825, 0)};
    const std::vector<Entry> exif = {Rational(0x920A, {{361, 100}}), Short(0xA405, 20), Long(0xA002, 4000), Long(0xA003, 2250)};
    const std::vector<Entry> gps = {Ascii(0x0001, "N"), Rational(0x0002, {{46, 1}, {50, 1}, {333855, 10000}}), Ascii(0x0003, "W"),
                                    Rational(0x0004, {{91, 1}, {59, 1}, {404156, 10000}}), Byte(0x0005, 0), Rational(0x0006, {{198309, 1000}})};
    const uint32_t ifd0_off = 8;
    const uint32_t exif_off = ifd0_off + IfdSize(ifd0);
    const uint32_t gps_off = exif_off + IfdSize(exif);
    data_off_ = gps_off + IfdSize(gps);

    out_.clear();
    Put8(le_ ? 'I' : 'M');
    Put8(le_ ? 'I' : 'M');
    Put16(42);
    Put32(ifd0_off);
    std::vector<Entry> ifd0_fixed = ifd0;
    ifd0_fixed[2].inline_value = exif_off;
    ifd0_fixed[3].inline_value = gps_off;
    WriteIfd(ifd0_fixed);
    WriteIfd(exif);
    WriteIfd(gps);
    out_.insert(out_.end(), data_.begin(), data_.end());
    return out_;
  }

private:
  struct Entry
  {
    uint16_t tag, type;
    uint32_t count;
    std::vector<unsigned char> payload; // > 4 bytes: stored in the data area
    uint32_t inline_value = 0;
  };

  static uint32_t IfdSize(const std::vector<Entry> &ifd) { return static_cast<uint32_t>(2 + 12 * ifd.size() + 4); }

  std::vector<unsigned char> Bytes(uint32_t v, int n) const
  {
    std::vector<unsigned char> b(n);
    for (int i = 0; i < n; ++i)
      b[le_ ? i : n - 1 - i] = static_cast<unsigned char>(v >> (8 * i));
    return b;
  }
  Entry Ascii(uint16_t tag, const std::string &s) const
  {
    Entry e{tag, 2, static_cast<uint32_t>(s.size() + 1), std::vector<unsigned char>(s.begin(), s.end())};
    e.payload.push_back(0);
    return e;
  }
  Entry Short(uint16_t tag, uint16_t v) const { return {tag, 3, 1, Bytes(v, 2)}; }
  Entry Long(uint16_t tag, uint32_t v) const { return {tag, 4, 1, Bytes(v, 4)}; }
  Entry Byte(uint16_t tag, uint8_t v) const { return {tag, 1, 1, {v}}; }
  Entry Rational(uint16_t tag, const std::vector<std::pair<uint32_t, uint32_t>> &vals) const
  {
    Entry e{tag, 5, static_cast<uint32_t>(vals.size()), {}};
    for (const auto &v : vals)
    {
      const auto n = Bytes(v.first, 4), d = Bytes(v.second, 4);
      e.payload.insert(e.payload.end(), n.begin(), n.end());
      e.payload.insert(e.payload.end(), d.begin(), d.end());
    }
    return e;
  }

  void Put8(uint8_t v) { out_.push_back(v); }
  void Put16(uint16_t v)
  {
    const auto b = Bytes(v, 2);
    out_.insert(out_.end(), b.begin(), b.end());
  }
  void Put32(uint32_t v)
  {
    const auto b = Bytes(v, 4);
    out_.insert(out_.end(), b.begin(), b.end());
  }

  void WriteIfd(const std::vector<Entry> &ifd)
  {
    Put16(static_cast<uint16_t>(ifd.size()));
    for (const Entry &e : ifd)
    {
      Put16(e.tag);
      Put16(e.type);
      Put32(e.count);
      if (e.inline_value)
        Put32(e.inline_value);
      else if (e.payload.size() <= 4)
      {
        std::vector<unsigned char> v = e.payload;
        v.resize(4, 0);
        out_.insert(out_.end(), v.begin(), v.end());
      }
      else
      {
        Put32(data_off_ + static_cast<uint32_t>(data_.size()));
        data_.insert(data_.end(), e.payload.begin(), e.payload.end());
      }
    }
    Put32(0); // next IFD
  }

  bool le_;
  uint32_t data_off_ = 0;
  std::vector<unsigned char> out_, data_;
};

void ExpectBrightonBeachExif(const ExifData &exif)
{
  EXPECT_EQ(exif.make, "DJI");
  EXPECT_EQ(exif.model, "FC300S");
  EXPECT_DOUBLE_EQ(exif.focal_mm, 3.61);
  EXPECT_DOUBLE_EQ(exif.focal_35mm, 20.0);
  EXPECT_EQ(exif.width, 4000);
  EXPECT_EQ(exif.height, 2250);
  ASSERT_TRUE(exif.has_gps);
  EXPECT_NEAR(exif.latitude, 46 + 50 / 60.0 + 33.3855 / 3600.0, 1e-9);
  EXPECT_NEAR(exif.longitude, -(91 + 59 / 60.0 + 40.4156 / 3600.0), 1e-9);
  EXPECT_NEAR(exif.altitude, 198.309, 1e-9);
}

TEST(Exif, ParsesBothByteOrders)
{
  for (bool le : {true, false})
  {
    const std::vector<unsigned char> tiff = ExifBuilder(le).Build();
    ExifData exif;
    ASSERT_TRUE(ParseExifTiff(tiff.data(), tiff.size(), &exif));
    ExpectBrightonBeachExif(exif);
  }
}

TEST(Exif, ReadsJpegApp1)
{
  std::vector<unsigned char> jpg;
  cv::imencode(".jpg", cv::Mat(16, 16, CV_8UC3, cv::Scalar(100, 150, 200)), jpg);
  const std::vector<unsigned char> tiff = ExifBuilder(true).Build();
  std::vector<unsigned char> app1 = {0xFF, 0xE1, 0, 0, 'E', 'x', 'i', 'f', 0, 0};
  app1.insert(app1.end(), tiff.begin(), tiff.end());
  const size_t len = app1.size() - 2;
  app1[2] = static_cast<unsigned char>(len >> 8);
  app1[3] = static_cast<unsigned char>(len & 0xFF);
  jpg.insert(jpg.begin() + 2, app1.begin(), app1.end()); // right after SOI

  const std::string path = (std::filesystem::path(testing::TempDir()) / "sfm_exif_test.jpg").string();
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(jpg.data()), static_cast<std::streamsize>(jpg.size()));
  ExifData exif;
  ASSERT_TRUE(ReadExif(path, &exif));
  ExpectBrightonBeachExif(exif);
  EXPECT_FALSE(cv::imread(path).empty()); // still a valid JPEG
}

TEST(Priors, CameraFromExif)
{
  ExifData exif;
  exif.make = "DJI";
  exif.model = "FC300S";
  exif.focal_mm = 3.61;
  exif.focal_35mm = 20;
  Camera cam;
  std::string method;
  ASSERT_TRUE(CameraFromExif(exif, 4000, 2250, 0.0, &cam, &method));
  EXPECT_NEAR(cam.fx, 3.61 / 6.17 * 4000, 1e-9); // known sensor
  EXPECT_DOUBLE_EQ(cam.cx, 2000);
  EXPECT_DOUBLE_EQ(cam.cy, 1125);

  exif.model = "UNKNOWN";
  ASSERT_TRUE(CameraFromExif(exif, 4000, 2250, 0.0, &cam, &method));
  EXPECT_NEAR(cam.fx, 20.0 / 36.0 * 4000, 1e-9); // 35 mm fallback
  ASSERT_TRUE(CameraFromExif(exif, 4000, 2250, 6.17, &cam, &method));
  EXPECT_NEAR(cam.fx, 3.61 / 6.17 * 4000, 1e-9); // configured sensor width
}

TEST(Priors, GeodeticToEnu)
{
  const double lat0 = 46.84, lon0 = -91.99;
  EXPECT_LT(GeodeticToEnu(lat0, lon0, 200, lat0, lon0, 200).norm(), 1e-6);
  const Eigen::Vector3d up = GeodeticToEnu(lat0, lon0, 250, lat0, lon0, 200);
  EXPECT_NEAR(up.z(), 50.0, 1e-6);
  const Eigen::Vector3d north = GeodeticToEnu(lat0 + 0.001, lon0, 200, lat0, lon0, 200);
  EXPECT_NEAR(north.y(), 111.2, 0.5); // ~111 m per 0.001 deg of latitude
  EXPECT_NEAR(north.x(), 0.0, 1e-3);
  const Eigen::Vector3d east = GeodeticToEnu(lat0, lon0 + 0.001, 200, lat0, lon0, 200);
  EXPECT_NEAR(east.x(), 111.32 * std::cos(lat0 * 3.14159265358979 / 180.0), 0.5);
}

TEST(Matching, TrajectoryPairSelection)
{
  // 10 images on a line, 1 m apart.
  std::vector<Image> images(10);
  for (int i = 0; i < 10; ++i)
  {
    images[i].id = i;
    images[i].has_prior = true;
    images[i].prior_position = Eigen::Vector3d(i, 0, 0);
  }
  MatchingParams params;
  params.mode = "trajectory";
  params.search = "knn";
  params.knn = 2;
  auto pairs = SelectPairs(images, params);
  std::set<std::pair<int, int>> s(pairs.begin(), pairs.end());
  EXPECT_TRUE(s.count({0, 1}) && s.count({0, 2}) && s.count({3, 4}) && s.count({4, 5})); // ends reach 2 ahead
  EXPECT_FALSE(s.count({0, 3}));
  for (const auto &p : pairs)
    EXPECT_LE(p.second - p.first, 2);

  params.search = "radius";
  params.radius = 3.5;
  pairs = SelectPairs(images, params);
  for (const auto &p : pairs)
    EXPECT_LE(p.second - p.first, 3);
  EXPECT_EQ(pairs.size(), 9u + 8u + 7u);

  // An image without a prior is matched with everything.
  images[5].has_prior = false;
  pairs = SelectPairs(images, params);
  int with5 = 0;
  for (const auto &p : pairs)
    with5 += (p.first == 5 || p.second == 5);
  EXPECT_EQ(with5, 9);
}

} // namespace
} // namespace sfm
