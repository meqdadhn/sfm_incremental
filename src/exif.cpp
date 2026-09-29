#include "sfm/exif.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace sfm
{
namespace
{
enum Tag : uint16_t
{
  kMake = 0x010F,
  kModel = 0x0110,
  kExifIfd = 0x8769,
  kGpsIfd = 0x8825,
  kFocalLength = 0x920A,
  kPixelXDimension = 0xA002,
  kPixelYDimension = 0xA003,
  kFocalPlaneXRes = 0xA20E,
  kFocalPlaneUnit = 0xA210,
  kFocalLength35mm = 0xA405,
  kGpsLatRef = 0x0001,
  kGpsLat = 0x0002,
  kGpsLonRef = 0x0003,
  kGpsLon = 0x0004,
  kGpsAltRef = 0x0005,
  kGpsAlt = 0x0006,
};

class TiffReader
{
public:
  TiffReader(const unsigned char *data, size_t size) : d_(data), n_(size) {}

  bool Init()
  {
    if (n_ < 8)
      return false;
    if (d_[0] == 'I' && d_[1] == 'I')
      le_ = true;
    else if (d_[0] == 'M' && d_[1] == 'M')
      le_ = false;
    else
      return false;
    return U16(2) == 42;
  }

  uint16_t U16(size_t off) const
  {
    if (off + 2 > n_)
      return 0;
    return le_ ? uint16_t(d_[off] | d_[off + 1] << 8) : uint16_t(d_[off] << 8 | d_[off + 1]);
  }
  uint32_t U32(size_t off) const
  {
    if (off + 4 > n_)
      return 0;
    return le_ ? uint32_t(d_[off]) | uint32_t(d_[off + 1]) << 8 | uint32_t(d_[off + 2]) << 16 | uint32_t(d_[off + 3]) << 24
               : uint32_t(d_[off]) << 24 | uint32_t(d_[off + 1]) << 16 | uint32_t(d_[off + 2]) << 8 | uint32_t(d_[off + 3]);
  }

  struct Entry
  {
    uint16_t tag, type;
    uint32_t count;
    size_t value_off; ///< where the value bytes start (inline or at the offset)
  };

  static size_t TypeSize(uint16_t type)
  {
    switch (type)
    {
    case 1: case 2: case 6: case 7: return 1;
    case 3: case 8: return 2;
    case 4: case 9: case 11: return 4;
    case 5: case 10: case 12: return 8;
    default: return 0;
    }
  }

  std::vector<Entry> ReadIfd(size_t off) const
  {
    std::vector<Entry> entries;
    const uint16_t count = U16(off);
    for (uint16_t i = 0; i < count; ++i)
    {
      const size_t e = off + 2 + 12 * size_t(i);
      if (e + 12 > n_)
        break;
      Entry entry{U16(e), U16(e + 2), U32(e + 4), e + 8};
      const size_t bytes = TypeSize(entry.type) * entry.count;
      if (bytes > 4)
        entry.value_off = U32(e + 8);
      if (entry.value_off + bytes <= n_)
        entries.push_back(entry);
    }
    return entries;
  }

  double Number(const Entry &e, uint32_t index = 0) const
  {
    const size_t off = e.value_off + TypeSize(e.type) * index;
    switch (e.type)
    {
    case 1: case 7: return d_[off];
    case 3: return U16(off);
    case 4: return U32(off);
    case 9: return int32_t(U32(off));
    case 5:
    {
      const uint32_t den = U32(off + 4);
      return den ? double(U32(off)) / den : 0.0;
    }
    case 10:
    {
      const int32_t den = int32_t(U32(off + 4));
      return den ? double(int32_t(U32(off))) / den : 0.0;
    }
    default: return 0.0;
    }
  }

  std::string String(const Entry &e) const
  {
    std::string s(reinterpret_cast<const char *>(d_ + e.value_off), e.count);
    s.erase(std::find(s.begin(), s.end(), '\0'), s.end());
    while (!s.empty() && s.back() == ' ')
      s.pop_back();
    return s;
  }

private:
  const unsigned char *d_;
  size_t n_;
  bool le_ = true;
};

double Dms(const TiffReader &r, const TiffReader::Entry &e)
{
  return e.count >= 3 ? r.Number(e, 0) + r.Number(e, 1) / 60.0 + r.Number(e, 2) / 3600.0 : 0.0;
}
} // namespace

bool ParseExifTiff(const unsigned char *data, size_t size, ExifData *exif)
{
  TiffReader r(data, size);
  if (!r.Init())
    return false;

  size_t exif_ifd = 0, gps_ifd = 0;
  for (const auto &e : r.ReadIfd(r.U32(4)))
  {
    if (e.tag == kMake && e.type == 2)
      exif->make = r.String(e);
    else if (e.tag == kModel && e.type == 2)
      exif->model = r.String(e);
    else if (e.tag == kExifIfd)
      exif_ifd = static_cast<size_t>(r.Number(e));
    else if (e.tag == kGpsIfd)
      gps_ifd = static_cast<size_t>(r.Number(e));
  }

  if (exif_ifd)
  {
    for (const auto &e : r.ReadIfd(exif_ifd))
    {
      switch (e.tag)
      {
      case kFocalLength: exif->focal_mm = r.Number(e); break;
      case kFocalLength35mm: exif->focal_35mm = r.Number(e); break;
      case kPixelXDimension: exif->width = static_cast<int>(r.Number(e)); break;
      case kPixelYDimension: exif->height = static_cast<int>(r.Number(e)); break;
      case kFocalPlaneXRes: exif->focal_plane_x_res = r.Number(e); break;
      case kFocalPlaneUnit: exif->focal_plane_unit = static_cast<int>(r.Number(e)); break;
      default: break;
      }
    }
  }

  if (gps_ifd)
  {
    char lat_ref = 'N', lon_ref = 'E';
    int alt_ref = 0;
    bool has_lat = false, has_lon = false;
    for (const auto &e : r.ReadIfd(gps_ifd))
    {
      switch (e.tag)
      {
      case kGpsLatRef: lat_ref = r.String(e).empty() ? 'N' : r.String(e)[0]; break;
      case kGpsLonRef: lon_ref = r.String(e).empty() ? 'E' : r.String(e)[0]; break;
      case kGpsLat: exif->latitude = Dms(r, e); has_lat = true; break;
      case kGpsLon: exif->longitude = Dms(r, e); has_lon = true; break;
      case kGpsAltRef: alt_ref = static_cast<int>(r.Number(e)); break;
      case kGpsAlt: exif->altitude = r.Number(e); break;
      default: break;
      }
    }
    if (lat_ref == 'S')
      exif->latitude = -exif->latitude;
    if (lon_ref == 'W')
      exif->longitude = -exif->longitude;
    if (alt_ref == 1)
      exif->altitude = -exif->altitude;
    exif->has_gps = has_lat && has_lon;
  }
  return true;
}

bool ReadExif(const std::string &path, ExifData *exif)
{
  std::ifstream in(path, std::ios::binary);
  unsigned char soi[2];
  if (!in.read(reinterpret_cast<char *>(soi), 2) || soi[0] != 0xFF || soi[1] != 0xD8)
    return false;
  // Walk the JPEG markers until the first "Exif" APP1 or the start of scan.
  while (in)
  {
    unsigned char marker[4];
    if (!in.read(reinterpret_cast<char *>(marker), 4) || marker[0] != 0xFF)
      return false;
    const int type = marker[1];
    const size_t len = (size_t(marker[2]) << 8 | marker[3]);
    if (type == 0xDA || type == 0xD9 || len < 2)
      return false;
    std::vector<unsigned char> seg(len - 2);
    if (!in.read(reinterpret_cast<char *>(seg.data()), static_cast<std::streamsize>(seg.size())))
      return false;
    if (type == 0xE1 && seg.size() > 6 && std::memcmp(seg.data(), "Exif\0\0", 6) == 0)
      return ParseExifTiff(seg.data() + 6, seg.size() - 6, exif);
  }
  return false;
}

} // namespace sfm
