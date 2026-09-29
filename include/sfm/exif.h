/*******************************************************************************
* @file    exif.h
* @brief   Minimal EXIF reader (JPEG APP1) for camera intrinsics and GPS.
*
* Reads only what the preprocessing needs: make/model, focal length (mm and
* 35 mm equivalent), focal-plane resolution, image size and GPS position.
*******************************************************************************/

#ifndef SFM_EXIF_H_
#define SFM_EXIF_H_

#include <string>

namespace sfm
{
struct ExifData
{
  std::string make;
  std::string model;
  double focal_mm = 0.0;
  double focal_35mm = 0.0;
  double focal_plane_x_res = 0.0; ///< pixels per focal_plane_unit
  int focal_plane_unit = 0;       ///< 2 = inch, 3 = cm, 4 = mm
  int width = 0;
  int height = 0;

  bool has_gps = false;
  double latitude = 0.0;  ///< degrees, north positive
  double longitude = 0.0; ///< degrees, east positive
  double altitude = 0.0;  ///< meters
};

/// Parses the EXIF block of a JPEG file. Returns false if there is none.
bool ReadExif(const std::string &path, ExifData *exif);

/// Parses a raw TIFF-structured EXIF payload (the bytes after "Exif\0\0").
bool ParseExifTiff(const unsigned char *data, size_t size, ExifData *exif);

} // namespace sfm

#endif // SFM_EXIF_H_
