"""Per-image metadata: EXIF (camera, focal length, size), GPS, and DJI XMP tags.

Add new sources here (other drone vendors' XMP, sidecar files, flight logs, ...) by filling
more fields of ImageMeta; the later steps only look at ImageMeta.
"""
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from PIL import ExifTags, Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class ImageMeta:
    name: str
    width: int = 0
    height: int = 0
    make: str = ""
    model: str = ""
    focal_mm: Optional[float] = None
    focal_35mm: Optional[float] = None
    focal_plane_x_res: Optional[float] = None  # pixels per focal_plane_unit
    focal_plane_unit: Optional[int] = None  # 2 inch, 3 cm, 4 mm
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    # DJI XMP (drone-dji:*); other vendors can fill the same fields
    relative_altitude: Optional[float] = None
    gimbal_roll: Optional[float] = None
    gimbal_pitch: Optional[float] = None
    gimbal_yaw: Optional[float] = None
    calibrated_focal_px: Optional[float] = None  # some DJI models store a calibrated focal length in pixels
    calibrated_cx: Optional[float] = None
    calibrated_cy: Optional[float] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None


def list_images(image_dir: Path) -> List[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _num(v) -> float:
    if isinstance(v, tuple) and len(v) == 2:  # (num, den) in older Pillow
        return v[0] / v[1] if v[1] else 0.0
    if isinstance(v, Fraction):
        return float(v)
    return float(v)


def _dms(v) -> float:
    d, m, s = (_num(x) for x in v)
    return d + m / 60.0 + s / 3600.0


def _parse_exif(img: Image.Image, meta: ImageMeta) -> None:
    raw = img._getexif() if hasattr(img, "_getexif") else None
    if not raw:
        return
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    meta.make = str(tags.get("Make", "")).strip("\x00 ")
    meta.model = str(tags.get("Model", "")).strip("\x00 ")
    if "FocalLength" in tags:
        meta.focal_mm = _num(tags["FocalLength"])
    if tags.get("FocalLengthIn35mmFilm"):
        meta.focal_35mm = _num(tags["FocalLengthIn35mmFilm"])
    if "FocalPlaneXResolution" in tags:
        meta.focal_plane_x_res = _num(tags["FocalPlaneXResolution"])
        meta.focal_plane_unit = int(tags.get("FocalPlaneResolutionUnit", 2))
    gps = tags.get("GPSInfo")
    if gps:
        g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        if "GPSLatitude" in g and "GPSLongitude" in g:
            meta.latitude = _dms(g["GPSLatitude"]) * (-1 if g.get("GPSLatitudeRef", "N") == "S" else 1)
            meta.longitude = _dms(g["GPSLongitude"]) * (-1 if g.get("GPSLongitudeRef", "E") == "W" else 1)
        if "GPSAltitude" in g:
            ref = g.get("GPSAltitudeRef", b"\x00")
            below = ref in (1, b"\x01")
            meta.altitude = _num(g["GPSAltitude"]) * (-1 if below else 1)


_XMP_FIELDS = {
    "RelativeAltitude": "relative_altitude",
    "GimbalRollDegree": "gimbal_roll",
    "GimbalPitchDegree": "gimbal_pitch",
    "GimbalYawDegree": "gimbal_yaw",
    "CalibratedFocalLength": "calibrated_focal_px",
    "CalibratedOpticalCenterX": "calibrated_cx",
    "CalibratedOpticalCenterY": "calibrated_cy",
}


def _parse_xmp(img: Image.Image, meta: ImageMeta) -> None:
    xmp = b""
    for segment, data in getattr(img, "applist", []):
        if segment == "APP1" and data.startswith(b"http://ns.adobe.com/xap/1.0/"):
            xmp = data
    if not xmp:
        return
    text = xmp.decode("utf-8", errors="ignore")
    # Both attribute (drone-dji:Tag="v") and element (<drone-dji:Tag>v</...>) forms occur.
    for tag, value in re.findall(r'drone-dji:(\w+)="([^"]*)"', text) + re.findall(r"<drone-dji:(\w+)>([^<]*)<", text):
        if tag in _XMP_FIELDS:
            try:
                setattr(meta, _XMP_FIELDS[tag], float(value))
            except ValueError:
                pass
        else:
            meta.extra[tag] = value


def read_metadata(path: Path) -> ImageMeta:
    meta = ImageMeta(name=path.name)
    with Image.open(path) as img:
        meta.width, meta.height = img.size
        _parse_exif(img, meta)
        _parse_xmp(img, meta)
    return meta


def to_dict(meta: ImageMeta) -> dict:
    return asdict(meta)
