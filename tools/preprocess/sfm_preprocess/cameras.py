"""Intrinsics from metadata, and grouping of images into physical cameras.

Focal length priority (first that applies):
  1. explicit override from prepare.yaml (fx/fy/cx/cy/dist per camera)
  2. calibrated focal length in pixels from XMP (some DJI models)
  3. focal_mm / sensor_width_mm * width, sensor width from prepare.yaml or sensor_db.json
  4. EXIF focal plane resolution
  5. 35 mm equivalent focal length (approximate)
Principal point defaults to the image center, distortion to zero (refine it in the final BA).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metadata import ImageMeta

SENSOR_DB = Path(__file__).with_name("sensor_db.json")


def load_sensor_db(extra: Optional[Path] = None) -> Dict[str, float]:
    db = {k: v for k, v in json.loads(SENSOR_DB.read_text()).items() if not k.startswith("_")}
    if extra:
        db.update({k.lower(): v for k, v in json.loads(Path(extra).read_text()).items() if not k.startswith("_")})
    return db


@dataclass
class Camera:
    id: int
    key: Tuple
    width: int
    height: int
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    dist: List[float] = field(default_factory=lambda: [0.0] * 5)
    method: str = ""
    images: List[str] = field(default_factory=list)


def camera_key(m: ImageMeta) -> Tuple:
    """Images sharing make, model, size and focal length are treated as one camera."""
    return (m.make, m.model, m.width, m.height, round(m.focal_mm or 0.0, 2))


def estimate_focal(m: ImageMeta, db: Dict[str, float], sensor_width_mm: Optional[float] = None) -> Tuple[float, str]:
    if m.calibrated_focal_px:
        return m.calibrated_focal_px, "XMP calibrated focal length"
    if m.focal_mm:
        sw = sensor_width_mm or db.get(f"{m.make} {m.model}".lower())
        if sw:
            src = "prepare.yaml" if sensor_width_mm else "sensor_db.json"
            return m.focal_mm / sw * m.width, f"{m.focal_mm} mm / {sw} mm sensor ({src})"
        if m.focal_plane_x_res and m.focal_plane_unit in (2, 3, 4):
            unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}[m.focal_plane_unit]
            return m.focal_mm * m.focal_plane_x_res / unit_mm, "EXIF focal plane resolution"
    if m.focal_35mm:
        return m.focal_35mm / 36.0 * max(m.width, m.height), "35 mm equivalent (approximate)"
    return 0.0, ""


def build_cameras(metas: List[ImageMeta], db: Dict[str, float], overrides: Optional[dict] = None) -> Tuple[List[Camera], Dict[str, int]]:
    """Groups images into cameras and estimates their intrinsics.

    overrides (from prepare.yaml 'cameras'): {"sensor_width_mm": w} applies to all cameras, or a list of
    {"match": {"model": "FC300S"}, "sensor_width_mm": ..., "fx": ..., "fy": ..., "cx": ..., "cy": ..., "dist": [...]}.
    """
    overrides = overrides or {}
    cameras: Dict[Tuple, Camera] = {}
    image_camera: Dict[str, int] = {}
    for m in metas:
        key = camera_key(m)
        if key not in cameras:
            cameras[key] = Camera(id=len(cameras), key=key, width=m.width, height=m.height)
        cameras[key].images.append(m.name)
        image_camera[m.name] = cameras[key].id

    for cam in cameras.values():
        first = next(m for m in metas if m.name == cam.images[0])
        ov = _override_for(first, overrides)
        focal, method = estimate_focal(first, db, ov.get("sensor_width_mm"))
        cam.fx = cam.fy = focal
        cam.method = method
        cam.cx = first.calibrated_cx or cam.width / 2.0
        cam.cy = first.calibrated_cy or cam.height / 2.0
        for k in ("fx", "fy", "cx", "cy", "dist"):
            if k in ov:
                setattr(cam, k, ov[k])
                cam.method = "prepare.yaml override"
        if cam.fx <= 0:
            raise ValueError(
                f"No focal length for camera {cam.key} (image {first.name}): add its sensor width to sensor_db.json "
                f"or set cameras.sensor_width_mm / fx in prepare.yaml"
            )
    return list(cameras.values()), image_camera


def _override_for(m: ImageMeta, overrides) -> dict:
    if isinstance(overrides, dict):
        return overrides
    for entry in overrides:
        match = entry.get("match", {})
        if all(str(getattr(m, k, "")).lower() == str(v).lower() for k, v in match.items()):
            return {k: v for k, v in entry.items() if k != "match"}
    return {}
