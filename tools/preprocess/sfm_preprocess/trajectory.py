"""Trajectory priors: GPS (WGS84) -> local ENU meters, written as 'name X Y Z roll pitch yaw'.

The C++ pipeline reads the first four columns (trajectory.source: file); the angles are
informative (gimbal angles when available, else 'nan').
"""
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .metadata import ImageMeta

_A = 6378137.0  # WGS84
_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo), (n + alt) * math.cos(la) * math.sin(lo), (n * (1 - _E2) + alt) * math.sin(la)])


def geodetic_to_enu(lat: float, lon: float, alt: float, origin: Tuple[float, float, float]) -> np.ndarray:
    lat0, lon0, alt0 = origin
    d = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(lat0, lon0, alt0)
    la, lo = math.radians(lat0), math.radians(lon0)
    R = np.array([
        [-math.sin(lo), math.cos(lo), 0.0],
        [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
    ])
    return R @ d


def build_trajectory(metas: List[ImageMeta], origin: Optional[Tuple[float, float, float]] = None):
    """Returns (origin, [(name, xyz, (roll, pitch, yaw))]) for images with GPS; origin = first GPS image."""
    gps = [m for m in metas if m.has_gps]
    if not gps:
        return None, []
    if origin is None:
        origin = (gps[0].latitude, gps[0].longitude, gps[0].altitude or 0.0)
    rows = []
    for m in gps:
        xyz = geodetic_to_enu(m.latitude, m.longitude, m.altitude or 0.0, origin)
        angles = tuple(v if v is not None else float("nan") for v in (m.gimbal_roll, m.gimbal_pitch, m.gimbal_yaw))
        rows.append((m.name, xyz, angles))
    return origin, rows


def write_trajectory(path: Path, rows) -> None:
    with open(path, "w") as f:
        f.write("# name X Y Z roll pitch yaw   (local ENU meters; gimbal angles in degrees, nan if unknown)\n")
        for name, xyz, (r, p, y) in rows:
            f.write(f"{name} {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f} {r:.2f} {p:.2f} {y:.2f}\n")


def median_spacing(rows) -> float:
    """Median distance to the nearest other image, a scale for choosing a search radius."""
    if len(rows) < 2:
        return 0.0
    P = np.array([xyz for _, xyz, _ in rows])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))
