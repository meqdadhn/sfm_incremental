"""Unit tests for sfm_preprocess:  python3 -m unittest discover tools/preprocess/tests"""
import io
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfm_preprocess import cameras, config_writer, metadata, prepare, trajectory  # noqa: E402


def exif_app1() -> bytes:
    """Little-endian TIFF with IFD0 (Make, Model, ExifIFD, GPSIFD), Exif IFD and GPS IFD (Brighton Beach values)."""
    entries0 = [(0x010F, 2, b"DJI\0"), (0x0110, 2, b"FC300S\0"), (0x8769, 4, None), (0x8825, 4, None)]
    exif = [(0x920A, 5, [(361, 100)]), (0xA405, 3, 20)]
    gps = [(0x0001, 2, b"N\0"), (0x0002, 5, [(46, 1), (50, 1), (333855, 10000)]), (0x0003, 2, b"W\0"),
           (0x0004, 5, [(91, 1), (59, 1), (404156, 10000)]), (0x0005, 1, 0), (0x0006, 5, [(198309, 1000)])]
    size = lambda ifd: 2 + 12 * len(ifd) + 4  # noqa: E731
    off0, off_exif = 8, 8 + size(entries0)
    off_gps = off_exif + size(exif)
    data_start = off_gps + size(gps)
    data = bytearray()

    def ifd(entries, pointers=None):
        out = struct.pack("<H", len(entries))
        for tag, typ, val in entries:
            if pointers and tag in pointers:
                payload = struct.pack("<I", pointers[tag])
                count = 1
            elif typ == 5:
                payload = b"".join(struct.pack("<II", n, d) for n, d in val)
                count = len(val)
            elif typ == 2:
                payload, count = val, len(val)
            elif typ == 3:
                payload, count = struct.pack("<H", val), 1
            else:
                payload, count = struct.pack("<B", val), 1
            if len(payload) <= 4:
                field = payload.ljust(4, b"\0")
            else:
                field = struct.pack("<I", data_start + len(data))
                data.extend(payload)
            out += struct.pack("<HHI", tag, typ, count) + field
        return out + struct.pack("<I", 0)

    tiff = b"II*\0" + struct.pack("<I", off0) + ifd(entries0, {0x8769: off_exif, 0x8825: off_gps}) + ifd(exif) + ifd(gps)
    return b"Exif\0\0" + tiff + bytes(data)


XMP = (b'http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><rdf:Description drone-dji:RelativeAltitude="+39.80" '
       b'drone-dji:GimbalPitchDegree="-89.90" drone-dji:GimbalYawDegree="+45.00" drone-dji:GimbalRollDegree="+0.00"/></x:xmpmeta>')


def write_jpeg(path: Path, size=(400, 225)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 150)).save(buf, "JPEG")
    jpg = buf.getvalue()
    segs = b""
    for payload in (exif_app1(), XMP):
        segs += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    path.write_bytes(jpg[:2] + segs + jpg[2:])


class MetadataTest(unittest.TestCase):
    def test_exif_gps_xmp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            write_jpeg(p)
            m = metadata.read_metadata(p)
        self.assertEqual((m.make, m.model, m.width, m.height), ("DJI", "FC300S", 400, 225))
        self.assertAlmostEqual(m.focal_mm, 3.61)
        self.assertAlmostEqual(m.focal_35mm, 20)
        self.assertAlmostEqual(m.latitude, 46 + 50 / 60 + 33.3855 / 3600, places=9)
        self.assertAlmostEqual(m.longitude, -(91 + 59 / 60 + 40.4156 / 3600), places=9)
        self.assertAlmostEqual(m.altitude, 198.309)
        self.assertAlmostEqual(m.relative_altitude, 39.8)
        self.assertAlmostEqual(m.gimbal_pitch, -89.9)
        self.assertAlmostEqual(m.gimbal_yaw, 45.0)


class CamerasTest(unittest.TestCase):
    def meta(self, **kw):
        base = dict(name="a.jpg", width=4000, height=2250, make="DJI", model="FC300S", focal_mm=3.61, focal_35mm=20)
        base.update(kw)
        return metadata.ImageMeta(**base)

    def test_focal_priority(self):
        db = cameras.load_sensor_db()
        f, _ = cameras.estimate_focal(self.meta(), db)
        self.assertAlmostEqual(f, 3.61 / 6.17 * 4000)
        f, _ = cameras.estimate_focal(self.meta(model="UNKNOWN"), db)
        self.assertAlmostEqual(f, 20 / 36 * 4000)  # 35 mm fallback
        f, _ = cameras.estimate_focal(self.meta(model="UNKNOWN"), db, sensor_width_mm=6.17)
        self.assertAlmostEqual(f, 3.61 / 6.17 * 4000)
        f, _ = cameras.estimate_focal(self.meta(calibrated_focal_px=2350.5), db)
        self.assertAlmostEqual(f, 2350.5)

    def test_grouping_and_overrides(self):
        metas = [self.meta(name="a.jpg"), self.meta(name="b.jpg"), self.meta(name="c.jpg", model="FC6310", width=5472, height=3648, focal_mm=8.8)]
        cams, image_camera = cameras.build_cameras(metas, cameras.load_sensor_db())
        self.assertEqual(len(cams), 2)
        self.assertEqual(image_camera, {"a.jpg": 0, "b.jpg": 0, "c.jpg": 1})
        self.assertAlmostEqual(cams[1].fx, 8.8 / 13.2 * 5472)
        self.assertEqual((cams[1].cx, cams[1].cy), (2736, 1824))
        cams, _ = cameras.build_cameras(metas, cameras.load_sensor_db(), [{"match": {"model": "fc6310"}, "fx": 3650.0, "dist": [0.01, 0, 0, 0, 0]}])
        self.assertEqual(cams[1].fx, 3650.0)
        self.assertEqual(cams[1].dist[0], 0.01)
        self.assertAlmostEqual(cams[0].fx, 3.61 / 6.17 * 4000)  # untouched

    def test_unknown_camera_raises(self):
        with self.assertRaises(ValueError):
            cameras.build_cameras([self.meta(model="UNKNOWN", focal_35mm=None)], cameras.load_sensor_db())


class TrajectoryTest(unittest.TestCase):
    def test_enu(self):
        o = (46.84, -91.99, 200.0)
        self.assertLess(abs(trajectory.geodetic_to_enu(*o, o)).max(), 1e-6)
        self.assertAlmostEqual(trajectory.geodetic_to_enu(46.84, -91.99, 250.0, o)[2], 50.0, places=5)
        n = trajectory.geodetic_to_enu(46.841, -91.99, 200.0, o)
        self.assertAlmostEqual(n[1], 111.2, delta=0.5)
        self.assertAlmostEqual(n[0], 0.0, places=3)
        e = trajectory.geodetic_to_enu(46.84, -91.989, 200.0, o)
        self.assertAlmostEqual(e[0], 111.32 * math.cos(math.radians(46.84)), delta=0.5)


class PrepareTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Path(d)
            (ds / "images").mkdir()
            for i in range(3):
                write_jpeg(ds / "images" / f"img_{i}.jpg")
            (ds / "prepare.yaml").write_text("exclude: [img_2.jpg]\nconfig:\n  matching: {knn: 5}\n  ortho: {gsd: 0.02}\n")
            target = prepare.prepare(ds, verbose=False)
            import yaml
            cfg = yaml.safe_load(target.read_text())
            self.assertEqual(cfg["io"]["images"], ["img_0.jpg", "img_1.jpg"])
            self.assertEqual(cfg["matching"], {"mode": "trajectory", "search": "knn", "knn": 5})
            self.assertEqual(cfg["ortho"], {"enabled": True, "gsd": 0.02})  # nadir detected + override merged
            self.assertEqual(cfg["final"]["ba"]["refine_focal_length"], False)
            self.assertAlmostEqual(cfg["cameras"][0]["fx"], round(3.61 / 6.17 * 400, 3))
            lines = [l for l in (ds / "preprocess/trajectory.txt").read_text().splitlines() if not l.startswith("#")]
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("img_0.jpg 0.0000 0.0000 0.0000"))

            # A hand-written config.yaml is never overwritten.
            target.write_text("io: {image_dir: images}\n")
            self.assertEqual(prepare.prepare(ds, verbose=False).name, "config.generated.yaml")

    def test_deep_merge(self):
        self.assertEqual(config_writer.deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4}), {"a": {"b": 1, "c": 3}, "d": 4})


if __name__ == "__main__":
    unittest.main()
