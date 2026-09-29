"""The preprocessing step sequence for one dataset directory.

    <dataset>/prepare.yaml   (optional) settings, all keys optional:
        images: images                 # image folder, relative to the dataset
        exclude: [IMG_0001.JPG]        # images to leave out
        sensor_db: my_sensors.json     # extra sensor widths, merged over sensor_db.json
        cameras: {sensor_width_mm: 6.17}          # or a list: [{match: {model: FC300S}, fx: ..., dist: [...]}]
        trajectory: gps                # gps | none | <path to an existing trajectory file>
        config: {matching: {knn: 8}}   # merged into the generated config.yaml
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import yaml

from . import cameras as cam_mod
from . import config_writer, metadata, trajectory


def prepare(dataset: Path, verbose: bool = True) -> Path:
    dataset = dataset.expanduser().resolve()
    settings_file = dataset / "prepare.yaml"
    settings = (yaml.safe_load(settings_file.read_text()) or {}) if settings_file.exists() else {}
    log = print if verbose else (lambda *a, **k: None)

    # 1. Images
    image_dir = dataset / settings.get("images", "images")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No image folder {image_dir} (set 'images' in prepare.yaml)")
    exclude = set(settings.get("exclude", []))
    paths = [p for p in metadata.list_images(image_dir) if p.name not in exclude]
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}")
    log(f"[1/4] {len(paths)} images in {image_dir}" + (f" ({len(exclude)} excluded)" if exclude else ""))

    # 2. Metadata
    with ThreadPoolExecutor(8) as pool:
        metas = list(pool.map(metadata.read_metadata, paths))
    out_dir = dataset / "preprocess"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps([metadata.to_dict(m) for m in metas], indent=1))
    n_gps = sum(m.has_gps for m in metas)
    log(f"[2/4] metadata: {n_gps}/{len(metas)} with GPS, {sum(m.gimbal_pitch is not None for m in metas)} with gimbal angles "
        f"-> {out_dir / 'metadata.json'}")

    # 3. Cameras
    db = cam_mod.load_sensor_db(dataset / settings["sensor_db"] if "sensor_db" in settings else None)
    cams, image_camera = cam_mod.build_cameras(metas, db, settings.get("cameras"))
    for c in cams:
        make, model, w, h, f_mm = c.key
        log(f"[3/4] camera {c.id}: {make} {model} {w}x{h}, {len(c.images)} images, fx = {c.fx:.1f} px ({c.method})")

    # 4. Trajectory
    traj_setting = str(settings.get("trajectory", "gps"))
    traj_file = None
    notes = []
    if traj_setting == "gps" and n_gps >= 2:
        origin, rows = trajectory.build_trajectory(metas)
        trajectory.write_trajectory(out_dir / "trajectory.txt", rows)
        (out_dir / "geo_origin.yaml").write_text(yaml.safe_dump({"latitude": origin[0], "longitude": origin[1], "altitude": origin[2]}))
        traj_file = "preprocess/trajectory.txt"
        spacing = trajectory.median_spacing(rows)
        notes.append(f"GPS trajectory: {len(rows)} images, median spacing {spacing:.1f} m (a radius search could use ~{3 * spacing:.0f} m)")
        log(f"[4/4] trajectory: {len(rows)} GPS positions, median spacing {spacing:.1f} m -> {out_dir / 'trajectory.txt'}")
    elif traj_setting not in ("gps", "none"):
        traj_file = traj_setting
        log(f"[4/4] trajectory: using {traj_file}")
    else:
        log("[4/4] trajectory: none")

    # Config
    rel_images = _relative(image_dir, dataset)
    cfg = config_writer.build_config(metas, cams, image_camera, rel_images, traj_file,
                                     image_list=[p.name for p in paths] if exclude else None, overrides=settings.get("config"))
    notes.insert(0, "Cameras: " + "; ".join(f"{c.id}: {c.key[0]} {c.key[1]} fx {c.fx:.1f} px ({c.method})" for c in cams))
    target = config_writer.write_config(dataset / "config.yaml", cfg, notes)
    log(f"Wrote {target}")
    return target


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
