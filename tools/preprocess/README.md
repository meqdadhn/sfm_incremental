# Preprocessing

Python scripts that turn a folder of images into what `sfm_main` needs. They are meant to be
edited: each step is a small module, dataset-specific settings live in the dataset's
`prepare.yaml`, and the C++ pipeline only reads the files written here.

```bash
python3 tools/preprocess/prep.py list                                           # registered datasets
python3 tools/preprocess/prep.py all brighton_beach ~/sfm_data/brighton_beach   # download + prepare
python3 tools/preprocess/prep.py prepare ~/sfm_data/my_flight                   # any folder with images/
python3 tools/preprocess/prep.py download github:OWNER/REPO ~/sfm_data/x --include images/
build/sfm_main ~/sfm_data/brighton_beach/config.yaml
```

Requires Python ≥ 3.8, Pillow, numpy and PyYAML.

## Outputs, in the dataset folder

| File | Content |
|---|---|
| `config.yaml` | Pipeline config: intrinsics per camera, image → camera map, trajectory, matching, ortho. It carries a "generated" header, so re-running overwrites it. If you hand-write a `config.yaml`, it's never overwritten: the output goes to `config.generated.yaml` instead. |
| `preprocess/metadata.json` | Everything extracted per image: EXIF, GPS, DJI XMP (gimbal angles, relative altitude, …) |
| `preprocess/trajectory.txt` | `name X Y Z roll pitch yaw`, in local ENU metres with the first GPS image as origin (the pipeline reads `name X Y Z`) |
| `preprocess/geo_origin.yaml` | WGS84 latitude / longitude / altitude of the ENU origin |

## Steps (`sfm_preprocess/prepare.py`)

1. **Images:** everything in `images/` (or `prepare.yaml: images`), minus `exclude`.
2. **Metadata** (`metadata.py`): EXIF make/model/focal/size, GPS, and DJI `drone-dji:*` XMP tags.
   Other vendors' tags or sidecar files go here, as more `ImageMeta` fields.
3. **Cameras** (`cameras.py`): images with the same make, model, size and focal length become one
   camera. Focal length priority:
   1. `prepare.yaml` override
   2. XMP calibrated focal length
   3. `focal_mm / sensor_width × width`, with the sensor width from `sensor_db.json` or `prepare.yaml`
   4. EXIF focal-plane resolution
   5. 35 mm equivalent

   The principal point defaults to the center and distortion to 0.
4. **Trajectory** (`trajectory.py`): GPS → local ENU metres.
5. **Config** (`config_writer.py`) uses these heuristics:
   - GPS present: trajectory matching (knn 12). Otherwise exhaustive matching, or sequential
     above 150 images.
   - SIFT at 2000 px on large images.
   - Distortion refined in the final BA.
   - Nadir flight (mean gimbal pitch < −75°): focal length kept fixed and the ortho enabled.
   - Finally, `prepare.yaml: config` is deep-merged on top.

## `prepare.yaml` (optional, per dataset)

```yaml
images: images                   # image folder inside the dataset
exclude: [DJI_0042.JPG]          # images to leave out
sensor_db: sensors.json          # extra sensor widths ({"make model": mm}), merged over sensor_db.json
cameras: {sensor_width_mm: 6.17} # for all cameras, or per camera:
# cameras:
#   - match: {model: FC6310}
#     fx: 3650.2
#     fy: 3650.2
#     dist: [-0.01, 0.02, 0, 0, 0]
trajectory: gps                  # gps | none | path/to/existing_trajectory.txt
config:                          # anything from config/example.yaml, merged into config.yaml
  matching: {knn: 8}
  ortho: {gsd: 0.02}
```

## Adding a dataset

Add an entry to `DATASETS` in `sfm_preprocess/datasets.py`. The fields are the GitHub repo,
the path prefixes to download, the licence, and optional `prepare` defaults, which are
written to `prepare.yaml` on download. For data hosted elsewhere, add a download function
next to `download_github`, or just put the images in `<dataset>/images/` and run `prepare`.

## Tests

```bash
python3 -m unittest discover tools/preprocess/tests
```
