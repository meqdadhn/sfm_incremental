"""Dataset preprocessing for sfm_incremental.

Turns a folder of images into everything the C++ pipeline needs:
    <dataset>/config.yaml                  intrinsics, trajectory, matching, ortho settings
    <dataset>/preprocess/metadata.json     everything extracted per image (EXIF, GPS, DJI XMP)
    <dataset>/preprocess/trajectory.txt    name X Y Z roll pitch yaw (local ENU meters)
    <dataset>/preprocess/geo_origin.yaml   WGS84 origin of the ENU frame

Each step is a plain module, meant to be edited:
    datasets.py       dataset registry + downloaders
    metadata.py       EXIF / GPS / XMP extraction
    cameras.py        intrinsics from metadata, grouping images into cameras (sensor_db.json)
    trajectory.py     GPS -> local ENU trajectory
    config_writer.py  config.yaml generation (+ per-dataset overrides from prepare.yaml)
    prepare.py        the step sequence
"""
