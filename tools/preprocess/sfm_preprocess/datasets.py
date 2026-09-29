"""Dataset registry and downloaders.

Add a dataset by adding an entry to DATASETS. Any public GitHub repo also works ad hoc:
    prep.py download github:OWNER/REPO <dir> --include images/
'prepare' (optional) is written to <dir>/prepare.yaml on download, as dataset-specific defaults.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class GitHubDataset:
    repo: str
    include: List[str]  # path prefixes to download
    license: str
    description: str
    images: str = "images"  # image folder inside the repo
    prepare: Dict = field(default_factory=dict)


DATASETS: Dict[str, GitHubDataset] = {
    "brighton_beach": GitHubDataset(
        repo="OpenDroneMap/drone_dataset_brighton_beach",
        include=["images/", "brighton_beach.jpg", "LICENSE", "README.md"],
        license="BSD-2-Clause",
        description="18 nadir DJI Phantom 3 images (4000x2250), park + shoreline, ~62 MB; brighton_beach.jpg is ODM's ortho",
    ),
    "pacifica": GitHubDataset(
        repo="OpenDroneMap/odm_data_pacifica",
        include=["images/", "license.html", "README.md"],
        license="CC BY-SA 4.0 (Michele Tobias)",
        description="12 aerial images of Pacifica State Beach, ~70 MB",
    ),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sfm_preprocess"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download_github(repo: str, include: List[str], target: Path, workers: int = 8) -> List[Path]:
    tree = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"))
    files = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob" and any(t["path"].startswith(p) for p in include)]
    if not files:
        raise RuntimeError(f"No files matching {include} in {repo}")

    def fetch(path: str) -> Path:
        out = target / path
        if out.exists() and out.stat().st_size > 0:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"))
        return out

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fetch, files))


def download(name: str, target: Path, include: Optional[List[str]] = None) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    if name.startswith("github:"):
        repo = name[len("github:"):]
        files = download_github(repo, include or ["images/"], target)
        print(f"Downloaded {len(files)} files from {repo} to {target}")
        return target
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)} (or github:OWNER/REPO)")
    ds = DATASETS[name]
    files = download_github(ds.repo, include or ds.include, target)
    print(f"Downloaded {len(files)} files of '{name}' ({ds.license}) to {target}")
    prepare_file = target / "prepare.yaml"
    if not prepare_file.exists():
        prepare_file.write_text(f"# Dataset-specific preprocessing settings ({name}). See tools/preprocess/README.md.\n"
                                + yaml.safe_dump({"images": ds.images, **ds.prepare}, sort_keys=False))
    return target
