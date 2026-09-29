"""CLI: python3 tools/preprocess/prep.py {list,download,prepare,all} ..."""
import argparse
import sys
from pathlib import Path

from . import datasets
from .prepare import prepare


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prep.py", description="Download and preprocess datasets for sfm_incremental")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered datasets")
    d = sub.add_parser("download", help="download a registered dataset or github:OWNER/REPO")
    d.add_argument("name")
    d.add_argument("dir", type=Path)
    d.add_argument("--include", nargs="+", help="path prefixes to download (default: the dataset's list, or images/)")
    p = sub.add_parser("prepare", help="extract metadata, intrinsics, trajectory and write config.yaml")
    p.add_argument("dir", type=Path)
    a = sub.add_parser("all", help="download + prepare")
    a.add_argument("name")
    a.add_argument("dir", type=Path)
    a.add_argument("--include", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name, ds in datasets.DATASETS.items():
            print(f"{name:16s} {ds.repo}\n{'':16s} {ds.description} [{ds.license}]")
        return 0
    if args.cmd in ("download", "all"):
        datasets.download(args.name, args.dir.expanduser(), args.include)
    if args.cmd in ("prepare", "all"):
        config = prepare(args.dir)
        print(f"\nRun:  build/sfm_main {config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
