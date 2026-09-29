#!/usr/bin/env python3
"""Entry point: python3 tools/preprocess/prep.py {list,download,prepare,all} ...

    prep.py list
    prep.py all brighton_beach ~/sfm_data/brighton_beach
    prep.py download github:OWNER/REPO ~/sfm_data/foo --include images/
    prep.py prepare ~/sfm_data/my_flight          # any folder with images/ (+ optional prepare.yaml)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sfm_preprocess.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
