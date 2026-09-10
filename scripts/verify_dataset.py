#!/usr/bin/env python3
"""Validate the distributed PERPDSCD files without model dependencies."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from able.data import verify_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/perpdscd"))
    parser.add_argument("--source-archive", type=Path, help="Optionally check all original fields against a source ZIP")
    args = parser.parse_args()
    report = verify_dataset(args.dataset_dir, source_archive=args.source_archive)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
