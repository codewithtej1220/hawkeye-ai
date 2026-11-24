"""Download the real thermal + RGB datasets listed in sources.py from Kaggle.

Requires a Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json.

    python download_datasets.py            # all sources
    python download_datasets.py hituav     # one source
"""

import argparse
import shutil
import sys

from sources import RAW_DIR, SOURCES


def free_mb(path) -> float:
    return shutil.disk_usage(path).free / 1024**2


def download(source: dict, api) -> None:
    target = RAW_DIR / source.get("raw", source["slug"])
    marker = target / ".complete"
    if marker.exists():
        print(f"[skip] {source['slug']} already downloaded")
        return

    # Zip + extracted copy both exist briefly during unzip.
    needed = source["approx_mb"] * 2.3
    if free_mb(RAW_DIR) < needed:
        sys.exit(f"[error] {source['slug']} needs ~{needed:.0f} MB free, only {free_mb(RAW_DIR):.0f} MB available")

    print(f"[download] {source['kaggle']} (~{source['approx_mb']} MB) -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(source["kaggle"], path=str(target), unzip=True, quiet=False)

    for leftover in target.glob("*.zip"):
        leftover.unlink()
    for rel in source["prune"]:
        pruned = target / rel
        if pruned.is_dir():
            shutil.rmtree(pruned)
            print(f"[prune] removed {pruned}")
        elif pruned.exists():
            pruned.unlink()
            print(f"[prune] removed {pruned}")

    marker.touch()
    print(f"[done] {source['slug']}  (disk free: {free_mb(RAW_DIR):.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*", help="subset of source slugs to download")
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        if not args.slugs or source["slug"] in args.slugs:
            download(source, api)


if __name__ == "__main__":
    main()
