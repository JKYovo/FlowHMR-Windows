"""Download the pinned public Linux inference models, with resume and SHA-256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowhmr.model_downloads import download_asset
from flowhmr.model_sources import PUBLIC_ASSETS
from flowhmr import hamer_setup
from flowhmr.model_sources import HAMER_CODE, HAMER_DIRECT_ASSETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=tuple(PUBLIC_ASSETS),
                        default=["flowhmr", "yolox", "sam", "vggt", "joint-regressor"])
    parser.add_argument("--all", action="store_true", help="Also install public torso optimization and HaMeR models")
    args = parser.parse_args()

    def fetch(asset):
        last = [0.0]

        def progress(stage, name, done, total):
            now = time.monotonic()
            if now - last[0] > 15 or done == total:
                print(f"{asset.relative}: {stage} {done / 2**20:.1f}/{total / 2**20:.1f} MiB", flush=True)
                last[0] = now

        for attempt in range(1, 4):
            try:
                path = download_asset(asset, progress, lambda: None)
                if asset == HAMER_CODE:
                    hamer_setup.install_code(path)
                print(f"Verified: {asset.relative}", flush=True)
                return
            except Exception as error:
                print(f"Attempt {attempt}/3: {asset.relative}: {error}", flush=True)
                if attempt == 3:
                    raise

    groups = PUBLIC_ASSETS if args.all else dict.fromkeys(args.groups)
    assets = [asset for group in groups for asset in PUBLIC_ASSETS[group]]
    if args.all:
        assets.extend(HAMER_DIRECT_ASSETS)
        if not hamer_setup.runtime_installed():
            assets.append(HAMER_CODE)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, assets))
    print("Import licensed SMPL-H/MANO separately through the Models dialog; see models/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
