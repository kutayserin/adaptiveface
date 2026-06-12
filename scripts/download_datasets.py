"""Download (or verify) the datasets we evaluate on.

Run from project root:
    python -m scripts.download_datasets --lfw --mfr2

RMFD does not have a stable open mirror, so we only check whether it has
been placed manually in `data/rmfd/`. The README documents how to obtain it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Allow running this file both as a module and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, LFW_DIR, MFR2_DIR, RMFD_DIR  # noqa: E402


# Multiple mirrors per dataset -- the script tries them in order so one
# dead host doesn't break the build. urlretrieve raises URLError on DNS
# failure, which we catch and try the next mirror.
LFW_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "https://conradsanderson.id.au/lfwcrop/lfwcrop_color.zip",  # alternative crop
]
LFW_PAIRS_URLS = [
    "http://vis-www.cs.umass.edu/lfw/pairs.txt",
]
MFR2_URL = "https://github.com/aqeelanwar/MaskTheFace/raw/master/datasets/mfr2.zip"


def _report(block_num: int, block_size: int, total_size: int) -> None:
    """Tiny progress reporter for urlretrieve so the console isn't silent."""
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"  downloaded {downloaded / 1e6:.1f} MB", end="\r")
        return
    pct = min(100.0, 100.0 * downloaded / total_size)
    print(f"  {pct:5.1f}%  ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="\r")


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    urlretrieve(url, dest, reporthook=_report)
    print()
    return dest


def download_first(urls: list[str], dest: Path) -> Path | None:
    """Try a list of mirrors; return on first success or None if all fail."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present")
        return dest
    for url in urls:
        try:
            return download(url, dest)
        except Exception as exc:  # URLError, HTTPError, OSError ...
            print(f"[warn] {url} failed: {exc}")
    return None


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[xtra] {archive.name} -> {out_dir}")
    if archive.suffix in {".tgz", ".gz"} or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        raise ValueError(f"unsupported archive type: {archive}")


def setup_lfw() -> None:
    # If LFW is already extracted (e.g. dropped in manually) we're done.
    if (LFW_DIR / "Aaron_Eckhart").exists():
        print(f"[ok  ] LFW already extracted at {LFW_DIR}")
        return

    archive = DATA_DIR / "lfw.tgz"
    pairs = DATA_DIR / "pairs.txt"
    archive_ok = download_first(LFW_URLS, archive)
    download_first(LFW_PAIRS_URLS, pairs)
    if archive_ok is None:
        print("[err ] could not fetch LFW from any mirror. Drop lfw.tgz into")
        print(f"       {DATA_DIR} manually, or extract LFW into {LFW_DIR}.")
        return
    extract(archive, DATA_DIR)
    extracted = DATA_DIR / "lfw"
    if extracted != LFW_DIR and extracted.exists():
        shutil.move(str(extracted), str(LFW_DIR))
    print(f"[ok  ] LFW ready at {LFW_DIR}")


def setup_mfr2() -> None:
    if any(MFR2_DIR.glob("*.txt")) or any(MFR2_DIR.iterdir() if MFR2_DIR.exists() else []):
        # Already populated -- could be from a prior run or manual placement.
        print(f"[ok  ] MFR2 already present at {MFR2_DIR}")
        return
    # The upstream zip is named arbitrarily; we keep a stable local name.
    archive = DATA_DIR / "mfr2.zip"
    try:
        download(MFR2_URL, archive)
    except Exception as exc:
        print(f"[warn] MFR2 download failed ({exc}).")
        print("       Grab it manually from https://github.com/aqeelanwar/MaskTheFace")
        return
    extract(archive, MFR2_DIR.parent)
    print(f"[ok  ] MFR2 ready at {MFR2_DIR}")


def setup_rmfd() -> None:
    """RMFD has no stable open mirror -- check for manual placement."""
    if not RMFD_DIR.exists() or not any(RMFD_DIR.iterdir()):
        print("[info] RMFD not found.")
        print("       Download AFDB_masked_face_dataset and AFDB_face_dataset from")
        print("       https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset")
        print(f"       and extract them under {RMFD_DIR}.")
        return
    print(f"[ok  ] RMFD present at {RMFD_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download AdaptiveFace datasets.")
    parser.add_argument("--lfw", action="store_true", help="Download LFW + pairs.txt.")
    parser.add_argument("--mfr2", action="store_true", help="Download MFR2 (paired masked/unmasked).")
    parser.add_argument("--rmfd", action="store_true", help="Verify manual RMFD placement.")
    parser.add_argument("--all", action="store_true", help="All of the above.")
    args = parser.parse_args(argv)

    if args.all:
        args.lfw = args.mfr2 = args.rmfd = True
    if not any([args.lfw, args.mfr2, args.rmfd]):
        parser.print_help()
        return 1

    if args.lfw:
        setup_lfw()
    if args.mfr2:
        setup_mfr2()
    if args.rmfd:
        setup_rmfd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
