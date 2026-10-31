"""Download and verify the GeNIS dataset from Zenodo.

Usage:
    python -m src.data.download --config configs/data.yaml

This downloads every file listed in the Zenodo record's file index to
data/raw/, verifying each against the MD5 checksum Zenodo publishes for it.
It does NOT unzip or preprocess anything -- that is preprocessing.py's job.

GeNIS is large (the full raw PCAPNG set totals 37.6M packets across all
scenarios). This script supports --only-preprocessed to skip the raw PCAPNG
archives and fetch only the preprocessed flow CSVs, which is all this
project's 60-second-interval analysis needs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_yaml, get_repo_root  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

ZENODO_API_URL_TEMPLATE = "https://zenodo.org/api/records/{record_id}"
CHUNK_SIZE = 1 << 20  # 1 MiB


def fetch_record_metadata(record_id: str) -> Dict[str, Any]:
    """Fetch a Zenodo record's metadata, including its file index.

    Args:
        record_id: The numeric Zenodo record id (e.g. "14919237").

    Returns:
        Parsed JSON metadata as returned by the Zenodo REST API.
    """
    url = ZENODO_API_URL_TEMPLATE.format(record_id=record_id)
    logger.info("Fetching Zenodo record metadata from %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None) -> None:
    """Stream-download one file, verifying its checksum if provided.

    If dest already exists and matches expected_md5, the download is
    skipped (idempotent re-runs).

    Args:
        url: Direct download URL for the file.
        dest: Local path to save to.
        expected_md5: MD5 checksum Zenodo reports for this file, or None
            if unavailable (in which case the check is skipped and a
            warning is logged rather than silently passing).
    """
    if dest.exists() and expected_md5:
        if _md5_of_file(dest) == expected_md5:
            logger.info("Already downloaded and verified: %s", dest.name)
            return
        logger.warning("Existing file %s failed checksum; re-downloading.", dest.name)

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

    if expected_md5:
        actual = _md5_of_file(dest)
        if actual != expected_md5:
            raise ValueError(
                f"Checksum mismatch for {dest.name}: expected {expected_md5}, got {actual}. "
                f"The download may be corrupt or the Zenodo record may have changed."
            )
        logger.info("Checksum OK for %s", dest.name)
    else:
        logger.warning(
            "No checksum available from Zenodo for %s -- integrity not verified.",
            dest.name,
        )


def select_files(files: List[Dict[str, Any]], only_preprocessed: bool) -> List[Dict[str, Any]]:
    """Filter the Zenodo file list.

    Args:
        files: The "files" list from the Zenodo record JSON.
        only_preprocessed: If True, keep only files whose key/path suggests
            they belong to the preprocessed flow tables (e.g. containing
            "preprocessed" or ending in .csv), skipping raw PCAPNG archives.

    Returns:
        The filtered file list.
    """
    if not only_preprocessed:
        return files
    keep = []
    for f in files:
        key = f.get("key", "").lower()
        if "preprocess" in key or key.endswith(".csv") or key.endswith(".zip"):
            keep.append(f)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--only-preprocessed",
        action="store_true",
        help="Skip raw PCAPNG archives; download only the preprocessed flow CSVs.",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    record_id = data_cfg["zenodo_record_id"]
    raw_dir = get_repo_root() / data_cfg["raw_data_dir"]

    metadata = fetch_record_metadata(record_id)
    files = metadata.get("files", [])
    if not files:
        raise RuntimeError(
            f"Zenodo record {record_id} returned no files. Check "
            f"{data_cfg['zenodo_url']} manually -- the record may require "
            f"browser-based access or the API response shape may have changed."
        )

    selected = select_files(files, args.only_preprocessed)
    logger.info("%d of %d files selected for download.", len(selected), len(files))

    for f in selected:
        key = f["key"]
        url = f["links"]["self"]
        checksum = f.get("checksum", "")
        expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        download_file(url, raw_dir / key, expected_md5)

    logger.info("Download complete. Files saved under %s", raw_dir)
    logger.info("Next step: python -m src.data.audit --config %s", args.config)


if __name__ == "__main__":
    main()
