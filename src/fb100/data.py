"""Idempotent download of the Facebook100 archive.

Source: the Internet Archive mirror of the original release by Traud, Mucha and Porter
(https://archive.org/details/oxford-2005-facebook-matrix). The archive is checked
against the MD5 published by the Internet Archive before anything is extracted.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://archive.org/download/oxford-2005-facebook-matrix/facebook100.zip"
ARCHIVE_MD5 = "a7687ec2362d369803a3463e2884dbe8"
N_SCHOOLS = 100


def md5sum(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(dest: Path) -> bool:
    schools = [p for p in dest.glob("*.mat") if p.stem != "schools"]
    return len(schools) == N_SCHOOLS


def fetch(dest: str | Path, url: str = ARCHIVE_URL, md5: str = ARCHIVE_MD5) -> Path:
    """Make sure `dest` holds the 100 school files; download and extract only if needed."""
    dest = Path(dest)
    if is_complete(dest):
        print(f"Facebook100 already present in {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "facebook100.zip"
    if not archive.exists() or md5sum(archive) != md5:
        print(f"Downloading {url} (~207 MB)")
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(url) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        partial.replace(archive)
    if (got := md5sum(archive)) != md5:
        raise RuntimeError(f"Checksum mismatch for {archive}: {got} != {md5}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or "__MACOSX" in member.filename or name.startswith("."):
                continue  # macOS resource forks ship in the archive next to the real files
            if not name.endswith((".mat", ".txt")):
                continue
            with zf.open(member) as src, (dest / name).open("wb") as out:
                shutil.copyfileobj(src, out)
    if not is_complete(dest):
        raise RuntimeError(f"{dest} does not contain {N_SCHOOLS} school files after extraction")
    archive.unlink()
    print(f"Extracted Facebook100 to {dest}")
    return dest
