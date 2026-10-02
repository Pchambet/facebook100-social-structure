import zipfile

import pytest

from fb100.data import N_SCHOOLS, fetch, md5sum


def _archive(tmp_path):
    path = tmp_path / "facebook100.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(N_SCHOOLS):
            zf.writestr(f"facebook100/School{i}.mat", b"x")
            zf.writestr(f"__MACOSX/facebook100/._School{i}.mat", b"fork")
        zf.writestr("facebook100/schools.mat", b"index")
        zf.writestr("facebook100/.DS_Store", b"")
    return path


def test_fetch_extracts_once_and_skips_resource_forks(tmp_path):
    archive = _archive(tmp_path)
    dest = tmp_path / "raw" / "facebook100"
    fetch(dest, url=archive.as_uri(), md5=md5sum(archive))

    names = sorted(p.name for p in dest.iterdir())
    assert len(names) == N_SCHOOLS + 1
    assert not any(n.startswith(".") for n in names)
    assert not (dest.parent / "facebook100.zip").exists()

    archive.unlink()  # a second call must not touch the network or the archive
    fetch(dest, url="file:///nonexistent.zip", md5="0")


def test_fetch_rejects_bad_checksum(tmp_path):
    archive = _archive(tmp_path)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        fetch(tmp_path / "raw" / "facebook100", url=archive.as_uri(), md5="0" * 32)
