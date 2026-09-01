"""R2 sync guards around SQLite files.

The failure these pin down: `products/<line>/scoring.db-wal` had been uploaded
by `upload_folder` (which shipped every file it found), and pulling it back
down crashed the whole folder sync — Windows keeps `-wal`/`-shm` locked for as
long as a connection is open:

    [r2_sync] sync_folder_down failed for products/blotting/: [WinError 32]
    The process cannot access the file because it is being used by another
    process: '...\\products\\blotting\\scoring.db-wal'

One locked file aborted the whole prefix, so nothing else in the product line
arrived either. Sidecars are now never shipped and never pulled, a live local
.db is never overwritten by a remote one, and a single failed download no
longer takes the rest of the folder with it.
"""
from pathlib import Path

import pytest

from storage import r2_sync


class FakeClient:
    """Just enough of the boto3 S3 client for the two sync paths."""

    def __init__(self, objects: dict[str, int], locked: tuple[str, ...] = ()):
        self.objects = dict(objects)
        self.locked = set(locked)
        self.downloaded: list[str] = []
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    def get_paginator(self, _name):
        objects = self.objects

        class _Pager:
            def paginate(self, **kw):
                prefix = kw.get("Prefix", "")
                yield {"Contents": [
                    {"Key": k, "Size": v, "LastModified": i}
                    for i, (k, v) in enumerate(objects.items())
                    if k.startswith(prefix)]}

        return _Pager()

    def download_file(self, _bucket, key, dest):
        if key in self.locked:
            raise PermissionError(
                32, "The process cannot access the file because it is being "
                    "used by another process")
        Path(dest).write_text("x" * self.objects[key], encoding="utf-8")
        self.downloaded.append(key)

    def upload_file(self, _path, _bucket, key):
        self.uploaded.append(key)

    def delete_object(self, Bucket=None, Key=None):
        self.deleted.append(Key)
        self.objects.pop(Key, None)


@pytest.fixture
def r2(monkeypatch):
    """Install a fake client and hand the test a factory for it."""

    def install(objects, locked=()):
        client = FakeClient(objects, locked)
        monkeypatch.setattr(r2_sync, "_client", client)
        monkeypatch.setattr(r2_sync, "_bucket", "test-bucket")
        monkeypatch.setattr(r2_sync, "_synced_prefixes", set())
        monkeypatch.setattr(r2_sync, "_synced_files", set())
        return client

    return install


def test_is_sidecar():
    assert r2_sync.is_sidecar(Path("products/x/scoring.db-wal"))
    assert r2_sync.is_sidecar(Path("products/x/scoring.db-shm"))
    assert r2_sync.is_sidecar("products/x/scoring.db-journal")
    assert not r2_sync.is_sidecar(Path("products/x/scoring.db"))
    assert not r2_sync.is_sidecar(Path("products/x/overrides.json"))


def test_sidecars_are_never_downloaded(tmp_path, r2):
    client = r2({"products/blotting/scoring.db": 3,
                 "products/blotting/scoring.db-wal": 5,
                 "products/blotting/overrides.json": 2})
    folder = tmp_path / "products" / "blotting"

    r2_sync.sync_folder_down(folder, tmp_path)

    assert client.downloaded == ["products/blotting/scoring.db",
                                 "products/blotting/overrides.json"]
    assert not (folder / "scoring.db-wal").exists()


def test_existing_local_db_is_not_overwritten(tmp_path, r2):
    """The local db is the one this process has open and has been writing to —
    it is ahead of R2, not behind it."""
    client = r2({"products/blotting/scoring.db": 99})
    folder = tmp_path / "products" / "blotting"
    folder.mkdir(parents=True)
    (folder / "scoring.db").write_text("live", encoding="utf-8")

    r2_sync.sync_folder_down(folder, tmp_path)

    assert client.downloaded == []
    assert (folder / "scoring.db").read_text(encoding="utf-8") == "live"


def test_one_locked_file_does_not_abort_the_folder(tmp_path, r2):
    client = r2({"products/blotting/a.json": 1,
                 "products/blotting/locked.xlsx": 2,
                 "products/blotting/b.json": 3},
                locked=("products/blotting/locked.xlsx",))
    folder = tmp_path / "products" / "blotting"

    r2_sync.sync_folder_down(folder, tmp_path)

    assert client.downloaded == ["products/blotting/a.json",
                                 "products/blotting/b.json"]
    # the prefix stays unmarked, so a later call in the same process retries
    assert "products/blotting/" not in r2_sync._synced_prefixes


def test_a_clean_folder_is_marked_synced_once(tmp_path, r2):
    client = r2({"products/blotting/a.json": 1})
    folder = tmp_path / "products" / "blotting"

    r2_sync.sync_folder_down(folder, tmp_path)
    r2_sync.sync_folder_down(folder, tmp_path)

    assert client.downloaded == ["products/blotting/a.json"]
    assert "products/blotting/" in r2_sync._synced_prefixes


def test_upload_folder_skips_sidecars_and_drops_stale_ones(tmp_path, r2):
    client = r2({"products/blotting/scoring.db-wal": 5})
    folder = tmp_path / "products" / "blotting"
    folder.mkdir(parents=True)
    for name in ("scoring.db", "scoring.db-wal", "scoring.db-shm",
                 "overrides.json"):
        (folder / name).write_text("x", encoding="utf-8")

    r2_sync.upload_folder(folder, tmp_path)

    assert sorted(client.uploaded) == ["products/blotting/overrides.json",
                                       "products/blotting/scoring.db"]
    # the sidecar an older build already shipped is cleaned out of the bucket
    assert "products/blotting/scoring.db-wal" in client.deleted


def test_upload_file_refuses_a_sidecar(tmp_path, r2):
    client = r2({})
    folder = tmp_path / "products" / "blotting"
    folder.mkdir(parents=True)
    (folder / "scoring.db-wal").write_text("x", encoding="utf-8")

    r2_sync.upload_file(folder / "scoring.db-wal", tmp_path)

    assert client.uploaded == []
