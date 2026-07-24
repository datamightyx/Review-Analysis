"""Cloudflare R2 sync layer for the review-scoring pipeline.

Streamlit Community Cloud (and similar free hosts) wipe local disk on
redeploy / sleep / restart — but this app writes almost everything to
local files under `products/<line>/` (PDFs, scoring.db, overrides.json,
gate_labels.json, generated .xlsx, ...). This module makes R2 the actual
persistence layer: local disk is treated as a per-process cache.

- A product-line folder is pulled down from R2 once per process, the
  first time it's touched (cached in `_synced_prefixes` so a Streamlit
  rerun doesn't re-hit R2 on every widget interaction).
- Every write path in app.py pushes the changed file(s) back up right
  after writing them locally.

If no `[r2]` section exists in `st.secrets`, every function here is a
no-op — the app then behaves exactly as before (local-disk-only), so R2
stays optional for local development.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - always run under streamlit
    st = None

_client = None
_bucket: str | None = None
_lock = threading.Lock()
_synced_prefixes: set[str] = set()
_synced_files: set[str] = set()


def _config() -> dict | None:
    if st is None:
        return None
    try:
        r2 = st.secrets.get("r2")
    except Exception:
        return None
    return dict(r2) if r2 else None


def enabled() -> bool:
    return _config() is not None


def _get_client():
    global _client, _bucket
    if _client is not None:
        return _client
    cfg = _config()
    if cfg is None:
        return None
    import boto3  # imported lazily so it's optional when R2 isn't configured
    with _lock:
        if _client is None:
            _client = boto3.client(
                "s3",
                endpoint_url=cfg["endpoint_url"],
                aws_access_key_id=cfg["access_key_id"],
                aws_secret_access_key=cfg["secret_access_key"],
                region_name="auto",
            )
            _bucket = cfg["bucket"]
    return _client


def _key(local_path: Path, root: Path) -> str:
    return local_path.resolve().relative_to(root.resolve()).as_posix()


MAX_BUCKET_BYTES = 1_000_000_000  # ~1GB — Cloudflare R2 free-tier storage cap


def _put(local_path: Path, root: Path) -> None:
    client = _get_client()
    if client is None or not local_path.exists():
        return
    try:
        client.upload_file(str(local_path), _bucket, _key(local_path, root))
    except Exception as e:
        # a transient R2 error shouldn't crash the whole Streamlit rerun —
        # the local write already succeeded; log so the desync is at least
        # visible in server logs instead of silently vanishing
        print(f"[r2_sync] upload failed for {local_path}: {e}", file=sys.stderr)


def upload_file(local_path: Path, root: Path) -> int:
    """Push one file to R2 under its path relative to `root`, then enforce
    the bucket-wide size cap. No-op if R2 isn't configured, or the file no
    longer exists (use delete_file for a removed file). Returns how many
    (oldest) objects got evicted to stay under the cap."""
    _put(local_path, root)
    return enforce_retention(root)


def delete_file(local_path: Path, root: Path) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.delete_object(Bucket=_bucket, Key=_key(local_path, root))
    except Exception as e:
        print(f"[r2_sync] delete failed for {local_path}: {e}", file=sys.stderr)


def upload_folder(local_folder: Path, root: Path) -> int:
    """Push every file currently in this folder to R2 — used once after a
    full pipeline run instead of tracking each intermediate write site.
    Retention is enforced once at the end, not per file. Returns how many
    (oldest) objects got evicted to stay under the cap."""
    client = _get_client()
    if client is None:
        return 0
    for p in local_folder.rglob("*"):
        if p.is_file():
            _put(p, root)
    return enforce_retention(root)


def enforce_retention(root: Path, max_bytes: int = MAX_BUCKET_BYTES) -> int:
    """Keep the whole bucket under `max_bytes` by deleting the oldest
    objects first, across every product line, once the cap is exceeded —
    Cloudflare R2's free tier caps storage, so the bucket has to trim
    itself instead of uploads just failing once full. Deletes the local
    (ephemeral) copy too, if still present, so a stale file doesn't linger
    for the rest of this process after its R2 copy is gone. Returns the
    number of objects evicted."""
    client = _get_client()
    if client is None:
        return 0
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket):
        objects.extend(page.get("Contents", []))
    total = sum(o["Size"] for o in objects)
    if total <= max_bytes:
        return 0
    objects.sort(key=lambda o: o["LastModified"])  # oldest first
    evicted = 0
    for obj in objects:
        if total <= max_bytes:
            break
        client.delete_object(Bucket=_bucket, Key=obj["Key"])
        try:
            (root / obj["Key"]).unlink()
        except FileNotFoundError:
            pass
        total -= obj["Size"]
        evicted += 1
    return evicted


def sync_file_down(local_path: Path, root: Path) -> None:
    """Pull a single object down (e.g. the root-level usage-history db)
    before anything opens/creates a local file at that path — sqlite would
    otherwise silently create a fresh empty file on a cold container and
    that empty file would win once a connection is bound to it. Only once
    per process per path."""
    client = _get_client()
    if client is None:
        return
    key = _key(local_path, root)
    with _lock:
        if key in _synced_files:
            return
        _synced_files.add(key)
    if local_path.exists():
        return
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(_bucket, key, str(local_path))
    except Exception:
        pass  # nothing in R2 yet at this key (first run ever) — fine


def sync_folder_down(local_folder: Path, root: Path) -> None:
    """Pull every object under this folder's R2 prefix into the local
    (ephemeral) folder — only once per process per folder. The prefix is
    marked synced only AFTER the download loop finishes without error: if
    a transient failure interrupts it partway through, the prefix stays
    unmarked so a later call in this process retries instead of leaving
    some files permanently missing locally for the rest of the process."""
    client = _get_client()
    if client is None:
        return
    prefix = _key(local_folder, root) + "/"
    with _lock:
        if prefix in _synced_prefixes:
            return
    local_folder.mkdir(parents=True, exist_ok=True)
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                dest = root / key
                if dest.exists() and dest.stat().st_size == obj["Size"]:
                    continue  # already present (warm process, repeated rerun)
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(_bucket, key, str(dest))
    except Exception as e:
        print(f"[r2_sync] sync_folder_down failed for {prefix}: {e}",
              file=sys.stderr)
        return
    with _lock:
        _synced_prefixes.add(prefix)


def delete_folder(local_folder: Path, root: Path) -> None:
    """Delete every R2 object under this folder's prefix — the counterpart
    to upload_folder, used when a product line is deleted so it doesn't
    reappear via sync_folder_down in a later/other session. No-op if R2
    isn't configured."""
    client = _get_client()
    if client is None:
        return
    prefix = _key(local_folder, root) + "/"
    paginator = client.get_paginator("list_objects_v2")
    keys = [obj["Key"] for page in paginator.paginate(Bucket=_bucket, Prefix=prefix)
            for obj in page.get("Contents", [])]
    for i in range(0, len(keys), 1000):  # delete_objects caps at 1000/call
        batch = keys[i:i + 1000]
        client.delete_objects(Bucket=_bucket,
                              Delete={"Objects": [{"Key": k} for k in batch]})
    with _lock:
        _synced_prefixes.discard(prefix)


def list_remote_lines(products_root: Path, root: Path) -> list[str]:
    """Product-line folder names that exist in R2 — needed so a freshly
    booted (locally empty) container still lists them in the sidebar."""
    client = _get_client()
    if client is None:
        return []
    prefix = _key(products_root, root) + "/"
    names: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix):].rstrip("/")
            if name:
                names.add(name)
    return sorted(names)
