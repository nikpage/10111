#!/usr/bin/env python3
"""Batch ingest for GitHub Actions / scheduled runs.

Reads `scope.json`, walks OneDrive for any images under the configured URLs,
embeds new ones with CLIP, and upserts to Supabase. Already-ingested files
are skipped by content hash.

Required env vars (set as GitHub repo secrets):
  SUPABASE_URL
  SUPABASE_SECRET_KEY
  AZURE_TENANT_ID
  AZURE_CLIENT_ID
  AZURE_CLIENT_SECRET
  ONEDRIVE_REFRESH_TOKEN  (one-time: obtained by running app locally + signing in)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np

import housekeep
import ingest
import review
from paths import TOKEN_FILE
from web import onedrive


def _ensure_onedrive_tokens():
    """If onedrive_tokens.json doesn't exist, materialize it from ONEDRIVE_REFRESH_TOKEN."""
    if TOKEN_FILE.exists():
        return
    refresh = os.environ.get("ONEDRIVE_REFRESH_TOKEN")
    if not refresh:
        print(
            "ERROR: no OneDrive tokens file and ONEDRIVE_REFRESH_TOKEN env var is unset.",
            file=sys.stderr,
        )
        sys.exit(1)
    tokens = onedrive._refresh_tokens(refresh)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(tokens))
    print(f"[{datetime.utcnow().isoformat()}] OneDrive tokens initialized from env.")


def _load_scope() -> dict:
    scope_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scope.json")
    if not scope_path.exists():
        print(f"ERROR: scope file not found: {scope_path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(scope_path.read_text())


def _embed_one(record: dict, file_hash: str, infer_request, supabase) -> dict:
    """Download bytes from OneDrive, embed, upsert. Returns {status, ...}."""
    try:
        data = onedrive.fetch_bytes(record)
        pixel_values = ingest.preprocess_image(data)
        infer_request.infer({0: pixel_values})
        embedding = infer_request.get_output_tensor(0).data[0].copy()
        embedding = embedding / np.linalg.norm(embedding)

        meta = {"filename": record.get("filename") or Path(record["path"]).name}
        dims = record.get("dimensions")
        if dims:
            meta["width"], meta["height"] = dims[0], dims[1]
        if record.get("web_url"):
            meta["web_url"] = record["web_url"]
        if record.get("source_url"):
            meta["source_url"] = record["source_url"]
        if record.get("item_id"):
            meta["item_id"] = record["item_id"]
        if record.get("drive_id"):
            meta["drive_id"] = record["drive_id"]
        meta["onedrive_path"] = record["path"]
        exif = record.get("exif") or {}
        if exif.get("date_taken"):
            meta["date_taken"] = exif["date_taken"]
        if exif.get("camera"):
            meta["camera"] = exif["camera"]
        if exif.get("gps"):
            meta["gps"] = exif["gps"]

        row = {
            "content_hash": file_hash,
            "file_path": record.get("web_url") or record["path"],
            "source_type": "onedrive",
            "media_type": "image",
            "embedding": embedding.tolist(),
            "metadata": meta,
        }
        if exif.get("date_taken"):
            try:
                dt = datetime.strptime(exif["date_taken"], "%Y:%m:%d %H:%M:%S")
                row["created_at"] = dt.isoformat()
            except (ValueError, TypeError):
                pass
        supabase.table("media").upsert(row, on_conflict="content_hash").execute()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    scope = _load_scope()
    urls = [u for u in scope.get("onedrive_urls", []) if isinstance(u, str) and u.strip()]
    if not urls:
        print("scope.json has no onedrive_urls. Nothing to do.")
        return

    _ensure_onedrive_tokens()

    print(f"Walking {len(urls)} OneDrive URL(s)...")
    manifest = onedrive.list_images_from_urls(urls)
    print(f"Found {len(manifest)} items.")
    if not manifest:
        return

    print("Running housekeeping (hash + dedup against Supabase)...")
    results = housekeep.run_housekeeping(
        manifest, thresholds=None, fetch_bytes=onedrive.fetch_bytes
    )

    approved = review.get_approved_files(results, decisions={})
    skipped = len(results) - len(approved)
    print(f"Approved: {len(approved)}, skipped/rejected: {skipped}.")
    if not approved:
        return

    print("Loading CLIP model...")
    infer_request, supabase = ingest.load_model()

    hash_lookup = {r["path"]: r["sha256"] for r in results}
    record_lookup = {r["path"]: r for r in results}

    start = time.monotonic()
    embedded = 0
    errors = 0
    for i, path in enumerate(approved, 1):
        rec = record_lookup.get(path)
        file_hash = hash_lookup.get(path)
        if not rec or not file_hash:
            continue
        outcome = _embed_one(rec, file_hash, infer_request, supabase)
        if outcome["status"] == "ok":
            embedded += 1
        else:
            errors += 1
            print(f"  [error] {path}: {outcome['error']}", file=sys.stderr)
        if i % 25 == 0 or i == len(approved):
            elapsed = time.monotonic() - start
            print(f"  {i}/{len(approved)} done ({embedded} embedded, {errors} errors, {elapsed:.0f}s)")

    print(f"Finished. Embedded {embedded}, errors {errors}.")


if __name__ == "__main__":
    main()
