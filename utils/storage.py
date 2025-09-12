"""Unified storage abstraction: local filesystem for dev, S3 for production.

Usage:
    from utils.storage import storage
    storage.write_bytes("uploads/abc.csv", b"data")
    data = storage.read_bytes("uploads/abc.csv")

The implementation is intentionally minimal – enough for the first integration slice
(uploads route). Extend as needed for listing, deleting, etc.
"""
from __future__ import annotations

import io
import os
import pathlib
from typing import Optional, List, Tuple

from utils.config import USE_S3, S3_BUCKET

try:
    import boto3  # type: ignore
except ImportError:  # boto3 is optional for local dev
    boto3 = None  # type: ignore


class Storage:
    """Abstract base storage."""

    def write_bytes(self, key: str, data: bytes) -> None:  # pragma: no cover – interface
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:  # pragma: no cover – interface
        raise NotImplementedError

    def exists(self, key: str) -> bool:  # pragma: no cover – interface
        raise NotImplementedError
    
    def list_keys(self, prefix: str = "", suffix: Optional[str] = None) -> List[Tuple[str, float]]:  # (key, mtime)
        """List keys under a prefix with optional suffix filter. Returns (key, modified_time_epoch)."""
        raise NotImplementedError

    # Convenience helpers -------------------------------------------------

    def write_stream(self, key: str, stream: io.BufferedReader) -> None:
        self.write_bytes(key, stream.read())


class LocalStorage(Storage):
    """Simple wrapper over the local filesystem (project root)."""

    def __init__(self, base_path: str = ".") -> None:
        self.base_path = pathlib.Path(base_path).resolve()

    # Helpers -------------------------------------------------------------

    def _full(self, key: str) -> pathlib.Path:
        return self.base_path / key

    def _ensure_parent(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Storage API ---------------------------------------------------------

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._full(key)
        self._ensure_parent(path)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        path = self._full(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._full(key).exists()

    def list_keys(self, prefix: str = "", suffix: Optional[str] = None) -> List[Tuple[str, float]]:
        base = self.base_path / prefix if prefix else self.base_path
        results: List[Tuple[str, float]] = []
        if base.is_dir():
            for entry in base.iterdir():
                if entry.is_file():
                    name = (pathlib.Path(prefix) / entry.name).as_posix() if prefix else entry.name
                    if suffix and not name.endswith(suffix):
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    results.append((name, mtime))
        return results


class S3Storage(Storage):
    """AWS S3-backed storage."""

    def __init__(self, bucket: str):
        if boto3 is None:
            raise RuntimeError("boto3 not installed; cannot use S3Storage")
        self.bucket = bucket
        self._client = boto3.client("s3")

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def list_keys(self, prefix: str = "", suffix: Optional[str] = None) -> List[Tuple[str, float]]:
        paginator = self._client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": self.bucket}
        if prefix:
            kwargs["Prefix"] = prefix
        results: List[Tuple[str, float]] = []
        for page in paginator.paginate(**kwargs):
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                lm = obj.get("LastModified")
                mtime = lm.timestamp() if lm is not None else 0.0
                results.append((key, mtime))
        return results


# Singleton instance ------------------------------------------------------

if USE_S3:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET must be configured when USE_S3 is true")
    storage: Storage = S3Storage(S3_BUCKET)
else:
    storage = LocalStorage()

__all__ = ["storage", "Storage"]
