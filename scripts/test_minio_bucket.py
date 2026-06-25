#!/usr/bin/env python3
"""Check whether a MinIO bucket can be reached and used."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime, timezone
from io import BytesIO

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print(
        "Missing dependency: minio\n"
        "Install it with: python -m pip install minio",
        file=sys.stderr,
    )
    sys.exit(2)


DEFAULT_ENDPOINT = "192.168.26.22:9000"
DEFAULT_ACCESS_KEY = "admin"
DEFAULT_SECRET_KEY = "XCtV6vNm7Li6ph1m"
DEFAULT_BUCKET = "ai-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test network, auth, bucket visibility, and object I/O for MinIO."
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MINIO_ENDPOINT", DEFAULT_ENDPOINT),
        help="MinIO endpoint in host:port form. Defaults to MINIO_ENDPOINT or %(default)s.",
    )
    parser.add_argument(
        "--access-key",
        default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_ACCESS_KEY),
        help="MinIO access key. Defaults to MINIO_ACCESS_KEY.",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("MINIO_SECRET_KEY", DEFAULT_SECRET_KEY),
        help="MinIO secret key. Defaults to MINIO_SECRET_KEY.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("MINIO_BUCKET", DEFAULT_BUCKET),
        help="Bucket to test. Defaults to MINIO_BUCKET or %(default)s.",
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        default=os.getenv("MINIO_SECURE", "").lower() in {"1", "true", "yes"},
        help="Use HTTPS instead of HTTP.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Only test connection and bucket visibility; do not write an object.",
    )
    return parser.parse_args()


def check_tcp(endpoint: str, timeout_seconds: float = 5.0) -> None:
    host, sep, port_text = endpoint.rpartition(":")
    if not sep:
        raise ValueError("endpoint must include a port, for example 192.168.26.22:9000")

    print(f"TCP check: connecting to {host}:{port_text} ...")
    with socket.create_connection((host, int(port_text)), timeout=timeout_seconds):
        print("TCP check: OK")


def check_bucket(client: Minio, bucket: str) -> None:
    print("S3 API check: listing buckets ...")
    buckets = client.list_buckets()
    names = sorted(item.name for item in buckets)
    print(f"S3 API check: OK, visible buckets: {', '.join(names) or '(none)'}")

    print(f"Bucket check: looking for {bucket!r} ...")
    if not client.bucket_exists(bucket):
        raise RuntimeError(f"bucket {bucket!r} does not exist or is not visible")
    print("Bucket check: OK")


def check_object_round_trip(client: Minio, bucket: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"codex-minio-connectivity-test/{timestamp}.txt"
    content = f"MinIO connectivity test at {timestamp}\n".encode("utf-8")

    print(f"Write check: uploading {object_name!r} ...")
    client.put_object(
        bucket,
        object_name,
        BytesIO(content),
        length=len(content),
        content_type="text/plain",
    )
    print("Write check: OK")

    print("Read check: downloading uploaded object ...")
    response = client.get_object(bucket, object_name)
    try:
        downloaded = response.read()
    finally:
        response.close()
        response.release_conn()

    if downloaded != content:
        raise RuntimeError("downloaded content did not match uploaded content")
    print("Read check: OK")

    print("Cleanup check: removing uploaded object ...")
    client.remove_object(bucket, object_name)
    print("Cleanup check: OK")


def main() -> int:
    args = parse_args()
    scheme = "https" if args.secure else "http"
    print(f"Testing MinIO bucket {args.bucket!r} at {scheme}://{args.endpoint}")

    try:
        check_tcp(args.endpoint)
        client = Minio(
            args.endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
            secure=args.secure,
        )
        check_bucket(client, args.bucket)
        if args.skip_write:
            print("Write/read check: skipped")
        else:
            check_object_round_trip(client, args.bucket)
    except (OSError, ValueError, RuntimeError, S3Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print("SUCCESS: MinIO bucket is reachable and usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
