"""Save source responses separately from derived metadata tables."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def setup_tls():
    """Use the macOS certificate store, including locally trusted CERN CAs."""
    if sys.platform == "darwin":
        import truststore
        truststore.inject_into_ssl()


def save_response(response, path):
    """Preserve the unmodified HTTP response body; never save login HTML as data."""
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower():
        raise ValueError("Expected a JSON response; check authentication")
    payload = json.loads(response.content)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    record = {
        "file": path.name,
        "url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "bytes": len(response.content),
    }
    return payload, record


def write_source(folder, records, **extra):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    value = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_format": "Unmodified HTTP response bodies, after HTTP decompression",
        "requests": records,
        **extra,
    }
    (folder / "source.json").write_text(json.dumps(value, indent=2) + "\n")
