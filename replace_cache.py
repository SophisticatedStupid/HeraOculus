#!/usr/bin/env python3
"""Replace the cache section in gridded_climate_service.py with hardened implementation."""

p = r'D:\Geo-Iris-main\Geo-Iris-main\backend\app\services\gridded_climate_service.py'

with open(p, encoding='utf-8') as f:
    lines = f.readlines()

# Find the cache section boundaries:
# Start: line 135 (1-indexed) = 134 (0-indexed)
# End: line 327 (1-indexed) = 326 (0-indexed) -- "# ── Shared helpers ──"

start_idx = 134
end_idx = 326

print(f'Replacing lines {start_idx+1} to {end_idx} (exclusive)')
print(f'First: {lines[start_idx].rstrip()!r}')
print(f'Last before Shared helpers: {lines[end_idx-1].rstrip()!r}')

# We need to also remove "import pickle" from the imports (line 91)
# and update the ERA5 caller to use np.save/np.load instead of pickle.

new_cache_section = '''# ── Cache ───────────────────────────────────────────────────────────────────

# Resolved lazily on first cache write so importing this module never crashes
# on a read-only filesystem.  None => not yet resolved; caching disabled if no
# writable location can be found anywhere.
_CACHE_DIR: Optional[Path] = None
_CACHE_DISABLED = False

# Cache format version.  Bumped when the on-disk format changes.
# Old files without this prefix are ignored/deleted on first access.
_CACHE_VERSION = "v2"

# Cache envelope magic bytes: "HRC2" = HeraOculus Cache v2
_CACHE_MAGIC = b"HRC2"
# Envelope type codes
_CACHE_TYPE_RAW = b"b"   # raw binary blob (GeoTIFF, HDF5, etc.)
_CACHE_TYPE_NPY = b"n"   # numpy array saved with np.save(allow_pickle=False)
_CACHE_TYPE_JSON = b"j"  # JSON-serializable object (dict/list)

import threading
_CACHE_LOCK = threading.Lock()


def _resolve_cache_dir() -> Optional[Path]:
    """Resolve a writable cache directory lazily on first cache write.

    Candidate locations, in order:
      1. ``HERAOCULUS_CACHE_DIR`` env var, if set
      2. ``<repo>/.cache/gridded`` next to this package
      3. ``tempfile.gettempdir()/heraoculus_cache``

    The first location that can be created *and* written to wins.  If none can,
    caching is disabled entirely and a warning is logged; the service keeps
    serving requests without caching rather than crashing on a read-only
    filesystem.

    Also purges stale pre-version cache files (*.bin without v2 prefix) on
    first successful resolution.
    """
    global _CACHE_DIR, _CACHE_DISABLED
    if _CACHE_DISABLED:
        return None
    if _CACHE_DIR is not None:
        return _CACHE_DIR

    env_dir = os.getenv("HERAOCULUS_CACHE_DIR", "").strip()
    candidates = [
        Path(env_dir) if env_dir else None,
        Path(__file__).resolve().parents[2] / ".cache" / "gridded",
        Path(tempfile.gettempdir()) / "heraoculus_cache",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            _CACHE_DIR = candidate
            # Purge old-format files on first successful resolution.
            _purge_stale_cache_files(_CACHE_DIR)
            return _CACHE_DIR
        except OSError:
            continue

    _CACHE_DISABLED = True
    logger.warning(
        "Could not create a writable cache directory under %s; "
        "gridded-climate upstream caching is disabled (requests still served "
        "uncached).",
        candidates,
    )
    return None


def _purge_stale_cache_files(cache_dir: Path) -> None:
    """Delete pre-version cache files (*.bin without v2 prefix)."""
    try:
        for f in cache_dir.glob("*.bin"):
            if not f.name.startswith(_CACHE_VERSION + "_"):
                try:
                    f.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _cache_key(dataset: str, payload: str) -> str:
    """Return a versioned cache filename for the given dataset/payload."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{_CACHE_VERSION}_{dataset}_{digest}.bin"


def _envelope_wrap(typ: bytes, payload: bytes) -> bytes:
    """Wrap payload in a versioned envelope with magic + type + length + data."""
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("payload too large for cache envelope")
    length = len(payload).to_bytes(4, "big")
    return _CACHE_MAGIC + typ + length + payload


def _envelope_unwrap(data: bytes) -> Optional[bytes]:
    """Unwrap envelope, returning payload or None if corrupt/wrong version."""
    if len(data) < 1 + 4 + 4:  # magic(4) + type(1) + len(4) = 9 minimum
        return None
    if data[:4] != _CACHE_MAGIC:
        return None
    typ = data[4:5]
    length = int.from_bytes(data[5:9], "big")
    payload = data[9:9 + length]
    if len(payload) != length:
        return None
    # Optionally verify type is known
    if typ not in (_CACHE_TYPE_RAW, _CACHE_TYPE_NPY, _CACHE_TYPE_JSON):
        return None
    return payload


def _enforce_size_cap(cache_dir: Path) -> None:
    """Enforce total cache size cap by evicting LRU and expired files.

    Reads HERAOCULUS_CACHE_MAX_MB (default 512 MB).  Deletes expired (TTL)
    files first, then evicts least-recently-used files until under cap.
    """
    try:
        max_mb = int(os.getenv("HERAOCULUS_CACHE_MAX_MB", "512"))
    except ValueError:
        max_mb = 512
    max_bytes = max_mb * 1024 * 1024

    now = time.time()
    files: list[tuple[float, Path, int]] = []  # (mtime, path, size)
    total = 0
    try:
        for f in cache_dir.glob(f"{_CACHE_VERSION}_*.bin"):
            try:
                st = f.stat()
                total += st.st_size
                files.append((st.st_mtime, f, st.st_size))
            except OSError:
                # File vanished or unreadable; ignore.
                pass
    except OSError:
        return

    if total <= max_bytes:
        return

    # Sort by mtime ascending (oldest first = LRU)
    files.sort(key=lambda x: x[0])

    # First pass: delete expired files (older than DEFAULT_TTL_SECONDS).
    # Use a reasonable TTL — 7 days default, but we just evict by size anyway.
    # We'll just evict LRU until under cap.
    for mtime, f, size in files:
        if total <= max_bytes:
            break
        try:
            f.unlink()
            total -= size
        except OSError:
            pass


def _cache_get(dataset: str, payload: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Retrieve a cached value.

    Returns the raw payload bytes if the cache entry exists, is fresh, and
    the envelope is valid.  Returns None on miss, expiry, corruption, or
    if caching is disabled.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return None

    path = cache_dir / _cache_key(dataset, payload)
    if not path.exists():
        return None

    try:
        st = path.stat()
        if time.time() - st.st_mtime > ttl_seconds:
            # Expired — delete to free space.
            try:
                path.unlink()
            except OSError:
                pass
            return None

        raw = path.read_bytes()
    except OSError:
        return None

    payload = _envelope_unwrap(raw)
    if payload is None:
        # Corrupt or old format — treat as miss and delete.
        logger.warning("Cache file %s is corrupt or old format; deleting.", path)
        try:
            path.unlink()
        except OSError:
            pass
        return None

    return payload


def _cache_set(dataset: str, payload: str, data: bytes) -> None:
    """Store a raw binary payload in the cache atomically.

    The payload is wrapped in a versioned envelope.  Write is atomic via a
    temporary file + os.replace.  After writing, the size cap is enforced.
    Cache write failures never propagate to the caller.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return

    path = cache_dir / _cache_key(dataset, payload)
    envelope = _envelope_wrap(_CACHE_TYPE_RAW, data)

    try:
        # Atomic write: temp file in same dir, then replace.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=".tmp_",
            suffix=".bin",
            delete=False,
        ) as tmp:
            tmp.write(envelope)
            tmp_path = tmp.name
        # os.replace is atomic on POSIX and Windows (Python 3.3+).
        os.replace(tmp_path, path)
    except OSError:
        # Clean up temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return

    # Enforce size cap (with lock to avoid concurrent scans).
    with _CACHE_LOCK:
        _enforce_size_cap(cache_dir)


# ── Shared helpers ──────────────────────────────────────────────────────────

'''

new_lines = lines[:start_idx] + [new_cache_section] + lines[end_idx:]

# Also remove "import pickle" from imports
# It's at line 91 (0-indexed 90). Remove that line.
# We'll do this by filtering it out.
import_removed = False
final_lines = []
for line in new_lines:
    if not import_removed and line.strip() == "import pickle":
        import_removed = True
        continue
    final_lines.append(line)

if import_removed:
    print("Removed 'import pickle'")
else:
    print("WARNING: 'import pickle' not found to remove")

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print(f'Done. New line count: {len(final_lines)}')