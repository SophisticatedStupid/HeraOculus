#!/usr/bin/env python3
p = r'D:\Geo-Iris-main\Geo-Iris-main\backend\app\services\gridded_climate_service.py'
with open(p, encoding='utf-8') as f:
    lines = f.readlines()

# Find _cache_get start and _cache_set end
start_get = None
end_set = None
for i, line in enumerate(lines):
    if line.startswith('def _cache_get('):
        start_get = i
        break
if start_get is not None:
    for i in range(start_get + 1, len(lines)):
        if lines[i].startswith('def ') or lines[i].startswith('# ──'):
            end_set = i
            break

print(f'_cache_get starts at line {start_get+1}')
print(f'_cache_set ends before line {end_set+1}')

new_get_set = '''def _cache_get(dataset: str, payload: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Retrieve a cached value.

    Returns the deserialized value: ``bytes`` for raw blobs, ``np.ndarray``
    for cached NumPy arrays, or ``dict``/``list`` for JSON payloads.
    Returns ``None`` on miss, expiry, corruption, or if caching is disabled.
    A corrupt or unreadable cache file is treated as a cache miss (logged at
    warning and deleted), never as a request failure.
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
        logger.warning(
            "Cache file %s is corrupt or old format; deleting.", path
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None

    typ = raw[4:5]
    try:
        if typ == _CACHE_TYPE_RAW:
            return payload
        elif typ == _CACHE_TYPE_NPY:
            return np.load(io.BytesIO(payload), allow_pickle=False)
        elif typ == _CACHE_TYPE_JSON:
            return json.loads(payload.decode("utf-8"))
    except Exception as exc:  # corrupt payload inside valid envelope
        logger.warning(
            "Cache file %s failed to deserialize (%s); deleting.", path, exc
        )
        try:
            path.unlink()
        except OSError:
            pass
    return None


def _cache_set(dataset: str, payload: str, data) -> None:
    """Store a value in the cache atomically.

    Accepts ``bytes`` (raw blob), ``np.ndarray`` (saved via np.save with
    allow_pickle=False), or ``dict``/``list`` (serialized via JSON).
    Write is atomic via a temporary file + os.replace.  After writing, the
    size cap is enforced.  Cache write failures never propagate to the
    caller.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return

    path = cache_dir / _cache_key(dataset, payload)

    try:
        if isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data, allow_pickle=False)
            payload_bytes = buf.getvalue()
            typ = _CACHE_TYPE_NPY
        elif isinstance(data, (dict, list)):
            payload_bytes = json.dumps(data).encode("utf-8")
            typ = _CACHE_TYPE_JSON
        elif isinstance(data, (bytes, bytearray)):
            payload_bytes = bytes(data)
            typ = _CACHE_TYPE_RAW
        else:
            # Fallback: treat as raw bytes via bytes().
            payload_bytes = bytes(data)
            typ = _CACHE_TYPE_RAW

        envelope = _envelope_wrap(typ, payload_bytes)
    except Exception:
        # Serialization failure must not fail the request.
        return

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
        # os.replace is atomic on POSIX and Windows.
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


'''

new_lines = lines[:start_get] + [new_get_set] + lines[end_set:]

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'Done. New line count: {len(new_lines)}')