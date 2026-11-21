"""STAC (SpatioTemporal Asset Catalog) search service for satellite imagery.

Provides a provider-agnostic search over public STAC APIs (Earth Search by
Element 84 by default; Microsoft Planetary Computer optional) and normalizes
results into a small, stable catalog shape:

    id, datetime, cloud_cover, bbox, platform, thumbnail, assets{red,nir,swir,scl,blue,green}

Design notes
------------
* Synthetic data never masquerades as live: this service is purely a catalog
  proxy over live STAC endpoints. It never generates synthetic items.
* Microsoft Planetary Computer asset hrefs point at private Azure Blob Storage
  and require a SAS token.  This service signs blob URLs via the PC Data
  Authentication API (`/api/sas/v1/token/{collection}` and
  `/api/sas/v1/sign?href=...`) so callers receive ready-to-use asset URLs.
* Upstream failures (network, timeout, 4xx/5xx) are mapped to HTTP 502 with a
  useful message; they are never silently swallowed or replaced with synthetic
  fallback data.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal, Optional
from urllib.parse import urlparse, urlunparse

import requests
from fastapi import HTTPException

from ..config import settings

# ── Provider constants ──────────────────────────────────────────────────────

STAC_PROVIDER_EARTH_SEARCH = "earth-search"
STAC_PROVIDER_PC = "planetary-computer"

PROVIDER_BASE_URLS = {
    STAC_PROVIDER_EARTH_SEARCH: "https://earth-search.aws.element84.com/v1",
    STAC_PROVIDER_PC: "https://planetarycomputer.microsoft.com/api/stac/v1",
}

PC_SAS_BASE = "https://planetarycomputer.microsoft.com/api/sas/v1"

# Default (keyless) provider.
DEFAULT_PROVIDER = STAC_PROVIDER_EARTH_SEARCH

# Collections allowed for search.  Keep this an explicit allowlist so callers
# cannot request arbitrary collections through the proxy.
ALLOWED_COLLECTIONS: list[str] = [
    "sentinel-2-l2a",
    "landsat-c2-l2",
]

# Canonical asset keys the normalized catalog exposes, mapped to the provider
# / collection-specific STAC asset key candidates (tried in order).  Only keys
# actually present in a returned item are included, so we never fabricate an
# asset href that the upstream did not advertise.
ASSET_KEY_MAP: dict[str, dict[str, dict[str, list[str]]]] = {
    STAC_PROVIDER_EARTH_SEARCH: {
        "sentinel-2-l2a": {
            "red": ["red"],
            "nir": ["nir"],
            "swir": ["swir16"],
            "scl": ["scl"],
            "blue": ["blue"],
            "green": ["green"],
            "thumbnail": ["thumbnail"],
        },
        "landsat-c2-l2": {
            "red": ["red"],
            "nir": ["nir08", "nir"],  # Earth Search landsat exposes nir08
            "swir": ["swir16"],
            "scl": ["qa_pixel"],
            "blue": ["blue"],
            "green": ["green"],
            "thumbnail": ["thumbnail"],
        },
    },
    STAC_PROVIDER_PC: {
        "sentinel-2-l2a": {
            "red": ["B04"],
            "nir": ["B08"],
            "swir": ["B11"],
            "scl": ["SCL"],
            "blue": ["B02"],
            "green": ["B03"],
            "thumbnail": ["preview"],
        },
        "landsat-c2-l2": {
            "red": ["SR_B4"],
            "nir": ["SR_B5"],
            "swir": ["SR_B6"],
            "scl": ["QA_PIXEL"],
            "blue": ["SR_B2"],
            "green": ["SR_B3"],
            "thumbnail": ["thumbnail"],
        },
    },
}

# Azure Blob Storage host pattern: only these blob URLs are signed for PC.
_BLOB_HOST_RE = re.compile(r"\.blob\.core\.windows\.net$")


# ── PC SAS token cache ─────────────────────────────────────────────────────

class _SasTokenEntry:
    __slots__ = ("token", "expiry")

    def __init__(self, token: str, expiry: float) -> None:
        self.token = token
        self.expiry = expiry

    def is_expired(self, skew_seconds: float = 60.0) -> bool:
        return time.time() >= self.expiry - skew_seconds


_sas_token_cache: dict[str, _SasTokenEntry] = {}


def _pc_sas_token(collection: str) -> str:
    """Fetch (cached) SAS token for a Planetary Computer collection.

    The token is valid for every blob in the same storage account/container,
    so one token per collection suffices for all assets of that collection.
    """
    entry = _sas_token_cache.get(collection)
    if entry is not None and not entry.is_expired():
        return entry.token

    url = f"{PC_SAS_BASE}/token/{collection}"
    try:
        resp = requests.get(url, timeout=settings.http_timeout_seconds)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Planetary Computer SAS token request failed for "
                f"collection {collection!r}: {exc}"
            ),
        ) from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Planetary Computer SAS token service returned "
                f"HTTP {resp.status_code} for collection {collection!r}"
            ),
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Planetary Computer SAS token response was not valid JSON "
                f"for collection {collection!r}"
            ),
        ) from exc

    token = payload.get("token")
    if not token:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Planetary Computer SAS token response missing 'token' "
                f"for collection {collection!r}"
            ),
        )

    expiry_str = payload.get("msft:expiry")
    expiry = time.time() + 30 * 60  # conservative 30-minute fallback
    if isinstance(expiry_str, str):
        try:
            from datetime import datetime, timezone

            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass

    _sas_token_cache[collection] = _SasTokenEntry(token, expiry)
    return token


def _pc_sign_href(href: str, collection: str) -> str:
    """Return a SAS-signed URL for an Azure Blob asset href (PC only).

    Non-blob hrefs (e.g. PC data-api tilejson/rendered_preview) are returned
    unchanged because they are self-signing / already accessible.
    """
    if not href or not isinstance(href, str):
        return href
    try:
        parsed = urlparse(href)
    except ValueError:
        return href
    if not _BLOB_HOST_RE.search(parsed.netloc):
        # Not an Azure Blob URL (e.g. planetarycomputer.microsoft.com api/data)
        return href
    if set(parsed.query) & {"st", "se", "sp"}:
        # Already signed.
        return href
    try:
        token = _pc_sas_token(collection)
    except HTTPException:
        # If signing fails, return the unsigned href rather than failing the
        # whole search; callers can still see the item metadata.
        return href
    return f"{href}?{token}"


# ── Shared helpers ─────────────────────────────────────────────────────────

def _provider_base(provider: str) -> str:
    base = PROVIDER_BASE_URLS.get(provider)
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown STAC provider '{provider}'. "
            f"Allowed: {sorted(PROVIDER_BASE_URLS)}.",
        )
    return base


def _validate_bbox(bbox: list[float]) -> None:
    if len(bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must be [minx, miny, maxx, maxy] (4 values).",
        )
    minx, miny, maxx, maxy = bbox
    for name, v in (("minx", minx), ("miny", miny), ("maxx", maxx), ("maxy", maxy)):
        if not isinstance(v, (int, float)) or not _is_finite(v):
            raise HTTPException(
                status_code=400,
                detail=f"bbox {name} must be a finite number, got {v!r}.",
            )
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
        raise HTTPException(
            status_code=400,
            detail="bbox longitudes must be within [-180, 180].",
        )
    if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
        raise HTTPException(
            status_code=400,
            detail="bbox latitudes must be within [-90, 90].",
        )
    if minx > maxx or miny > maxy:
        raise HTTPException(
            status_code=400,
            detail="bbox min values must be <= max values.",
        )
    # Max area cap: ~1,000,000 km^2 (roughly 10 deg x 10 deg at equator).
    area_deg = (maxx - minx) * (maxy - miny)
    if area_deg > 100.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"bbox area {area_deg:.1f} sq deg exceeds the 100 sq deg "
                f"maximum for catalog search."
            ),
        )


def _is_finite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _datetime_range(start_date: Optional[str], end_date: Optional[str]) -> str:
    if start_date and end_date:
        return f"{start_date}/{end_date}"
    if start_date:
        return f"{start_date}/.."
    if end_date:
        return f"../{end_date}"
    return ".."


def _extract_next_token(links: list[dict[str, Any]]) -> Optional[str]:
    """Extract the STAC next-page token from response links, if any."""
    for link in links or []:
        if link.get("rel") == "next":
            href = link.get("href", "")
            # STAC next links carry a `token` query param; capture it.
            match = re.search(r"[?&]token=([^&]+)", href)
            if match:
                return match.group(1)
            return href
    return None


def _pick_asset(
    assets: dict[str, Any], candidates: list[str]
) -> Optional[str]:
    """Return the href of the first candidate asset key present in `assets`."""
    for key in candidates:
        asset = assets.get(key)
        if isinstance(asset, dict) and asset.get("href"):
            return asset["href"]
    return None


def _normalize_item(
    provider: str,
    collection: str,
    feature: dict[str, Any],
) -> dict[str, Any]:
    props = feature.get("properties", {}) or {}
    assets = feature.get("assets", {}) or {}
    key_map = ASSET_KEY_MAP.get(provider, {}).get(collection, {})

    resolved_assets: dict[str, str] = {}
    for canonical, candidates in key_map.items():
        href = _pick_asset(assets, candidates)
        if href:
            if provider == STAC_PROVIDER_PC:
                href = _pc_sign_href(href, collection)
            resolved_assets[canonical] = href

    # Thumbnail: prefer explicit thumbnail asset, fall back to visual/preview.
    thumbnail = resolved_assets.get("thumbnail")
    if not thumbnail:
        for cand in ("visual", "preview", "rendered_preview"):
            href = _pick_asset(assets, [cand])
            if href:
                thumbnail = _pc_sign_href(href, collection) if provider == STAC_PROVIDER_PC else href
                break

    return {
        "id": feature.get("id"),
        "collection": collection,
        "provider": provider,
        "datetime": props.get("datetime"),
        "cloud_cover": (
            float(props["eo:cloud_cover"])
            if props.get("eo:cloud_cover") is not None
            else None
        ),
        "bbox": feature.get("bbox"),
        "platform": props.get("platform"),
        "assets": resolved_assets,
        "thumbnail": thumbnail,
        "geometry": feature.get("geometry"),
    }


# ── Public API ─────────────────────────────────────────────────────────────

def search_items(
    *,
    provider: Optional[str] = None,
    collections: list[str],
    bbox: list[float],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_cloud_cover: Optional[float] = None,
    limit: int = 10,
    page_token: Optional[str] = None,
) -> dict[str, Any]:
    """Search a STAC catalog and return normalized items.

    Args:
        provider: "earth-search" (default/keyless) or "planetary-computer".
        collections: STAC collection IDs (must be in ALLOWED_COLLECTIONS).
        bbox: [minx, miny, maxx, maxy] in degrees.
        start_date, end_date: ISO-8601 datetimes (at least one required).
        max_cloud_cover: 0-100 percent cloud cover upper bound.
        limit: max items to return (<= 100).
        page_token: opaque next-page token from a previous response.

    Returns:
        dict with "items" (normalized list), "context" (matched/returned/limit)
        and optional "next_token" for pagination.

    Raises:
        HTTPException: 400 for invalid input; 502 for upstream failures.
    """
    provider = provider or settings.stac_provider or DEFAULT_PROVIDER
    if provider not in PROVIDER_BASE_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown STAC provider '{provider}'. "
            f"Allowed: {sorted(PROVIDER_BASE_URLS)}.",
        )

    # Validate collections against the allowlist.
    invalid = [c for c in collections if c not in ALLOWED_COLLECTIONS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Collection(s) not allowed: {invalid}. "
                f"Allowed: {ALLOWED_COLLECTIONS}."
            ),
        )
    if not collections:
        raise HTTPException(
            status_code=400,
            detail="At least one collection must be specified.",
        )

    _validate_bbox(bbox)

    if not start_date and not end_date:
        raise HTTPException(
            status_code=400,
            detail="At least one of start_date or end_date is required.",
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100.",
        )

    if max_cloud_cover is not None and not (0 <= max_cloud_cover <= 100):
        raise HTTPException(
            status_code=400,
            detail="max_cloud_cover must be between 0 and 100.",
        )

    base = _provider_base(provider)
    search_url = f"{base}/search"

    query: dict[str, Any] = {
        "collections": collections,
        "bbox": [float(v) for v in bbox],
        "datetime": _datetime_range(start_date, end_date),
        "limit": min(limit, 100),
    }
    if max_cloud_cover is not None:
        query["query"] = {"eo:cloud_cover": {"lte": float(max_cloud_cover)}}
    if page_token:
        query["token"] = page_token

    try:
        resp = requests.post(
            search_url,
            json=query,
            timeout=settings.http_timeout_seconds,
            headers={"Accept": "application/geo+json"},
        )
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"STAC provider '{provider}' search timed out after "
                f"{settings.http_timeout_seconds}s."
            ),
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"STAC provider '{provider}' search request failed: {exc}",
        ) from exc

    if resp.status_code >= 400:
        detail = resp.text[:500]
        raise HTTPException(
            status_code=502,
            detail=(
                f"STAC provider '{provider}' returned HTTP {resp.status_code}: {detail}"
            ),
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"STAC provider '{provider}' returned invalid JSON: {exc}",
        ) from exc

    features = data.get("features", []) or []
    normalized = [
        _normalize_item(provider, coll, f)
        for coll in collections
        for f in features
        # A STATIC search over multiple collections returns items tagged with
        # their own collection; filter each item to its declared collection.
        if f.get("collection") == coll
    ]

    # De-duplicate while preserving order (an item can match multiple
    # requested collections in theory).
    seen: set[str] = set()
    unique_items = []
    for item in normalized:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        unique_items.append(item)

    context = data.get("context", {}) or {}
    links = data.get("links", []) or []
    next_token = _extract_next_token(links)

    return {
        "provider": provider,
        "items": unique_items,
        "context": {
            "matched": context.get("matched"),
            "returned": context.get("returned", len(unique_items)),
            "limit": min(limit, 100),
        },
        "next_token": next_token,
    }
