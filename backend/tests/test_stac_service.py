"""Tests for the STAC catalog search service (upstream mocked, no live network).

Covers:
  * normal Earth Search search -> normalized items
  * empty results
  * upstream 500 -> HTTP 502
  * malformed bbox -> HTTP 400
  * pagination next_token extraction
  * Planetary Computer SAS signing (mocked token endpoint)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.services import stac_service as svc


class _FakeResponse:
    def __init__(self, status_code: int, json_data: Any, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self) -> Any:
        return self._json


def _set_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """Replace the frozen Settings with a stub exposing stac_provider + timeout."""
    monkeypatch.setattr(
        svc,
        "settings",
        SimpleNamespace(stac_provider=provider, http_timeout_seconds=25.0),
    )


def _make_item(
    item_id: str,
    collection: str,
    hrefs: dict[str, str],
    datetime_: str = "2024-06-07T19:04:26.934000Z",
    cloud: float = 12.5,
) -> dict[str, Any]:
    assets = {k: {"href": v} for k, v in hrefs.items()}
    return {
        "id": item_id,
        "collection": collection,
        "properties": {
            "datetime": datetime_,
            "eo:cloud_cover": cloud,
            "platform": "sentinel-2b",
        },
        "bbox": [-122.5, 37.0, -122.0, 37.5],
        "geometry": {"type": "Point", "coordinates": [-122.25, 37.25]},
        "assets": assets,
        "links": [],
    }


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(svc.requests, "post", MagicMock(return_value=response))


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(svc.requests, "get", lambda *a, **k: response)


def test_search_items_normal_earth_search(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "earth-search")
    item = _make_item(
        "S2B_10SEF_20240607_0_L2A",
        "sentinel-2-l2a",
        {
            "red": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B04.tif",
            "nir": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B08.tif",
            "swir16": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B11.tif",
            "scl": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../SCL.tif",
            "blue": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B02.tif",
            "green": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B03.tif",
            "thumbnail": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../thumbnail.jpg",
        },
    )
    _patch_post(
        monkeypatch,
        _FakeResponse(
            200,
            {
                "context": {"matched": 1, "returned": 1, "limit": 10},
                "features": [item],
                "links": [],
            },
        ),
    )

    result = svc.search_items(
        collections=["sentinel-2-l2a"],
        bbox=[-122.5, 37.0, -122.0, 37.5],
        start_date="2024-06-01T00:00:00Z",
        end_date="2024-06-10T23:59:59Z",
        max_cloud_cover=30,
        limit=10,
    )

    assert result["provider"] == "earth-search"
    assert len(result["items"]) == 1
    it = result["items"][0]
    assert it["id"] == "S2B_10SEF_20240607_0_L2A"
    assert it["collection"] == "sentinel-2-l2a"
    assert it["cloud_cover"] == 12.5
    assert it["platform"] == "sentinel-2b"
    assert set(it["assets"].keys()) == {"red", "nir", "swir", "scl", "blue", "green", "thumbnail"}
    assert it["thumbnail"].endswith("thumbnail.jpg")
    # The upstream request used the earth-search base and eo:cloud_cover query.
    call = svc.requests.post.call_args
    assert call.kwargs["json"]["collections"] == ["sentinel-2-l2a"]
    assert call.kwargs["json"]["query"]["eo:cloud_cover"]["lte"] == 30


def test_search_items_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "earth-search")
    _patch_post(
        monkeypatch,
        _FakeResponse(200, {"context": {"matched": 0}, "features": [], "links": []}),
    )
    result = svc.search_items(
        collections=["landsat-c2-l2"],
        bbox=[-122.5, 37.0, -122.0, 37.5],
        start_date="2024-06-01T00:00:00Z",
    )
    assert result["items"] == []
    assert result["context"]["returned"] == 0


def test_search_items_upstream_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "earth-search")
    _patch_post(
        monkeypatch,
        _FakeResponse(500, {}, text="Internal Server Error"),
    )
    with pytest.raises(HTTPException) as exc:
        svc.search_items(
            collections=["sentinel-2-l2a"],
            bbox=[-122.5, 37.0, -122.0, 37.5],
            start_date="2024-06-01T00:00:00Z",
        )
    assert exc.value.status_code == 502
    assert "500" in str(exc.value.detail)


def test_search_items_malformed_bbox() -> None:
    # bbox area too large
    with pytest.raises(HTTPException) as exc:
        svc.search_items(
            collections=["sentinel-2-l2a"],
            bbox=[-180, -90, 180, 90],
            start_date="2024-06-01T00:00:00Z",
        )
    assert exc.value.status_code == 400
    assert "area" in str(exc.value.detail).lower()

    # inverted box
    with pytest.raises(HTTPException) as exc:
        svc.search_items(
            collections=["sentinel-2-l2a"],
            bbox=[-122.0, 37.5, -122.5, 37.0],
            start_date="2024-06-01T00:00:00Z",
        )
    assert exc.value.status_code == 400


def test_search_items_pagination_next_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "earth-search")
    _patch_post(
        monkeypatch,
        _FakeResponse(
            200,
            {
                "context": {"matched": 25, "returned": 10, "limit": 10},
                "features": [],
                "links": [
                    {
                        "rel": "next",
                        "href": "https://earth-search.aws.element84.com/v1/search?token=abc123&limit=10",
                    }
                ],
            },
        ),
    )
    result = svc.search_items(
        collections=["sentinel-2-l2a"],
        bbox=[-122.5, 37.0, -122.0, 37.5],
        start_date="2024-06-01T00:00:00Z",
        limit=10,
    )
    assert result["next_token"] == "abc123"
    assert result["context"]["matched"] == 25


def test_search_items_passthrough_page_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "earth-search")
    captured = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _FakeResponse(200, {"context": {}, "features": [], "links": []})

    monkeypatch.setattr(svc.requests, "post", fake_post)
    svc.search_items(
        collections=["sentinel-2-l2a"],
        bbox=[-122.5, 37.0, -122.0, 37.5],
        start_date="2024-06-01T00:00:00Z",
        page_token="tokXYZ",
    )
    assert captured["json"].get("token") == "tokXYZ"


def test_search_items_pc_sas_signing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "planetary-computer")
    # Clear SAS cache so the token endpoint is hit.
    svc._sas_token_cache.clear()

    token_resp = _FakeResponse(
        200,
        {"token": "se=2024-06-08T00%3A00%3A00Z&sp=rl", "msft:expiry": "2024-06-08T00:00:00Z"},
    )
    _patch_get(monkeypatch, token_resp)

    item = _make_item(
        "S2B_MSIL2A_20240607T184919_R113_T10SEG",
        "sentinel-2-l2a",
        {
            "red": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/.../B04.tif",
            "nir": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/.../B08.tif",
            "preview": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/.../preview.tif",
        },
    )
    _patch_post(
        monkeypatch,
        _FakeResponse(200, {"context": {}, "features": [item], "links": []}),
    )

    result = svc.search_items(
        provider="planetary-computer",
        collections=["sentinel-2-l2a"],
        bbox=[-122.5, 37.0, -122.0, 37.5],
        start_date="2024-06-01T00:00:00Z",
    )
    assert len(result["items"]) == 1
    it = result["items"][0]
    # All blob asset hrefs should be signed with the SAS token.
    for canonical, href in it["assets"].items():
        assert "se=2024-06-08T00%3A00%3A00Z" in href, f"{canonical} not signed: {href}"
    assert "se=2024-06-08T00%3A00%3A00Z" in it["thumbnail"]
