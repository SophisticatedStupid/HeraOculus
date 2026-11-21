/**
 * STAC Imagery Catalog Connector — satellite scene discovery (Sentinel-2, Landsat).
 *
 * Implements EarthDataConnector as a catalog/metadata-only provider: it searches
 * public STAC APIs (Earth Search by Element 84 by default; Microsoft Planetary
 * Computer optional) and returns normalized scene metadata + Cloud-Optimized
 * GeoTIFF asset URLs.  It produces no time series and supports no variables for
 * time-series selection — consistent with the CMR connector pattern.
 *
 * Synthetic data never masquerades as live: every item originates from a live
 * STAC endpoint, and empty results are reported explicitly rather than
 * fabricated.
 */

import type {
  ConnectorCapabilities,
  ConnectorFetchOptions,
  ConnectorFetchResult,
  EarthDataConnector,
  VariableId,
} from './types';
import { globalCoverage } from './types';

const STAC_API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';
const STAC_SEARCH_URL = `${STAC_API_BASE}/stac/search`;

// Default (keyless) STAC provider.  The backend proxies Earth Search and, when
// STAC_PROVIDER is set, Microsoft Planetary Computer (with SAS-signed assets).
const DEFAULT_COLLECTIONS = ['sentinel-2-l2a', 'landsat-c2-l2'];

// Canonical band keys the UI displays.  The backend already resolves these to
// provider/collection-specific STAC asset keys; we simply pass through hrefs.
export type StacBandKey = 'red' | 'nir' | 'swir' | 'scl' | 'blue' | 'green' | 'thumbnail';

export interface StacItem {
  id: string;
  collection: string;
  provider: string;
  datetime: string | null;
  cloudCover: number | null;
  bbox: [number, number, number, number] | null;
  platform: string | null;
  /** Canonical band key -> asset href (signed when Planetary Computer). */
  assets: Record<StacBandKey, string>;
  thumbnail: string | null;
}

export interface StacSearchOptions {
  collections: string[];
  bbox: [number, number, number, number];
  startDate?: string;
  endDate?: string;
  maxCloudCover?: number;
  limit?: number;
}

export interface StacSearchResult {
  provider: string;
  items: StacItem[];
  context: { matched?: number; returned: number; limit: number };
  nextToken: string | null;
}

function emptyStacItem(): Partial<StacItem> {
  return {
    assets: {} as Record<StacBandKey, string>,
  };
}

async function fetchWithTimeout(
  url: string,
  body: unknown,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

/** Search the STAC catalog via the HeraOculus backend proxy. */
export async function searchStacItems(
  options: StacSearchOptions
): Promise<StacSearchResult> {
  const {
    collections = DEFAULT_COLLECTIONS,
    bbox,
    startDate,
    endDate,
    maxCloudCover,
    limit = 10,
  } = options;

  const response = await fetchWithTimeout(STAC_SEARCH_URL, {
    collections,
    bbox,
    start_date: startDate ?? null,
    end_date: endDate ?? null,
    max_cloud_cover: maxCloudCover ?? null,
    limit,
  }, 25000);

  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(
      `STAC search returned HTTP ${response.status}: ${detail.slice(0, 300)}`
    );
  }

  const json = await response.json();

  const items: StacItem[] = (json.items || []).map((item: Record<string, unknown>) => ({
    id: (item.id as string) || '',
    collection: (item.collection as string) || '',
    provider: (item.provider as string) || '',
    datetime: (item.datetime as string) ?? null,
    cloudCover:
      typeof item.cloud_cover === 'number' ? item.cloud_cover : null,
    bbox: Array.isArray(item.bbox)
      ? ([item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]] as [
          number,
          number,
          number,
          number
        ])
      : null,
    platform: (item.platform as string) ?? null,
    assets: (item.assets as Record<StacBandKey, string>) || {},
    thumbnail: (item.thumbnail as string) ?? null,
  }));

  return {
    provider: (json.provider as string) || '',
    items,
    context: {
      matched: (json.context?.matched as number) ?? undefined,
      returned: typeof json.context?.returned === 'number'
        ? json.context.returned
        : items.length,
      limit: typeof json.context?.limit === 'number'
        ? json.context.limit
        : limit,
    },
    nextToken: (json.next_token as string) ?? null,
  };
}

const stacCapabilities: ConnectorCapabilities = {
  timeSeries: false,
  mapTiles: false,
  catalogSearch: true,
  pointSubset: false,
  regionSubset: true,
  coverage: globalCoverage(
    'Global — public STAC catalogs (Earth Search / Planetary Computer) for Sentinel-2 and Landsat imagery. Catalog/metadata search only; returns scene metadata and COG asset URLs, no time series.'
  ),
};

export const stacConnector: EarthDataConnector & {
  searchItems: (options: StacSearchOptions) => Promise<StacSearchResult>;
} = {
  id: 'stac-catalog',
  name: 'STAC Imagery Catalog',
  shortName: 'STAC',
  description:
    'SpatioTemporal Asset Catalog (STAC) search for satellite imagery scenes. Proxies Earth Search (Element 84, keyless) and, when configured, Microsoft Planetary Computer (SAS-signed assets). Returns scene metadata and Cloud-Optimized GeoTIFF asset URLs; catalog/metadata only, not time series.',
  baseUrl: STAC_API_BASE,
  capabilities: stacCapabilities,
  supportedVariables: [], // Catalog search; not tied to specific time-series variables
  provenanceType: 'satellite_observation',
  requiresAuth: false,
  requiresProxy: false,

  supportsVariable(_variableId: VariableId): boolean {
    // STAC is a catalog search; it does not produce time series for variables.
    return false;
  },

  async fetchTimeSeries(_options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    return {
      points: [],
      provenance: 'satellite_observation',
      sourceLabel: 'STAC Catalog',
      statusMessage:
        'STAC is a catalog search service. Use searchStacItems() to discover imagery scenes and their COG asset URLs.',
    };
  },

  searchItems: searchStacItems,
};
