/**
 * Earth Data Connectors — Unified Registry & Connectors Export
 */
import { connectorRegistry } from './registry';
import { powerConnector } from './powerConnector';
import { modisConnector } from './modisConnector';
import { cmrConnector } from './cmrConnector';
import { gibsConnector } from './gibsConnector';
import { noaaConnector } from './noaaConnector';
import { usgsConnector } from './usgsConnector';
import { stacConnector } from './stacConnector';
import { chirpsConnector } from './chirpsConnector';
import { era5LandConnector } from './era5LandConnector';
import { imergConnector } from './imergConnector';
import { smapConnector } from './smapConnector';

// Register all available client connectors
connectorRegistry.register(powerConnector);
connectorRegistry.register(modisConnector);
connectorRegistry.register(cmrConnector);
connectorRegistry.register(gibsConnector);
connectorRegistry.register(noaaConnector);
connectorRegistry.register(usgsConnector);
connectorRegistry.register(stacConnector);
connectorRegistry.register(chirpsConnector);
connectorRegistry.register(imergConnector);
connectorRegistry.register(era5LandConnector);
connectorRegistry.register(smapConnector);

export { connectorRegistry };
export { powerConnector } from './powerConnector';
export { modisConnector } from './modisConnector';
export { cmrConnector } from './cmrConnector';
export { gibsConnector } from './gibsConnector';
export { noaaConnector } from './noaaConnector';
export { usgsConnector } from './usgsConnector';
export { stacConnector } from './stacConnector';
export { chirpsConnector } from './chirpsConnector';
export { imergConnector } from './imergConnector';
export { era5LandConnector } from './era5LandConnector';
export { smapConnector } from './smapConnector';
export type {
  StacItem,
  StacSearchOptions,
  StacSearchResult,
  StacBandKey,
} from './stacConnector';
export type { GriddedSeriesResponse } from './chirpsConnector';
export type { GriddedSeriesResponse as ChirpsGriddedSeriesResponse } from './chirpsConnector';
export type { GriddedSeriesResponse as ImergGriddedSeriesResponse } from './imergConnector';
export type { GriddedSeriesResponse as Era5LandGriddedSeriesResponse } from './era5LandConnector';
export type { GriddedSeriesResponse as SmapGriddedSeriesResponse } from './smapConnector';
export * from './types';
export * from './registry';

