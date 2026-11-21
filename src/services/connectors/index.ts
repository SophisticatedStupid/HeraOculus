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

// Register all available client connectors
connectorRegistry.register(powerConnector);
connectorRegistry.register(modisConnector);
connectorRegistry.register(cmrConnector);
connectorRegistry.register(gibsConnector);
connectorRegistry.register(noaaConnector);
connectorRegistry.register(usgsConnector);
connectorRegistry.register(stacConnector);

export { connectorRegistry };
export { powerConnector } from './powerConnector';
export { modisConnector } from './modisConnector';
export { cmrConnector } from './cmrConnector';
export { gibsConnector } from './gibsConnector';
export { noaaConnector } from './noaaConnector';
export { usgsConnector } from './usgsConnector';
export { stacConnector } from './stacConnector';
export type { StacItem, StacSearchOptions, StacSearchResult, StacBandKey } from './stacConnector';

export * from './types';
export * from './registry';
