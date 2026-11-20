import { defineConfig } from 'vitest/config';

// Vitest configuration for Geo-Iris frontend validation.
// Runs pure logic tests in Node (no DOM required) so the suite is fast and
// reproducible on Windows and Linux without a browser.
export default defineConfig({
  test: {
    globals: false,
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    exclude: ['node_modules', 'dist', '.git'],
    // Stop on first failure for a clear validation sequence.
    stopOnFailure: false,
    // Detailed reporter output for CI/fresh-install verification.
    reporter: ['verbose'],
  },
  // Resolve TypeScript path aliases and ESM imports used by src.
  resolve: {
    conditions: ['development', 'module'],
  },
});
