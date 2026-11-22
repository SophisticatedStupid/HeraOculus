/**
 * Client-side secrets guard.
 *
 * Asserts that no source file reads a VITE_* environment variable whose name
 * contains "KEY" or "SECRET". Such a variable would be embedded in the public
 * JS bundle at build time, defeating the purpose of a server-side API key.
 * The API key must only ever be read server-side (Vercel proxy function and
 * FastAPI backend).
 */
import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';

const ROOT = process.cwd();
const SRC_DIR = join(ROOT, 'src');
const API_DIR = join(ROOT, 'api');

// Matches import.meta.env.VITE_<NAME> where <NAME> contains KEY or SECRET
// (case-insensitive), e.g. VITE_BACKEND_API_KEY or VITE_SECRET_TOKEN.
const SECRET_ENV_PATTERN =
  /import\.meta\.env\.VITE_[A-Za-z0-9_]*(?:KEY|SECRET)[A-Za-z0-9_]*/gi;

function walk(dir: string): string[] {
  let results: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      results = results.concat(walk(full));
    } else if (entry.isFile() && /\.(ts|tsx|js|jsx)$/.test(entry.name)) {
      results.push(full);
    }
  }
  return results;
}

describe('client secrets guard', () => {
  it('no source file reads a VITE_* variable containing KEY or SECRET', () => {
    const files = [...walk(SRC_DIR), ...walk(API_DIR)];
    const violations: string[] = [];
    for (const file of files) {
      const content = readFileSync(file, 'utf8');
      for (const match of content.matchAll(SECRET_ENV_PATTERN)) {
        violations.push(`${file}:${match.index}`);
      }
    }
    expect(violations).toEqual([]);
  });
});
