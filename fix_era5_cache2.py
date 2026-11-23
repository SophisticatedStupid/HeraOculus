#!/usr/bin/env python3
p = r'D:\Geo-Iris-main\Geo-Iris-main\backend\app\services\gridded_climate_service.py'
with open(p, encoding='utf-8') as f:
    lines = f.readlines()

# Fix cache-hit branch (lines 1650-1660 1-indexed = 0-indexed 1649-1659)
# Current:
# 1650:     cache_key = ...
# 1651:     cached = _cache_get(...)
# 1652:     if cached is not None:
# 1653:         arr = np.load(...)
# 1654:         if arr.ndim >= 2:
# 1655:             ...
# 1659:             arr = arr.reshape(n_lat, n_lon)
# 1660: 
# Need to add `return arr` after the if block, then fall through to Zarr read.

# Find the exact line indices
start_cache_hit = None
for i in range(1645, 1665):
    if 'cache_key = f"era5_land_monthly' in lines[i]:
        start_cache_hit = i
        break

print(f'cache_key at line {start_cache_hit+1}')

# Find the line after the reshape block (the blank line after arr.reshape)
end_cache_hit = None
for i in range(start_cache_hit, 1670):
    if lines[i].strip() == '' and i > start_cache_hit + 5:
        end_cache_hit = i
        break

print(f'End of cache hit block at line {end_cache_hit+1}')
print('Lines to fix:')
for i in range(start_cache_hit, end_cache_hit+2):
    print(f'{i+1}: {lines[i].rstrip()}')

# Replace the cache-hit block (start_cache_hit to end_cache_hit) with fixed version
new_cache_hit = [
    '    cache_key = f"era5_land_monthly_{year}_{month:02d}"\n',
    '    cached = _cache_get("era5-land", cache_key)\n',
    '    if cached is not None:\n',
    '        arr = np.load(io.BytesIO(cached), allow_pickle=False)\n',
    '        return arr\n',
    '\n',
]

lines[start_cache_hit:end_cache_hit+1] = new_cache_hit

# Fix cache-set (around line 1703-1706 1-indexed = 0-indexed 1702-1705)
# Find the pickle.dumps line
cache_set_idx = None
for i in range(1695, 1715):
    if 'pickle.dumps' in lines[i]:
        cache_set_idx = i
        break

print(f'\npickle.dumps at line {cache_set_idx+1}')

# Replace with np.save to BytesIO
lines[cache_set_idx] = '            buf = io.BytesIO()\n            np.save(buf, arr, allow_pickle=False)\n            _cache_set("era5-land", cache_key, buf.getvalue())\n'

print('Fixed cache-set')

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done')