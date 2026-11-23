#!/usr/bin/env python3
p = r'D:\Geo-Iris-main\Geo-Iris-main\backend\app\services\gridded_climate_service.py'
with open(p, encoding='utf-8') as f:
    lines = f.readlines()

# Replace lines 1652-1653 (1-indexed) -> 0-indexed 1651 and 1652
# Line 1652: "    cached = _cache_get("era5-land", cache_key)"
# Line 1653: "        return pickle.loads(cached)"

old_lines = lines[1650:1653]  # 0-indexed 1650,1651,1652
print(f'OLD: {old_lines}')

new_lines = [
    '    cached = _cache_get("era5-land", cache_key)\n',
    '    if cached is not None:\n',
    '        arr = np.load(io.BytesIO(cached), allow_pickle=False)\n',
    '        if arr.ndim >= 2:\n',
    '            # arr may have been stored flattened from a global grid; reshape\n',
    '            # back to (n_lat, n_lon) using known ERA5-Land grid shape.\n',
    '            n_lat = int(round(180.0 / ERA5_LAND_RES_DEG))\n',
    '            n_lon = int(round(360.0 / ERA5_LAND_RES_DEG))\n',
    '            arr = arr.reshape(n_lat, n_lon)\n',
]

lines[1650:1653] = new_lines
print('Replaced')
print(f'NEW: {lines[1648:1660]}')

with open(p, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done')
