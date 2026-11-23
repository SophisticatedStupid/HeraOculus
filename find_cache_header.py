#!/usr/bin/env python3
p = r'D:\Geo-Iris-main\Geo-Iris-main\backend\app\services\gridded_climate_service.py'
with open(p, encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if '#' in line and 'Cache' in line:
        print(f'{i+1}: {line[:30]!r}')
