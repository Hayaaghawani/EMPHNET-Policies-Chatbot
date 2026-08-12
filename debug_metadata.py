#!/usr/bin/env python3
"""Debug metadata types"""
import json

data = json.load(open('data/chunks_inspection.json', encoding='utf-8'))
c = data[0]
print("Chunk 0 metadata:")
for k, v in c['metadata'].items():
    print(f"  {k}: {type(v).__name__} = {repr(v)}")
