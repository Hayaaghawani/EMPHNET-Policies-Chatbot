#!/usr/bin/env python3
"""Verify chunk quality from chunks_inspection.json"""
import json

data = json.load(open('data/chunks_inspection.json', encoding='utf-8'))
print(f'✓ {len(data)} chunks created\n')

# Sample chunks
for i in [0, 10, 20, 29]:
    if i < len(data):
        c = data[i]
        print(f'Chunk {i}:')
        print(f'  Text: {c["text"][:100]}...')
        print(f'  Metadata: {c["metadata"]}')
        print()
