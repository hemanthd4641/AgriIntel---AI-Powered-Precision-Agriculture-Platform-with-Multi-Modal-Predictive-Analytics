"""Standalone label loader that searches for common label files in models/plant_disease
and prints the labels found. Does not import Django or any project modules.
"""
import os
import json
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_labels_dir = os.path.join(repo_root, 'Agri', 'models', 'plant_disease')

candidates = [
    os.path.join(model_labels_dir, 'labels.txt'),
    os.path.join(model_labels_dir, 'classes.txt'),
    os.path.join(model_labels_dir, 'labels.json')
]

found = None
for p in candidates:
    if os.path.exists(p):
        found = p
        break

if not found:
    print('No label file found in', model_labels_dir)
    sys.exit(0)

print('Found label file:', found)
labels = None
try:
    if found.endswith('.json'):
        with open(found, 'r', encoding='utf-8') as fh:
            labels = json.load(fh)
    else:
        with open(found, 'r', encoding='utf-8') as fh:
            labels = [l.strip() for l in fh if l.strip()]
except Exception as e:
    print('Error reading label file:', e)
    sys.exit(1)

print('Loaded', len(labels), 'labels')
print('Sample labels:')
for i, lab in enumerate(labels[:20]):
    print(i, lab)

print('\nDone')
