#!/usr/bin/env python3
import bioread
import os

egg_data_path = '../egg_data'

for subject in ['VITD0107', 'VITD0108']:
    # Try different path patterns
    paths = [
        f'{egg_data_path}/{subject}_Acq/{subject}_EGG.acq',
        f'{egg_data_path}/{subject}/{subject}_Acq/{subject}_EGG.acq',
    ]
    
    for path in paths:
        if os.path.exists(path):
            data = bioread.read_file(path)
            print(f'{subject} ({path}):')
            for i, ch in enumerate(data.channels):
                print(f'  Channel {i}: {ch.name}')
            print(f'  Total: {len(data.channels)} channels')
            print()
            break
    else:
        print(f'{subject}: File not found')
        print()
