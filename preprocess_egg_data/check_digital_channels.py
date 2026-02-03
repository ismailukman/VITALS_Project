#!/usr/bin/env python3
"""Check the content of Digital channels to understand which has the MRI trigger."""
import bioread
import numpy as np

egg_data_path = '../egg_data'

for subject in ['VITD0107', 'VITD0108']:
    path = f'{egg_data_path}/{subject}_Acq/{subject}_EGG.acq'
    data = bioread.read_file(path)
    
    print(f'\n{"="*60}')
    print(f'{subject}')
    print(f'{"="*60}')
    
    for i, ch in enumerate(data.channels):
        if 'Digital' in ch.name or 'STP' in ch.name:
            signal = ch.data
            unique_vals = np.unique(signal)
            num_transitions = np.sum(np.abs(np.diff(signal)) > 0)
            
            print(f'\nChannel {i}: {ch.name}')
            print(f'  Unique values: {unique_vals[:10]}{"..." if len(unique_vals) > 10 else ""}')
            print(f'  Num unique values: {len(unique_vals)}')
            print(f'  Min: {signal.min()}, Max: {signal.max()}')
            print(f'  Number of transitions (edges): {num_transitions}')
            print(f'  First 20 samples: {signal[:20]}')
            
            # Look for trigger-like behavior (regular pulses)
            if len(unique_vals) == 2:
                # Binary signal - likely a trigger
                high_val = unique_vals[1]
                pulse_starts = np.where(np.diff(signal) > 0)[0]
                if len(pulse_starts) > 1:
                    intervals = np.diff(pulse_starts) / ch.samples_per_second
                    print(f'  Pulse intervals (first 5): {intervals[:5]} seconds')
                    print(f'  Mean interval: {np.mean(intervals):.3f}s, Std: {np.std(intervals):.3f}s')
