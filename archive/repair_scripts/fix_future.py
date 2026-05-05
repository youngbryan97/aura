import os
import re

for root, _, files in os.walk('core'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r') as file:
                lines = file.readlines()
            
            future_idx = -1
            atomic_idx = -1
            
            for i, line in enumerate(lines):
                if 'from __future__ import annotations' in line:
                    future_idx = i
                if 'from core.runtime.atomic_writer import atomic_write_text' in line:
                    atomic_idx = i
                    
            if future_idx != -1 and atomic_idx != -1 and atomic_idx < future_idx:
                # swap them
                lines[atomic_idx], lines[future_idx] = lines[future_idx], lines[atomic_idx]
                with open(path, 'w') as file:
                    file.writelines(lines)
                print(f"Fixed {path}")
