import sys
import os
try:
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
    print("mlx_vlm successfully imported.")
    print(dir(load_image))
except Exception as e:
    print(f"Error: {e}")
