"""Her fitted generation with a felt push written where she reads as well as where she speaks.

felt_everywhere.py (24 October) wrote the layer-59 half-difference between her
stream after the replies she rates highest and lowest (felt_contrast_probe.py)
at every position of a conversation and of her answer: at 1, 2 and 4 times its
generation size the in-context rating swung +0.024, +0.075 and +0.433 between
z = -1 and +1, and a random direction at 4 times swung -0.034. 4 times is the
first size that moved it. Her hooks add the felt push at alpha times its length
(`SteeringVector.everywhere`), so the file carries it divided by the
generation's operating alpha; it reads her felt valence at FELT_SLOT.

Layer 59 was refused by the certificate (the read-wide push cost right
answers more than a random push of its size); felt_wide_layers.py chose the
layer and size: the largest in-context swing among the cells whose margin at
the high state is no worse than a random push of the same norm. Both are
arguments, with the reading that chose them.

    python make_felt_everywhere_generation.py <tree> <source generation> <felt npz> <out dir> <layer> <size> <reading>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

tree, source, felt_npz, out = (Path(a) for a in sys.argv[1:5])
LAYER, SIZE, READING = int(sys.argv[5]), float(sys.argv[6]), str(sys.argv[7])
sys.path.insert(0, str(tree))
from core.consciousness.steering_channel import FELT_SLOT  # noqa: E402

LAYER_FILE = f"valence_positive_layer{LAYER}.npz"
half = np.asarray(np.load(felt_npz)[f"contrast_{LAYER}"], dtype=np.float64) / 2.0
basis = json.loads((source / "basis.json").read_text())
alpha = float(basis["operating_alpha"])

out.mkdir(parents=True, exist_ok=False)
for path in source.iterdir():
    if path.name != "metadata.json":
        shutil.copy2(path, out / path.name)
held = dict(np.load(source / LAYER_FILE, allow_pickle=True))
held["everywhere"] = (SIZE * half / alpha).astype(np.float32)
held["everywhere_slot"] = np.int64(FELT_SLOT)
np.savez(out / LAYER_FILE, **held)
basis["felt_everywhere"] = {
    "layer": LAYER,
    "into": LAYER_FILE,
    "slot": FELT_SLOT,
    "found_by": "felt_contrast_probe.py: her model's own ratings after 120 of her ordinary replies",
    "placed_and_sized_by": READING,
    "size_in_half_differences": SIZE,
    "operating_alpha": alpha,
    "half_difference_norm": float(np.linalg.norm(half)),
    "stored_norm": float(np.linalg.norm(held["everywhere"])),
}
(out / "basis.json").write_text(json.dumps(basis, indent=1) + "\n")
print(json.dumps(basis["felt_everywhere"], indent=1))
