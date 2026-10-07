import glob
import json
import random
from pathlib import Path

base = "/scratch/project_465002884/kidney/"
cases = sorted(glob.glob(f"{base}tuh_train/cases/images/train/*.nii.gz"))
controls = sorted(glob.glob(f"{base}tuh_train/controls/images/train/*.nii.gz"))

cases += sorted(glob.glob(f"{base}tuh_extra/cases/images/train/*.nii.gz"))
controls += sorted(glob.glob(f"{base}tuh_extra/controls/images/train/*.nii.gz"))

rng = random.Random(1234)
val = []
for group in (cases, controls):
    g = list(group)  # already sorted -> deterministic
    rng.shuffle(g)
    val += g[: round(0.2 * len(g))]

key = lambda p: str(Path(p))
json.dump(sorted(key(p) for p in val), open("val_split.json", "w"), indent=1)

print(f"val: {len(val)} ({sum(p in set(cases) for p in val)} cases, "
      f"{sum(p in set(controls) for p in val)} controls)")
print(f"train remainder: {len(cases) + len(controls) - len(val)}")
