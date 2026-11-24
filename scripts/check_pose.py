"""Summarise a DWPose raw.npz: frames, keypoints, lost frames, median body-keypoint score."""
import sys
import numpy as np
for path in sys.argv[1:]:
    d = np.load(path)
    print(f"{path.split('/')[-2]:7s} {d['kpts'].shape[0]:4d} frames  {d['kpts'].shape[1]} kpts  "
          f"lost {int(d['lost'].sum())}  median score {np.nanmedian(d['scores'][:, :17]):.2f}")
