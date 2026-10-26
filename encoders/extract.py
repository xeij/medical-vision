"""Resumable embedding extraction: one float32 row per image, written to a .npy memmap in metadata
order, plus a <name>.parquet mapping image_id -> row. A <name>.progress file records how many rows
are complete so an interrupted multi-hour run restarts where it stopped."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class _Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]))


def extract(
    paths, image_ids, transform, forward, dim, out_dir, name, device, batch_size=64, workers=0
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, progress = out_dir / f"{name}.npy", out_dir / f"{name}.progress"
    if npy.exists() and progress.exists():
        arr, start = open_memmap(npy, mode="r+"), int(progress.read_text())
    else:
        arr, start = open_memmap(npy, mode="w+", dtype=np.float32, shape=(len(paths), dim)), 0
    loader = DataLoader(
        _Images(paths[start:], transform), batch_size=batch_size, num_workers=workers
    )
    i = start
    for step, x in enumerate(tqdm(loader, initial=start // batch_size, desc=name)):
        with torch.inference_mode():
            z = forward(x.to(device)).float().cpu().numpy()
        arr[i : i + len(z)] = z
        i += len(z)
        if step % 50 == 0:
            arr.flush()
            progress.write_text(str(i))
    arr.flush()
    progress.write_text(str(i))
    pd.DataFrame({"image_id": image_ids, "row": range(len(image_ids))}).to_parquet(
        out_dir / f"{name}.parquet"
    )
