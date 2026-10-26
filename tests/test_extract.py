import numpy as np
import torch
from PIL import Image

from encoders.extract import extract


def _fake_images(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("L", (8, 8), color=10 * i).save(p)
        paths.append(p)
    return paths


def _fake_encoder():
    transform = lambda img: torch.tensor([float(img.getpixel((0, 0)))])  # noqa: E731
    forward = lambda x: torch.cat([x, x * 2], dim=1)  # noqa: E731  (B,1) -> (B,2)
    return transform, forward, 2


def test_extract_writes_npy_in_metadata_order_and_id_parquet(tmp_path):
    paths = _fake_images(tmp_path, 5)
    transform, forward, dim = _fake_encoder()
    out = tmp_path / "emb"

    extract(
        paths, [p.stem for p in paths], transform, forward, dim, out, "fake", "cpu", batch_size=2
    )

    arr = np.load(out / "fake.npy")
    assert arr.shape == (5, 2) and arr.dtype == np.float32
    assert arr[:, 0].tolist() == [0, 10, 20, 30, 40] and arr[3, 1] == 60
    import pandas as pd

    ids = pd.read_parquet(out / "fake.parquet")
    assert ids["image_id"].tolist() == [p.stem for p in paths] and ids["row"].tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_extract_resumes_from_progress_file(tmp_path):
    paths = _fake_images(tmp_path, 5)
    transform, forward, dim = _fake_encoder()
    out = tmp_path / "emb"
    extract(
        paths, [p.stem for p in paths], transform, forward, dim, out, "fake", "cpu", batch_size=2
    )
    arr = np.lib.format.open_memmap(out / "fake.npy", mode="r+")
    arr[2:] = -1  # simulate a crash after 2 rows
    arr.flush()
    (out / "fake.progress").write_text("2")
    calls = []

    def counting_forward(x):
        calls.append(len(x))
        return forward(x)

    extract(
        paths,
        [p.stem for p in paths],
        transform,
        counting_forward,
        dim,
        out,
        "fake",
        "cpu",
        batch_size=2,
    )

    assert sum(calls) == 3  # only the missing rows were recomputed
    assert np.load(out / "fake.npy")[:, 0].tolist() == [0, 10, 20, 30, 40]
