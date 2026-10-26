"""Image normalisation shared by every dataset: 8-bit grayscale PNG, shortest side = `size` px,
aspect ratio preserved (each encoder's own processor crops/resizes from there)."""

import io

import numpy as np
from PIL import Image


def _resize_png(img: Image.Image, size: int) -> bytes:
    scale = size / min(img.size)
    out = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def to_gray_shortest_side(raw: bytes, size: int) -> tuple[bytes, dict]:
    """Any PIL-readable image (NIH PNGs, a few of which are RGBA) -> (png_bytes, original info)."""
    img = Image.open(io.BytesIO(raw))
    info = {"width": img.width, "height": img.height, "mode": img.mode}
    return _resize_png(img.convert("L"), size), info


def dicom_to_gray_shortest_side(raw: bytes, size: int) -> tuple[bytes, dict]:
    """DICOM bytes -> (png_bytes, header info). Applies the VOI LUT / window if present, inverts
    MONOCHROME1 (0 = white), then min-max scales to 8 bit."""
    import pydicom
    from pydicom.pixels import apply_voi_lut

    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = arr.max() - arr
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6) * 255
    info = {
        "width": int(ds.Columns),
        "height": int(ds.Rows),
        "mode": str(ds.PhotometricInterpretation),
        "view": str(ds.get("ViewPosition", "")),
        "age": str(ds.get("PatientAge", "")),
        "sex": str(ds.get("PatientSex", "")),
    }
    return _resize_png(Image.fromarray(arr.astype(np.uint8)), size), info
