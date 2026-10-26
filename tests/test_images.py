import io

from PIL import Image

from data.images import to_gray_shortest_side


def _png_bytes(mode, size):
    buf = io.BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (10, 20, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def test_rgba_input_becomes_8bit_gray_with_shortest_side_resized():
    out, info = to_gray_shortest_side(_png_bytes("RGBA", (1000, 1200)), 518)
    img = Image.open(io.BytesIO(out))
    assert img.mode == "L"
    assert img.size == (518, 622)  # shortest side 518, aspect ratio kept
    assert info == {"width": 1000, "height": 1200, "mode": "RGBA"}


def test_square_gray_input_is_square_output():
    out, _ = to_gray_shortest_side(_png_bytes("L", (1024, 1024)), 518)
    assert Image.open(io.BytesIO(out)).size == (518, 518)


def _dicom_bytes(photometric):
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 40, 60, 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = np.tile(np.linspace(0, 4095, 60, dtype=np.uint16), (40, 1)).tobytes()
    ds.ViewPosition, ds.PatientSex, ds.PatientAge = "PA", "F", "045Y"
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_dicom_monochrome1_is_inverted_and_windowed_to_8bit():
    from data.images import dicom_to_gray_shortest_side

    out, info = dicom_to_gray_shortest_side(_dicom_bytes("MONOCHROME1"), 20)
    img = Image.open(io.BytesIO(out))
    assert img.mode == "L" and img.size == (30, 20)
    assert img.getpixel((0, 10)) > 240 and img.getpixel((29, 10)) < 15  # MONOCHROME1: 0 is white
    assert info == {
        "width": 60,
        "height": 40,
        "mode": "MONOCHROME1",
        "view": "PA",
        "age": "045Y",
        "sex": "F",
    }


def test_dicom_monochrome2_keeps_polarity():
    from data.images import dicom_to_gray_shortest_side

    out, _ = dicom_to_gray_shortest_side(_dicom_bytes("MONOCHROME2"), 20)
    img = Image.open(io.BytesIO(out))
    assert img.getpixel((0, 10)) < 15 and img.getpixel((29, 10)) > 240
