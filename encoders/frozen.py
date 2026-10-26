"""Frozen image encoders. load(name, device) -> (transform, forward, dim) where transform maps a
PIL image to a tensor and forward maps a batch to (B, dim) embeddings. One vector per image:
CLS token for the ViTs (RAD-DINO, DINOv2), projected image embedding for BiomedCLIP, global
average pool for ResNet-50. Each encoder runs at its native resolution and normalisation."""

import torch

from encoders import ENCODERS


class _RGB:
    """Picklable (DataLoader workers use spawn on macOS): PIL -> RGB -> underlying transform."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, img):
        return self.fn(img.convert("RGB"))


class _HF:
    def __init__(self, proc):
        self.proc = proc

    def __call__(self, img):
        return self.proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]


HF_IDS = {"rad_dino": "microsoft/rad-dino", "dinov2": "facebook/dinov2-base"}
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load(name: str, device: str):
    """ViTs run in fp16 on GPU (MPS/CUDA): 2-4x faster, embeddings identical to 5 decimals."""
    dtype = torch.float32 if device == "cpu" else torch.float16
    if name in HF_IDS:
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(HF_IDS[name])
        model = AutoModel.from_pretrained(HF_IDS[name], dtype=dtype).eval().to(device)

        def forward(x):
            return model(pixel_values=x.to(dtype)).pooler_output  # CLS token after final layernorm

        return _HF(proc), forward, model.config.hidden_size

    if name == "biomedclip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP)
        model = model.eval().to(device=device, dtype=dtype)
        return _RGB(preprocess), (lambda x: model.encode_image(x.to(dtype))), 512

    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model = model.eval().to(device)
        tf = weights.transforms()
        return _RGB(tf), model, 2048

    raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
