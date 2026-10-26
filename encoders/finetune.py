"""Fully fine-tuned ResNet-50 baseline, classic recipe: ImageNet initialisation, every layer
trained with BCE over the six labels, AdamW, learning rate divided by 10 on a validation plateau,
early stopping on validation macro AUROC over the pathologies (validation BCE when AUROC is
undefined, i.e. a validation subset with no positives). Evaluation uses the same
Resize(256)+CenterCrop(224) transform as the frozen ResNet-50 encoder."""

import copy
import math
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50

from eval.metrics import auroc, macro

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
EVAL_TF = WEIGHTS.transforms()
TRAIN_TF = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(WEIGHTS.transforms().mean, WEIGHTS.transforms().std),
    ]
)


class _Images(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.transform = list(paths), transform
        self.labels = None if labels is None else torch.from_numpy(np.array(labels, np.float32))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return x if self.labels is None else (x, self.labels[i])


def make_model(n_labels):
    model = resnet50(weights=WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, n_labels)
    return model


@torch.no_grad()
def predict(model, paths, device, batch_size=128, workers=6):
    model.eval()
    loader = DataLoader(_Images(paths, None, EVAL_TF), batch_size, num_workers=workers)
    return np.concatenate([torch.sigmoid(model(x.to(device))).cpu().numpy() for x in loader])


def train(
    paths_tr,
    y_tr,
    paths_va,
    y_va,
    device,
    pathology_cols,
    lr=1e-4,
    batch_size=64,
    max_epochs=10,
    patience=3,
    workers=6,
    amp=False,
    log=print,
):
    """-> (best model on CPU, info). Selection score = val macro AUROC, or -val BCE when nan."""
    y_tr, y_va = np.asarray(y_tr, np.float32), np.asarray(y_va, np.float32)
    model = make_model(y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        _Images(paths_tr, y_tr, TRAIN_TF),
        batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=len(paths_tr) > batch_size,
        persistent_workers=workers > 0,
    )
    best, bad, history = (-math.inf, None, 0), 0, []
    for epoch in range(1, max_epochs + 1):
        t0, model_loss, n = time.time(), 0.0, 0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model_loss += loss.item() * len(x)
            n += len(x)
        p = predict(model, paths_va, device, workers=workers)
        val_auc = macro([auroc(y_va[:, j], p[:, j]) for j in pathology_cols])
        val_bce = float(
            nn.functional.binary_cross_entropy(torch.as_tensor(p), torch.as_tensor(y_va))
        )
        score, select = (val_auc, "auroc") if not np.isnan(val_auc) else (-val_bce, "bce")
        history.append(
            {
                "epoch": epoch,
                "train_bce": model_loss / max(n, 1),
                "val_auroc": val_auc,
                "val_bce": val_bce,
                "seconds": round(time.time() - t0),
            }
        )
        log(
            f"  epoch {epoch}: train_bce={history[-1]['train_bce']:.4f} val_auroc={val_auc:.4f} "
            f"val_bce={val_bce:.4f} lr={opt.param_groups[0]['lr']:.0e} {history[-1]['seconds']}s"
        )
        sched.step(score)
        if score > best[0]:
            best, bad = (score, copy.deepcopy(model.state_dict()), epoch), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1])
    info = {"best_epoch": best[2], "epochs": epoch, "select": select, "history": history}
    return model.cpu(), info
