# features/dino_embedder.py
"""
DINOv2 embedding extractor.

Loads the fine-tuned StoneEmbedder once, exposes:
  - embed_image(path)  → 256-dim L2-normalised numpy vector
  - embed_batch(paths) → (N, 256) array
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from config.settings import CKPT_DIR, EMBED_DIM, PROJ_DIM, IMG_SIZE

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class StoneEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE,
        )
        bdim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(bdim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(PROJ_DIM, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        feat = self.backbone(x)
        emb  = F.normalize(self.projector(feat), dim=-1)
        if return_embedding:
            return emb
        return emb, self.classifier(emb)


# ── Singleton loader ──────────────────────────────────────────────────────────
_model       = None
_num_classes = None


def _load_model(num_classes: int):
    global _model, _num_classes
    if _model is not None and _num_classes == num_classes:
        return _model

    ckpt_dir  = Path(CKPT_DIR)
    ckpt_file = ckpt_dir / "best_stone_model_stage2.pt"
    if not ckpt_file.exists():
        ckpt_file = ckpt_dir / "best_stone_model.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"[DINOEmbedder] No model weights in '{ckpt_dir}'. "
            "Expected best_stone_model_stage2.pt or best_stone_model.pt"
        )

    ck    = torch.load(ckpt_file, map_location=_DEVICE)
    model = StoneEmbedder(num_classes).to(_DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    _model       = model
    _num_classes = num_classes
    print(f"[DINOEmbedder] ✅ Loaded {ckpt_file.name} | device={_DEVICE}")
    return model


def _get_num_classes() -> int:
    """Read num_classes from stone_index_meta.pkl."""
    import pickle
    meta_path = Path(CKPT_DIR) / "stone_index_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return len(meta["family_names"])


def get_model() -> StoneEmbedder:
    return _load_model(_get_num_classes())


# ── Public API ────────────────────────────────────────────────────────────────

def embed_image(image_path: str) -> np.ndarray | None:
    """Return 256-dim L2-normalised embedding, or None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = VAL_TF(image=img)["image"].unsqueeze(0).to(_DEVICE)
    model  = get_model()
    with torch.no_grad():
        vec = model(tensor, return_embedding=True).cpu().numpy().astype(np.float32)
    return vec[0]   # (256,)


def embed_batch(image_paths: list[str], batch_size: int = 32) -> tuple[list[str], np.ndarray]:
    """
    Embed a list of image paths in batches.
    Returns (valid_paths, embeddings_array).
    """
    import torch
    model  = get_model()
    valid_paths = []
    all_embs    = []

    imgs_buf = []
    path_buf = []

    def _flush():
        if not imgs_buf:
            return
        batch  = torch.stack(imgs_buf).to(_DEVICE)
        with torch.no_grad():
            embs = model(batch, return_embedding=True).cpu().numpy().astype(np.float32)
        all_embs.append(embs)
        valid_paths.extend(path_buf)
        imgs_buf.clear()
        path_buf.clear()

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = VAL_TF(image=img)["image"]
        imgs_buf.append(t)
        path_buf.append(str(p))
        if len(imgs_buf) >= batch_size:
            _flush()

    _flush()

    if not all_embs:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)

    return valid_paths, np.concatenate(all_embs, axis=0)