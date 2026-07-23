"""
SimilarityStream 학습 스크립트

- Scene 이미지 + 고정 Target 이미지 → S_pred (similarity distribution map)
- Weighted MSE(sigmoid(S_pred), GT/255) 로 학습
  GT가 높을수록 (foreground일수록) 오차에 더 높은 weight 부여 → scale collapse 방지
- Target DINOv3 feature는 학습 시작 전 1회만 계산 후 캐싱
- DINOv3 backbone은 완전 frozen; scene_projs / target_projs / matching_blocks /
  feature_head / dist_head 만 학습
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from similarity_stream import SimilarityStream, load_image_tensor, make_fg_mask


# ── 경로 설정 ──────────────────────────────────────────────────────────────────

SCENE_DIR   = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/2D-PDM_data/scene/fruit_1/scene/rgb"
GT_DIR      = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/2D-PDM_data/similarity/distribution_map(fruit_1_beta0.0)/distribution_map"
TARGET_PATH = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/2D-PDM_data/target/fruit_1/target.png"
LOCAL_PTH   = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/model/dinov3_vit7b16_hf_converted.pth"
CKPT_DIR    = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/checkpoints"

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────

BATCH_SIZE  = 16       # GPU 메모리에 따라 조정
NUM_EPOCHS  = 30
LR          = 1e-4
VAL_RATIO   = 0.1     # 전체의 10%를 validation
GRAD_CLIP   = 1.0
SAVE_EVERY  = 5       # 몇 epoch마다 중간 checkpoint 저장
NUM_WORKERS = 4
SEED        = 42
FG_WEIGHT   = 10.0    # foreground(GT>0) 픽셀 오차에 곱하는 weight
                      # GT=0 → weight=1, GT=1.0 → weight=FG_WEIGHT (선형 보간)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PDMDataset(Dataset):
    """
    scene RGB + GT distribution map 쌍을 반환.
    파일명이 동일한 (scene_dir/*.png ↔ gt_dir/*.png) 이미지를 매칭.

    __getitem__ returns:
        scene_t: [3, H, W]  float [0,1]
        gt_t:    [1, H, W]  float [0,1]
    """

    def __init__(self, scene_dir: str, gt_dir: str, patch_size: int = 16):
        scene_dir = Path(scene_dir)
        gt_dir    = Path(gt_dir)

        pairs = []
        for sf in sorted(scene_dir.glob("*.png")):
            gf = gt_dir / sf.name
            if gf.exists():
                pairs.append((sf, gf))

        if not pairs:
            raise RuntimeError(
                f"Scene dir: {scene_dir}\nGT dir: {gt_dir}\n매칭된 파일이 없습니다."
            )

        self.pairs      = pairs
        self.patch_size = patch_size
        print(f"[Dataset] {len(pairs):,} 쌍 로드 완료")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        scene_path, gt_path = self.pairs[idx]
        ps = self.patch_size

        # ── Scene ──────────────────────────────────────────────────────────
        scene = Image.open(scene_path).convert("RGB")
        W, H  = scene.size
        new_W = (W // ps) * ps
        new_H = (H // ps) * ps
        if (W, H) != (new_W, new_H):
            scene = scene.resize((new_W, new_H), Image.BILINEAR)
        scene_t = torch.from_numpy(
            np.asarray(scene, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]

        # ── GT distribution map ────────────────────────────────────────────
        gt = Image.open(gt_path).convert("L")
        if (gt.width, gt.height) != (new_W, new_H):
            gt = gt.resize((new_W, new_H), Image.BILINEAR)
        gt_t = torch.from_numpy(
            np.asarray(gt, dtype=np.float32) / 255.0
        ).unsqueeze(0)  # [1, H, W]

        return scene_t, gt_t


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weighted_mse_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """
    GT 값에 비례한 weighted MSE.

    weight = 1 + (fg_weight - 1) * gt
      → GT=0.0 (배경)     : weight = 1
      → GT=0.2            : weight = 1 + 0.2*(fg_weight-1)
      → GT=1.0 (최고유사도): weight = fg_weight

    단순 이진 fg/bg 분리보다 부드럽고, 고유사도 픽셀을 더 강하게 학습.
    """
    weight = 1.0 + (fg_weight - 1.0) * gt          # [B, 1, H, W]
    return (weight * (pred - gt) ** 2).mean()


def save_checkpoint(path: Path, stream: SimilarityStream, optimizer, epoch: int, val_loss: float):
    """DINOv3 backbone 제외하고 저장 (용량 절약)."""
    state = {k: v for k, v in stream.state_dict().items()
             if not k.startswith("matching.extractor.backbone.")}
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     state,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
    }, path)


def load_checkpoint(path: Path, stream: SimilarityStream, optimizer=None):
    """저장된 checkpoint 로드 (backbone 제외 부분만)."""
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = stream.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {len(unexpected)}")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [INFO] checkpoint loaded: epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
    return ckpt["epoch"]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 모델 초기화 ──────────────────────────────────────────────────────────
    stream = SimilarityStream(
        dino_model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        dino_model_path=LOCAL_PTH,
        num_classes=1,
        freeze_dino=True,
    ).to(device)

    patch_size    = stream.matching.extractor.patch_size
    layer_indices = stream.matching.layer_indices
    print(f"[INFO] patch_size={patch_size}  layers={layer_indices}")
    print(f"[INFO] trainable params: {count_trainable(stream):,}")

    # ── Target feature 1회 사전 계산 및 캐싱 ─────────────────────────────────
    target_t, (H_t, W_t) = load_image_tensor(TARGET_PATH, patch_size, device)

    # gray 배경 자동 마스킹: FG 패치만 pooling에 사용
    tgt_mask = make_fg_mask(target_t, patch_size)       # [1, 1, H_p, W_p]
    fg_ratio = tgt_mask.mean().item()
    print(f"[INFO] target FG mask: {fg_ratio*100:.1f}% 패치가 foreground"
          f"  ({'정상' if fg_ratio > 0.05 else '주의: FG 패치가 너무 적음'})")

    with torch.no_grad():
        _tgt_feats = stream.matching.extractor(target_t)  # {b: [1, D, H_p, W_p]}
    # batch dim 제거; 학습 시 expand로 B에 맞춤
    cached_target = {b: f.squeeze(0) for b, f in _tgt_feats.items()}  # {b: [D, H_p, W_p]}
    cached_tgt_mask = tgt_mask.squeeze(0)               # [1, H_p, W_p]
    del _tgt_feats
    print(f"[INFO] target features cached: shape={tuple(next(iter(cached_target.values())).shape)}")

    def expand_target(B: int) -> dict:
        """캐싱된 target features를 batch size B로 expand."""
        return {b: f.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
                for b, f in cached_target.items()}

    def expand_tgt_mask(B: int) -> torch.Tensor:
        """캐싱된 target mask를 batch size B로 expand."""
        return cached_tgt_mask.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    full_dataset = PDMDataset(SCENE_DIR, GT_DIR, patch_size)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    idx = list(range(n_total))
    random.shuffle(idx)
    train_ds = Subset(full_dataset, idx[:n_train])
    val_ds   = Subset(full_dataset, idx[n_train:])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    print(f"[INFO] train={n_train:,}  val={n_val:,}  steps/epoch={len(train_loader):,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    trainable = [p for p in stream.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.05
    )
    print(f"[INFO] loss: weighted MSE  (fg_weight={FG_WEIGHT})")

    # ── Resume 지원 ────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val     = float("inf")
    best_ckpt    = Path(CKPT_DIR) / "best.pt"
    resume_ckpt  = Path(CKPT_DIR) / "last.pt"
    if resume_ckpt.exists():
        ans = input(f"이전 checkpoint 발견 ({resume_ckpt}). 이어서 학습하시겠습니까? [y/N] ").strip().lower()
        if ans == "y":
            start_epoch = load_checkpoint(resume_ckpt, stream, optimizer) + 1
            if best_ckpt.exists():
                best_val = torch.load(best_ckpt, map_location="cpu")["val_loss"]
            for _ in range(start_epoch - 1):
                scheduler.step()

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Start")
    print("=" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        stream.train()
        t0         = time.time()
        train_loss = 0.0

        for step, (scene_batch, gt_batch) in enumerate(train_loader, 1):
            scene_batch = scene_batch.to(device, non_blocking=True)  # [B, 3, H, W]
            gt_batch    = gt_batch.to(device,    non_blocking=True)  # [B, 1, H, W]
            B = scene_batch.shape[0]

            optimizer.zero_grad()

            S_pred, _ = stream(scene_batch, target_feats=expand_target(B),
                               target_mask=expand_tgt_mask(B))              # [B, 1, H, W]
            prob  = torch.sigmoid(S_pred)
            loss  = weighted_mse_loss(prob, gt_batch, FG_WEIGHT)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                avg = train_loss / step
                elapsed = time.time() - t0
                print(f"  [{epoch:3d}/{NUM_EPOCHS}] step {step:5d}/{len(train_loader)}"
                      f"  loss={avg:.6f}  ({elapsed:.0f}s)")

        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        stream.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scene_batch, gt_batch in val_loader:
                scene_batch = scene_batch.to(device, non_blocking=True)
                gt_batch    = gt_batch.to(device,    non_blocking=True)
                B = scene_batch.shape[0]
                S_pred, _ = stream(scene_batch, target_feats=expand_target(B),
                                   target_mask=expand_tgt_mask(B))
                prob = torch.sigmoid(S_pred)
                val_loss += weighted_mse_loss(prob, gt_batch, FG_WEIGHT).item()
        val_loss /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_wmse={train_loss:.6f}  val_wmse={val_loss:.6f}  "
              f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

        # ── Checkpoint 저장 ─────────────────────────────────────────────────
        save_checkpoint(resume_ckpt, stream, optimizer, epoch, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_ckpt, stream, optimizer, epoch, val_loss)
            print(f"  → best model updated  (val_wmse={val_loss:.6f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(
                Path(CKPT_DIR) / f"epoch_{epoch:03d}.pt",
                stream, optimizer, epoch, val_loss,
            )

    print("\n" + "=" * 60)
    print(f"  Training Complete  |  best val MSE: {best_val:.6f}")
    print("=" * 60)
