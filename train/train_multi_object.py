"""
Multi-Object Episodic 학습 스크립트

변경점 (단일 object 버전 대비):
  - 4개 object 동시 학습 (fruit_1/2, packaged_food_1/2)
  - 배치 내 샘플마다 다른 target → head가 "비교하는 방법" 자체를 학습
  - target features(patch + CLS) 사전 캐싱 → 학습 중 DINOv3 재연산 없음
  - CLS token 가산 융합으로 zero-shot 의미 일반화 강화
"""

import sys
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from similarity_stream import SimilarityStream, load_image_tensor, make_fg_mask


# ── 경로 설정 ──────────────────────────────────────────────────────────────────

BASE      = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/data"
LOCAL_PTH = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/model/dinov3_vit7b16_hf_converted.pth"
CKPT_DIR  = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/checkpoints_multi"

OBJECT_CONFIGS = [
    {
        "id":          "fruit_1",
        "scene_dir":   f"{BASE}/scene/fruit_1/scene/rgb",
        "gt_dir":      f"{BASE}/similarity/distribution_map(fruit_1_beta0.0)/distribution_map",
        "target_path": f"{BASE}/target/fruit_1/target.png",
    },
    {
        "id":          "fruit_2",
        "scene_dir":   f"{BASE}/scene/fruit_2/scene/rgb",
        "gt_dir":      f"{BASE}/similarity/distribution_map(fruit_2_beta0.0)/distribution_map",
        "target_path": f"{BASE}/target/fruit_2/target.png",
    },
    {
        "id":          "packaged_food_1",
        "scene_dir":   f"{BASE}/scene/packaged_food_1/scene/rgb",
        "gt_dir":      f"{BASE}/similarity/distribution_map(packaged_food_1_beta0.0)/distribution_map",
        "target_path": f"{BASE}/target/packaged_food_1/target.png",
    },
    {
        "id":          "packaged_food_2",
        "scene_dir":   f"{BASE}/scene/packaged_food_2/scene/rgb",
        "gt_dir":      f"{BASE}/similarity/distribution_map(packaged_food_2_beta0.0)/distribution_map",
        "target_path": f"{BASE}/target/packaged_food_2/target.png",
    },
]

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────

BATCH_SIZE  = 4       # object 수(4)의 배수가 자연스러움
NUM_EPOCHS  = 30
LR          = 1e-4
VAL_RATIO   = 0.1
GRAD_CLIP   = 1.0
SAVE_EVERY  = 5
NUM_WORKERS = 4
SEED        = 42
FG_WEIGHT   = 10.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiObjectDataset(Dataset):
    """
    여러 object의 (scene, GT, obj_idx) 쌍을 하나의 dataset으로 통합.

    __getitem__ returns:
        scene_t:  [3, H, W]  float [0,1]
        gt_t:     [1, H, W]  float [0,1]
        obj_idx:  int        OBJECT_CONFIGS 인덱스
    """

    def __init__(self, configs: list, patch_size: int = 16):
        self.samples:    list = []   # (scene_path, gt_path, obj_idx)
        self.patch_size: int  = patch_size

        for obj_idx, cfg in enumerate(configs):
            scene_dir = Path(cfg["scene_dir"])
            gt_dir    = Path(cfg["gt_dir"])
            matched   = 0
            for sf in sorted(scene_dir.glob("*.png")):
                gf = gt_dir / sf.name
                if gf.exists():
                    self.samples.append((sf, gf, obj_idx))
                    matched += 1
            print(f"  [{cfg['id']}] {matched:,} 쌍")

        print(f"  전체: {len(self.samples):,} 쌍  ({len(configs)} objects)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        scene_path, gt_path, obj_idx = self.samples[idx]
        ps = self.patch_size

        scene = Image.open(scene_path).convert("RGB")
        W, H  = scene.size
        nW = (W // ps) * ps
        nH = (H // ps) * ps
        if (W, H) != (nW, nH):
            scene = scene.resize((nW, nH), Image.BILINEAR)
        scene_t = torch.from_numpy(
            np.asarray(scene, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        gt = Image.open(gt_path).convert("L")
        if (gt.width, gt.height) != (nW, nH):
            gt = gt.resize((nW, nH), Image.BILINEAR)
        gt_t = torch.from_numpy(
            np.asarray(gt, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        return scene_t, gt_t, obj_idx


# ── Target Cache 빌더 ─────────────────────────────────────────────────────────

def build_target_cache(stream, configs, patch_size, device):
    """
    모든 object의 target features를 사전 계산.

    returns:
        patch_cache: {obj_idx: {layer: Tensor[D, H_p, W_p]}}
        cls_cache:   {obj_idx: {layer: Tensor[D]}}
        mask_cache:  {obj_idx: Tensor[1, H_p, W_p]}
    """
    patch_cache = {}
    cls_cache   = {}
    mask_cache  = {}

    for obj_idx, cfg in enumerate(configs):
        tgt_t, _ = load_image_tensor(cfg["target_path"], patch_size, device)
        tgt_mask  = make_fg_mask(tgt_t, patch_size)             # [1, 1, H_p, W_p]

        with torch.no_grad():
            patch_feats, cls_feats = stream.matching.extractor(tgt_t)

        patch_cache[obj_idx] = {b: f.squeeze(0) for b, f in patch_feats.items()}
        cls_cache[obj_idx]   = {b: c.squeeze(0) for b, c in cls_feats.items()}
        mask_cache[obj_idx]  = tgt_mask.squeeze(0)              # [1, H_p, W_p]

        fg_pct = tgt_mask.mean().item() * 100
        print(f"  [{cfg['id']}] FG={fg_pct:.1f}%  patch shape={tuple(next(iter(patch_cache[obj_idx].values())).shape)}")

    return patch_cache, cls_cache, mask_cache


def collate_targets(obj_ids, patch_cache, cls_cache, mask_cache, layer_indices, device):
    """
    배치 내 각 샘플의 obj_idx를 보고 사전 캐싱된 target tensors를 batch 형태로 조립.

    returns:
        tgt_feats: {layer: [B, D, H_p, W_p]}
        tgt_cls:   {layer: [B, D]}
        tgt_mask:  [B, 1, H_p, W_p]
    """
    tgt_feats = {
        b: torch.stack([patch_cache[oi][b] for oi in obj_ids]).to(device)
        for b in layer_indices
    }
    tgt_cls = {
        b: torch.stack([cls_cache[oi][b] for oi in obj_ids]).to(device)
        for b in layer_indices
    }
    tgt_mask = torch.stack([mask_cache[oi] for oi in obj_ids]).to(device)
    return tgt_feats, tgt_cls, tgt_mask


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mse_loss(pred, gt, fg_weight=10.0):
    weight = 1.0 + (fg_weight - 1.0) * gt
    return (weight * (pred - gt) ** 2).mean()


def save_checkpoint(path, stream, optimizer, epoch, val_loss):
    state = {k: v for k, v in stream.state_dict().items()
             if not k.startswith("matching.extractor.backbone.")}
    torch.save({
        "epoch": epoch,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, stream, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = stream.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [WARN] missing  : {len(missing)} keys")
    if unexpected:
        print(f"  [WARN] unexpected: {len(unexpected)} keys")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [INFO] checkpoint loaded  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
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
    print(f"[INFO] trainable params: {sum(p.numel() for p in stream.parameters() if p.requires_grad):,}")

    # ── 모든 Object target features 사전 계산 ────────────────────────────────
    print("\n[INFO] target features 사전 계산 ...")
    patch_cache, cls_cache, mask_cache = build_target_cache(
        stream, OBJECT_CONFIGS, patch_size, device
    )

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    print("\n[INFO] dataset 로드 ...")
    full_ds = MultiObjectDataset(OBJECT_CONFIGS, patch_size)

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    idx = list(range(n_total))
    random.shuffle(idx)
    train_ds = Subset(full_ds, idx[:n_train])
    val_ds   = Subset(full_ds, idx[n_train:])

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

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    best_ckpt   = Path(CKPT_DIR) / "best.pt"
    last_ckpt   = Path(CKPT_DIR) / "last.pt"
    if last_ckpt.exists():
        ans = input(f"이전 checkpoint 발견. 이어서 학습하시겠습니까? [y/N] ").strip().lower()
        if ans == "y":
            start_epoch = load_checkpoint(last_ckpt, stream, optimizer) + 1
            if best_ckpt.exists():
                best_val = torch.load(best_ckpt, map_location="cpu")["val_loss"]
            for _ in range(start_epoch - 1):
                scheduler.step()

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Multi-Object Episodic Training Start")
    print(f"  Objects: {[c['id'] for c in OBJECT_CONFIGS]}")
    print("=" * 60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        stream.train()
        t0         = time.time()
        train_loss = 0.0

        for step, (scene_batch, gt_batch, obj_ids) in enumerate(train_loader, 1):
            scene_batch = scene_batch.to(device, non_blocking=True)
            gt_batch    = gt_batch.to(device,    non_blocking=True)
            obj_ids     = obj_ids.tolist()

            # 배치 내 각 샘플의 target features/CLS/mask를 캐시에서 조립
            tgt_feats, tgt_cls, tgt_mask = collate_targets(
                obj_ids, patch_cache, cls_cache, mask_cache, layer_indices, device
            )

            optimizer.zero_grad()
            S_pred, _ = stream(
                scene_batch,
                target_feats=tgt_feats,
                target_cls=tgt_cls,
                target_mask=tgt_mask,
            )
            prob = torch.sigmoid(S_pred)
            loss = weighted_mse_loss(prob, gt_batch, FG_WEIGHT)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

            if step % 500 == 0 or step == len(train_loader):
                avg     = train_loss / step
                elapsed = time.time() - t0
                print(f"  [{epoch:3d}/{NUM_EPOCHS}] step {step:6d}/{len(train_loader)}"
                      f"  loss={avg:.6f}  ({elapsed:.0f}s)")

        train_loss /= len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        stream.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scene_batch, gt_batch, obj_ids in val_loader:
                scene_batch = scene_batch.to(device, non_blocking=True)
                gt_batch    = gt_batch.to(device,    non_blocking=True)
                obj_ids     = obj_ids.tolist()

                tgt_feats, tgt_cls, tgt_mask = collate_targets(
                    obj_ids, patch_cache, cls_cache, mask_cache, layer_indices, device
                )
                S_pred, _ = stream(
                    scene_batch,
                    target_feats=tgt_feats,
                    target_cls=tgt_cls,
                    target_mask=tgt_mask,
                )
                prob      = torch.sigmoid(S_pred)
                val_loss += weighted_mse_loss(prob, gt_batch, FG_WEIGHT).item()

        val_loss /= len(val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_wmse={train_loss:.6f}  val_wmse={val_loss:.6f}  "
              f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

        save_checkpoint(last_ckpt, stream, optimizer, epoch, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_ckpt, stream, optimizer, epoch, val_loss)
            print(f"  → best updated  (val_wmse={val_loss:.6f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(
                Path(CKPT_DIR) / f"epoch_{epoch:03d}.pt",
                stream, optimizer, epoch, val_loss,
            )

    print("\n" + "=" * 60)
    print(f"  Training Complete  |  best val wmse: {best_val:.6f}")
    print("=" * 60)
