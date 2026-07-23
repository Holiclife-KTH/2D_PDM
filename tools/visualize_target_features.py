"""
Target 이미지의 DINOv3 feature 시각화.

출력 패널 (왼쪽 → 오른쪽):
  1. 원본 target 이미지
  2. FG mask  (gray 배경 제외한 foreground 패치)
  3~5. Layer 33/36/39 feature PCA (상위 3 PC → RGB)
  6. FG-pooled feature와 각 패치의 cosine similarity 히트맵
     (학습 시 scene과 비교하는 실제 쿼리 벡터 기준)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from similarity_stream import (
    DINOFeatureExtractor,
    pool_target_feature,
    make_fg_mask,
    load_image_tensor,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────

_ROOT       = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws"
TARGET_PATH = f"{_ROOT}/data/target/fruit_1/target.png"
LOCAL_PTH   = f"{_ROOT}/model/dinov3_vit7b16_hf_converted.pth"
MODEL_NAME  = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
SAVE_PATH   = "target_feature_vis.png"   # None이면 화면 표시만

# ── Feature 추출 ───────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] device: {device}")

extractor = DINOFeatureExtractor(
    model_name=MODEL_NAME,
    model_path=LOCAL_PTH,
    freeze=True,
).to(device)

patch_size    = extractor.patch_size
layer_indices = extractor.layer_indices
print(f"[INFO] patch_size={patch_size}  layers={layer_indices}")

target_t, (H, W) = load_image_tensor(TARGET_PATH, patch_size, device)
H_p, W_p = H // patch_size, W // patch_size
print(f"[INFO] image size: {H}×{W}  →  patch grid: {H_p}×{W_p}")

# FG mask
tgt_mask = make_fg_mask(target_t, patch_size)          # [1, 1, H_p, W_p]
fg_ratio = tgt_mask.mean().item()
print(f"[INFO] FG 패치: {int(tgt_mask.sum())}/{H_p*W_p}  ({fg_ratio*100:.1f}%)")

# DINOv3 feature 추출
with torch.no_grad():
    feats = extractor(target_t)                        # {b: [1, D, H_p, W_p]}

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().cpu().float().numpy()


def pca_rgb(feat_map: torch.Tensor) -> np.ndarray:
    """
    feat_map: [1, D, H_p, W_p]
    → [H_p, W_p, 3]  (PCA 상위 3 PC → RGB, [0,1] normalize)
    """
    hw_d = feat_map.squeeze(0).permute(1, 2, 0).reshape(-1, feat_map.shape[1])
    hw_d = hw_d.cpu().float().numpy()                  # [H_p*W_p, D]

    pca  = PCA(n_components=3)
    rgb  = pca.fit_transform(hw_d)                     # [H_p*W_p, 3]
    for c in range(3):
        lo, hi = rgb[:, c].min(), rgb[:, c].max()
        rgb[:, c] = (rgb[:, c] - lo) / (hi - lo + 1e-8)

    return rgb.reshape(H_p, W_p, 3)


def cosine_sim_map(feat_map: torch.Tensor, query: torch.Tensor) -> np.ndarray:
    """
    feat_map: [1, D, H_p, W_p]
    query:    [1, D]
    → [H_p, W_p]  cosine similarity  [-1, 1]
    """
    import torch.nn.functional as F
    q = query[:, :, None, None].expand_as(feat_map)   # [1, D, H_p, W_p]
    sim = F.cosine_similarity(feat_map, q, dim=1)      # [1, H_p, W_p]
    return to_np(sim)                                  # [H_p, W_p]

# ── FG-pooled query 벡터 (실제 학습에서 사용하는 쿼리) ────────────────────────

# 마지막 레이어 feature로 pooled query 계산 (학습과 동일 조건)
last_b   = layer_indices[-1]
last_feat = feats[last_b]                              # [1, D, H_p, W_p]
q_fg     = pool_target_feature(last_feat, tgt_mask)   # [1, D]  ← FG mask 적용
q_all    = pool_target_feature(last_feat, None)        # [1, D]  ← mask 없음 (비교용)

sim_fg  = cosine_sim_map(last_feat, q_fg)
sim_all = cosine_sim_map(last_feat, q_all)

# ── 시각화 ────────────────────────────────────────────────────────────────────

n_layers = len(layer_indices)
# 패널 구성: 원본 | FG mask | PCA×n | sim(FG mask) | sim(no mask)
n_cols   = 2 + n_layers + 2
fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.5))

# ① 원본 이미지
orig_np = to_np(target_t.permute(0, 2, 3, 1))        # [H, W, 3]
axes[0].imshow(np.clip(orig_np, 0, 1))
axes[0].set_title(f"Target Image\n{H}×{W}", fontsize=9)
axes[0].axis("off")

# ② FG mask
mask_np = to_np(tgt_mask)                             # [H_p, W_p]
im = axes[1].imshow(mask_np, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title(f"FG Mask\n{fg_ratio*100:.1f}% FG patches", fontsize=9)
axes[1].axis("off")
fg_patch = mpatches.Patch(color="green", label="FG")
bg_patch = mpatches.Patch(color="red",   label="BG (gray)")
axes[1].legend(handles=[fg_patch, bg_patch], loc="lower right", fontsize=7)

# ③ PCA per layer
for k, b in enumerate(layer_indices):
    rgb = pca_rgb(feats[b])
    axes[2 + k].imshow(rgb, interpolation="nearest")
    axes[2 + k].set_title(f"Layer {b} PCA\n(3 PC → RGB)", fontsize=9)
    axes[2 + k].axis("off")

# ④ Cosine sim: FG-masked query
col = 2 + n_layers
im_fg = axes[col].imshow(sim_fg, cmap="hot", vmin=sim_fg.min(), vmax=sim_fg.max(),
                          interpolation="nearest")
axes[col].set_title(f"Layer {last_b} Cosine Sim\n(FG-masked query)", fontsize=9)
axes[col].axis("off")
plt.colorbar(im_fg, ax=axes[col], fraction=0.046, pad=0.04)

# FG 패치 경계선 표시
mask_overlay = to_np(tgt_mask).astype(bool)
for r in range(H_p):
    for c in range(W_p):
        if mask_overlay[r, c]:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  linewidth=0.5, edgecolor="cyan", facecolor="none")
            axes[col].add_patch(rect)

# ⑤ Cosine sim: 전체 평균 query (mask 없음 비교)
col2 = col + 1
im_all = axes[col2].imshow(sim_all, cmap="hot", vmin=sim_all.min(), vmax=sim_all.max(),
                            interpolation="nearest")
axes[col2].set_title(f"Layer {last_b} Cosine Sim\n(no mask, full avg)", fontsize=9)
axes[col2].axis("off")
plt.colorbar(im_all, ax=axes[col2], fraction=0.046, pad=0.04)

# ── 통계 출력 ──────────────────────────────────────────────────────────────────
print(f"\n[Cosine Sim  — FG-masked query vs all patches]")
print(f"  FG 패치 sim : mean={sim_fg[mask_overlay].mean():.4f}  "
      f"std={sim_fg[mask_overlay].std():.4f}  "
      f"max={sim_fg[mask_overlay].max():.4f}")
print(f"  BG 패치 sim : mean={sim_fg[~mask_overlay].mean():.4f}  "
      f"std={sim_fg[~mask_overlay].std():.4f}  "
      f"max={sim_fg[~mask_overlay].max():.4f}")
print(f"  FG-BG 차이  : {sim_fg[mask_overlay].mean() - sim_fg[~mask_overlay].mean():.4f}")

print(f"\n[Cosine Sim  — no-mask query vs all patches]")
print(f"  FG 패치 sim : mean={sim_all[mask_overlay].mean():.4f}")
print(f"  BG 패치 sim : mean={sim_all[~mask_overlay].mean():.4f}")
print(f"  FG-BG 차이  : {sim_all[mask_overlay].mean() - sim_all[~mask_overlay].mean():.4f}")

plt.suptitle(
    f"DINOv3 Target Feature Visualization  |  patch_size={patch_size}  |  grid={H_p}×{W_p}",
    fontsize=11, y=1.01,
)
plt.tight_layout()

if SAVE_PATH:
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] 저장 완료: {SAVE_PATH}")
else:
    plt.show()
