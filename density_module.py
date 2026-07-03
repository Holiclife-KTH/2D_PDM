import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import generic_filter

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output/rgb/000002_rgb.png"
K           = 8      # 클러스터 수
WINDOW      = 3      # 공간 엔트로피 로컬 윈도우 반경 (패치 단위)
USE_GMM     = False  # True=GMM, False=K-means
FG_THRESH   = 0.4    # PCA foreground threshold (0~1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m")
model     = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m").to(device)
patch_size        = model.config.patch_size
num_register_tokens = model.config.num_register_tokens


# ── Feature 추출 (mask_module과 동일) ─────────────────────────────────────────
def extract_patches(image_path):
    image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = image.size

    new_W = (W_orig // patch_size) * patch_size
    new_H = (H_orig // patch_size) * patch_size
    original = image.resize((new_W, new_H))

    inputs = processor(images=original, return_tensors="pt", do_resize=False).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    h, w = new_H // patch_size, new_W // patch_size

    patch_tokens = last_hidden[:, 1 + num_register_tokens:, :]
    patch_grid   = patch_tokens.unflatten(1, (h, w)).squeeze(0).cpu()  # [h, w, D]
    return patch_grid, original


# ── 군집화 ────────────────────────────────────────────────────────────────────
def cluster_patches(feats_np, k, use_gmm=False):
    """feats_np: [hw, D] → labels: [hw] int"""
    if use_gmm:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              max_iter=200, random_state=42)
        gmm.fit(feats_np)
        labels = gmm.predict(feats_np)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(feats_np)
    return labels  # [hw]


# ── 공간적 엔트로피 계산 ──────────────────────────────────────────────────────
def spatial_entropy(label_map, window, k):
    """
    label_map: [h, w] int
    각 패치의 로컬 윈도우(2*window+1 × 2*window+1) 안의
    클러스터 분포 엔트로피를 계산.
    """
    size = 2 * window + 1

    def _entropy(flat):
        counts = np.bincount(flat.astype(int), minlength=k).astype(float)
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    entropy_map = generic_filter(
        label_map.astype(float),
        function=_entropy,
        size=size,
        mode="nearest",
    )
    return entropy_map  # [h, w]


# ── PCA Foreground 마스크 ─────────────────────────────────────────────────────
def pca_foreground(feats_np, h, w, threshold):
    """첫 번째 PCA 성분으로 foreground/background 분리"""
    from sklearn.decomposition import PCA
    pca  = PCA(n_components=1)
    comp = pca.fit_transform(feats_np).reshape(h, w)  # [h, w]
    comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
    # 값이 threshold 이상이면 foreground, 이하면 background
    # 경우에 따라 반전이 필요할 수 있음
    fg = comp_norm > threshold
    if fg.mean() < 0.1:       # 너무 적으면 반전
        fg = ~fg
    return fg.astype(float)   # [h, w] 0 or 1


# ── 업샘플 유틸 ───────────────────────────────────────────────────────────────
def upsample(patch_map, patch_size):
    """[h, w] → [H, W] nearest-neighbor (kron)"""
    return np.kron(patch_map, np.ones((patch_size, patch_size)))


# ── 메인 ─────────────────────────────────────────────────────────────────────
patch_grid, original = extract_patches(IMAGE_PATH)
h, w, D = patch_grid.shape
print(f"[INFO] 패치 그리드: {h}×{w}  |  feature dim: {D}  |  K={K}  |  GMM={USE_GMM}")

feats     = F.normalize(patch_grid.reshape(h * w, D), p=2, dim=-1)
feats_np  = feats.numpy()  # [hw, D]

# 1. 군집화
labels     = cluster_patches(feats_np, k=K, use_gmm=USE_GMM)
label_map  = labels.reshape(h, w)  # [h, w]

# 2. 공간적 엔트로피
ent_map    = spatial_entropy(label_map, window=WINDOW, k=K)  # [h, w]
ent_norm   = (ent_map - ent_map.min()) / (ent_map.max() - ent_map.min() + 1e-8)

# 3. PCA foreground 마스크
fg_mask    = pca_foreground(feats_np, h, w, threshold=FG_THRESH)  # [h, w]

# 4. 밀도맵 = 엔트로피 × foreground (배경 억제)
density    = ent_norm * fg_mask                                    # [h, w]
density    = (density - density.min()) / (density.max() - density.min() + 1e-8)

# ── 업샘플 → 이미지 해상도 ────────────────────────────────────────────────────
base       = np.asarray(original).astype(np.float32) / 255.0      # [H, W, 3]

label_up   = upsample(label_map / (K - 1), patch_size)            # [H, W] 0~1
ent_up     = upsample(ent_norm,            patch_size)             # [H, W]
fg_up      = upsample(fg_mask,             patch_size)             # [H, W]
density_up = upsample(density,             patch_size)             # [H, W]

cmap_jet     = plt.get_cmap("jet")
cmap_viridis = plt.get_cmap("viridis")
cmap_tab     = plt.get_cmap("tab10")

overlay_density = 0.5 * base + 0.5 * cmap_jet(density_up)[..., :3]

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].imshow(original)
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(cmap_tab(label_up)[..., :3])
axes[0, 1].set_title(f"Cluster Map  (K={K}, {'GMM' if USE_GMM else 'K-means'})")

axes[0, 2].imshow(fg_up, cmap="gray")
axes[0, 2].set_title(f"PCA Foreground Mask  (thresh={FG_THRESH})")

axes[1, 0].imshow(cmap_viridis(ent_up)[..., :3])
axes[1, 0].set_title(f"Spatial Entropy  (window={WINDOW})")

axes[1, 1].imshow(cmap_jet(density_up)[..., :3])
axes[1, 1].set_title("Density Map  (Entropy × Foreground)")

axes[1, 2].imshow(np.clip(overlay_density, 0, 1))
axes[1, 2].set_title("Density Overlay")

for ax in axes.flat:
    ax.axis("off")

method = "GMM" if USE_GMM else "K-means"
plt.suptitle(
    f"DINOv3 Spatial Entropy Density  |  {method} K={K}  |  "
    f"window={WINDOW}  |  grid={h}×{w}",
    fontsize=13,
)
plt.tight_layout()
plt.show()
